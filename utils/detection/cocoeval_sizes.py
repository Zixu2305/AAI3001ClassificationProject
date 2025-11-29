# utils/detection/cocoeval_sizes.py
import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


AREA_INDEX = {"all": 0, "small": 1, "medium": 2, "large": 3}


def parse_iou_list(iou_arg: str) -> np.ndarray:
    """
    Accepts:
      - comma list: "0.50,0.60,0.75"
      - range step: "0.50:0.95:0.05"
      - single value: "0.75"
    Returns float32 numpy array sorted ascending.
    """
    s = iou_arg.strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) not in (2, 3):
            raise ValueError("IoU range format must be start:end[:step]")
        start = float(parts[0]); end = float(parts[1])
        step = float(parts[2]) if len(parts) == 3 else 0.05
        vals = np.arange(start, end + 1e-9, step)
    elif "," in s:
        vals = np.array([float(x) for x in s.split(",")], dtype=np.float32)
    else:
        vals = np.array([float(s)], dtype=np.float32)
    vals = np.clip(vals, 0.0, 1.0)
    vals.sort()
    return vals.astype(np.float32)


def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def maybe_shift_category_ids(preds: List[dict], shift: int) -> None:
    if shift == 0:
        return
    for p in preds:
        p["category_id"] = int(p["category_id"]) + int(shift)


def get_class_names(coco_gt: COCO) -> Tuple[List[int], List[str]]:
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    # Keep COCO's intrinsic order (id ascending)
    cats = sorted(cats, key=lambda c: c["id"])
    ids = [c["id"] for c in cats]
    names = [c.get("name", str(c["id"])) for c in cats]
    return ids, names


def eval_coco(gt_path: str, pred_path: str, area: str, ious: np.ndarray,
              shift_cid: int = 0, write_shifted: str = ""):
    # Load GT & predictions
    coco_gt = COCO(gt_path)
    preds = json.load(open(pred_path, "r"))
    if shift_cid != 0:
        maybe_shift_category_ids(preds, shift_cid)
        if write_shifted:
            with open(write_shifted, "w") as f:
                json.dump(preds, f)
            print(f"[info] wrote shifted predictions to: {write_shifted}")

    coco_dt = coco_gt.loadRes(preds)

    # COCOeval setup
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.params.iouThrs = ious
    ev.params.areaRng = [
        [0**2, 1e5**2],     # all
        [0**2, 32**2],      # small
        [32**2, 96**2],     # medium
        [96**2, 1e5**2],    # large
    ]
    ev.params.areaRngLbl = ["all", "small", "medium", "large"]

    ev.evaluate()
    ev.accumulate()

    return ev, coco_gt


def plot_map_vs_iou(ev: COCOeval, area: str, out_png: str):
    aidx = AREA_INDEX[area]
    prec = ev.eval["precision"]  # shape [T, R, K, A, M]
    # mean over recall, classes, maxDets
    ap_per_iou = np.nanmean(prec[:, :, :, [aidx], :], axis=(1, 2, 4)).flatten()
    ious = ev.params.iouThrs

    plt.figure(figsize=(6, 4))
    plt.plot(ious, ap_per_iou, marker="o")
    plt.xlabel("IoU threshold")
    plt.ylabel(f"AP ({area})")
    plt.title(f"mAP vs IoU ({area})")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[saved] {out_png}")

    return ious, ap_per_iou


def combined_pr_curves_for_all_ious(ev: COCOeval, coco_gt: COCO, area: str, out_dir: str):
    """
    For every IoU in ev.params.iouThrs, create ONE combined PR chart (all classes on the same
    plot) and a CSV listing AP per class at that IoU. No individual class charts are saved.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    aidx = AREA_INDEX[area]
    ious = ev.params.iouThrs       # [T]
    rec_thrs = ev.params.recThrs   # [R]
    prec = ev.eval["precision"]    # shape [T, R, K, A, M]

    cat_ids, class_names = get_class_names(coco_gt)
    K, R, T = len(class_names), len(rec_thrs), len(ious)

    for tidx in range(T):
        iou = float(ious[tidx])

        # gather PR per class at this IoU
        per_class_pr = np.full((K, R), np.nan, dtype=np.float32)
        per_class_ap = []

        for k in range(K):
            pr = prec[tidx, :, k, aidx, -1]  # [R], -1 where undefined
            valid = pr > -1
            per_class_pr[k, valid] = pr[valid]
            ap_cls = float(np.nanmean(pr[valid])) if valid.any() else float("nan")
            per_class_ap.append(ap_cls)

        # combined chart
        fig = plt.figure(figsize=(12, 8))
        for k, name in enumerate(class_names):
            prk = per_class_pr[k]
            valid = ~np.isnan(prk)
            if valid.any():
                plt.plot(rec_thrs[valid], prk[valid], linewidth=1.6,
                         label=f"{name} {per_class_ap[k]:.3f}")

        mean_pr = np.nanmean(per_class_pr, axis=0)
        valid_mean = ~np.isnan(mean_pr)
        if valid_mean.any():
            mean_ap = float(np.nanmean(mean_pr[valid_mean]))
            plt.plot(rec_thrs[valid_mean], mean_pr[valid_mean],
                     linewidth=4.0, color="blue",
                     label=f"all classes {mean_ap:.3f} mAP@{iou:.2f}")

        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
        plt.tight_layout()

        subdir = ensure_outdir(os.path.join(out_dir, f"pr_combined_IoU{iou:.2f}_{area}"))
        png_path = os.path.join(subdir, f"PR_combined_IoU{iou:.2f}_{area}.png")
        fig.savefig(png_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {png_path}")

        # CSV with AP per class at this IoU
        csv_path = os.path.join(subdir, f"per_class_AP_IoU{iou:.2f}_{area}.csv")
        with open(csv_path, "w") as f:
            f.write("category_id,class,AP_at_IoU,area\n")
            for cid, name, apv in zip(cat_ids, class_names, per_class_ap):
                f.write(f"{cid},{name},{apv:.6f},{area}\n")
        print(f"[saved] {csv_path}")


    # Write CSV
    with open(csv_path, "w") as f:
        f.write("category_id,class,AP_at_IoU,area\n")
        for cid, name, apv in zip(cat_ids, class_names, per_class_ap):
            f.write(f"{cid},{name},{apv:.6f},{area}\n")
    print(f"[saved] {csv_path}")


def main():
    ap = argparse.ArgumentParser("COCO eval + plots")
    ap.add_argument("--gt", required=True, help="Path to COCO GT (instances_val.json)")
    ap.add_argument("--pred", required=True, help="Path to detections (predictions.json)")
    ap.add_argument("--outdir", default="coco_eval_out", help="Directory to save plots/CSVs")
    ap.add_argument("--area", default="all", choices=["all", "small", "medium", "large"],
                    help="Area split for plots")
    ap.add_argument("--ious", default="0.50:0.95:0.05",
                    help='IoUs to evaluate for mAP-vs-IoU plot. '
                         'Formats: "0.50,0.60,0.75" or "0.50:0.95:0.05" or "0.75"')
    ap.add_argument("--pr_iou", default="0.50",
                    help='IoU to use for per-class PR curves (single value). Example: "0.50" or "0.75"')
    ap.add_argument("--shift_cid", type=int, default=0,
                    help="Shift prediction category_id by this integer BEFORE eval (e.g., -1, 0, +1)")
    ap.add_argument("--write_shifted", default="",
                    help="Optional path to write the shifted predictions JSON")
    args = ap.parse_args()

    outdir = ensure_outdir(args.outdir)
    ious = parse_iou_list(args.ious)
    pr_iou = float(parse_iou_list(args.pr_iou)[0])

    # Run evaluation
    ev, coco_gt = eval_coco(args.gt, args.pred, args.area, ious,
                            shift_cid=args.shift_cid, write_shifted=args.write_shifted)

    # Standard COCO summary (for sanity)
    ev.summarize()

    # Plot mAP vs IoU
    map_plot = os.path.join(outdir, f"mAP_vs_IoU_{args.area}.png")
    plot_map_vs_iou(ev, args.area, map_plot)

    # Per-class PR curves at chosen IoU
    combined_pr_curves_for_all_ious(
		ev, coco_gt, args.area,
		out_dir=os.path.join(outdir, "pr_curves_combined")
	)

    print("[done]")


if __name__ == "__main__":
    main()
