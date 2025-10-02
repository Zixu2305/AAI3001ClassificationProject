# -*- coding: utf-8 -*-
import os
import json
from pathlib import Path
from datetime import datetime

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from infer import load_predict_fn

# ---------- Page & CSS ----------
st.set_page_config(page_title="Label Assistant - crop -> classify (Top-3)", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1800px; padding-top: 1rem; padding-left: 1rem; padding-right: 1rem;}
</style>
""", unsafe_allow_html=True)

# ---------- Constants / helpers ----------
MODELS_ROOT = "deployment"

def list_model_dirs(root=MODELS_ROOT):
    out = []
    for d in sorted(os.listdir(root)):
        if d.startswith(".") or d.startswith("__"):
            continue
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        if (os.path.exists(os.path.join(p, "model_config.json")) and
            os.path.exists(os.path.join(p, "class_map.json")) and
            os.path.exists(os.path.join(p, "weights", "best.pth"))):
            out.append(d)
    return out

def xyxy_to_xywh(b):
    x0, y0, x1, y1 = b
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]

def safe_stem(name: str):
    stem = Path(name).stem
    return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in stem)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def export_one_image(item, class_to_id: dict, export_root: str):
    """
    item: {name, pil, boxes, preds, user_labels, canvas_json}
    Writes:
      exports/<stem>/
        original.jpg
        crops/crop_XXX.jpg
        annotations.json
    """
    stem = safe_stem(item["name"])
    img_dir = Path(export_root) / stem
    crops_dir = img_dir / "crops"
    ensure_dir(crops_dir)

    # save original once
    orig_path = img_dir / "original.jpg"
    if not orig_path.exists():
        item["pil"].save(orig_path, "JPEG")

    W, H = item["pil"].width, item["pil"].height
    boxes = item.get("boxes", []) or []
    user_labels = item.get("user_labels", [None] * len(boxes))
    preds = item.get("preds", []) or []

    categories = [{"id": cid, "name": cname} for cname, cid in class_to_id.items()]
    categories = sorted(categories, key=lambda x: x["id"])

    annotations = []
    for i, b in enumerate(boxes, start=1):
        xywh = xyxy_to_xywh(b)
        crop = item["pil"].crop(b)
        crop_name = f"crop_{i:03d}.jpg"
        crop_path = crops_dir / crop_name
        crop.save(crop_path, "JPEG")

        chosen_name = user_labels[i-1]
        model_top3 = []
        if i-1 < len(preds) and preds:
            model_top3 = [{"name": n, "prob": float(p)} for n, p in preds[i-1]]
            if chosen_name is None and model_top3:
                chosen_name = model_top3[0]["name"]
        if chosen_name is None:
            chosen_name = "unlabeled"

        cat_id = int(class_to_id.get(chosen_name, -1))

        annotations.append({
            "id": i,
            "bbox": xywh,                       # COCO [x,y,w,h]
            "category_id": cat_id,
            "category_name": chosen_name,
            "crop_file": str(Path("crops") / crop_name),
            "model_top3": model_top3
        })

    out = {
        "image": {
            "file_name": item["name"],
            "width": W,
            "height": H,
            "export_time": datetime.now().isoformat(timespec="seconds")
        },
        "categories": categories,
        "annotations": annotations
    }
    with open(img_dir / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return str(img_dir)

def parse_boxes_from_json(json_data, scale, W, H):
    out = []
    if not json_data or "objects" not in json_data:
        return out
    for obj in json_data.get("objects", []):
        if obj.get("type") != "rect":
            continue
        x, y = obj["left"], obj["top"]
        w, h = obj["width"], obj["height"]
        X0 = int(round(x / scale)); Y0 = int(round(y / scale))
        X1 = int(round((x + w) / scale)); Y1 = int(round((y + h) / scale))
        X0 = max(0, min(W-1, X0));  Y0 = max(0, min(H-1, Y0))
        X1 = max(1, min(W,   X1));  Y1 = max(1, min(H,   Y1))
        if X1 > X0 and Y1 > Y0:
            out.append((X0, Y0, X1, Y1))
    return out

def snapshot_boxes_from_live_canvas_if_needed(item, canvas_json, scale, W, H):
    """If user didn't Lock but canvas has shapes, snapshot now."""
    if item.get("boxes"):
        return False
    if canvas_json and "objects" in canvas_json and canvas_json["objects"]:
        item["canvas_json"] = canvas_json
        new_boxes = parse_boxes_from_json(canvas_json, scale, W, H)
        if new_boxes:
            item["boxes"] = new_boxes
            item["user_labels"] = [None] * len(new_boxes)
            return True
    return False

# ---------- Sidebar: model + export root ----------
st.sidebar.title("Model")
MODEL_CHOICE = st.sidebar.selectbox("Choose a model", list_model_dirs(), index=0)
MODEL_DIR = os.path.join(MODELS_ROOT, MODEL_CHOICE)

CFG_PATH  = os.path.join(MODEL_DIR, "model_config.json")
CKPT_PATH = os.path.join(MODEL_DIR, "weights", "best.pth")
CLS_PATH  = os.path.join(MODEL_DIR, "class_map.json")

EXPORT_ROOT = st.sidebar.text_input("Export folder", value="exports")

@st.cache_resource(show_spinner=False)
def load_model(cfg_path, ckpt_path, cls_path):
    pred, class_names = load_predict_fn(cfg_path, ckpt_path, cls_path)
    return pred, class_names

predict_fn, CLASS_NAMES = load_model(CFG_PATH, CKPT_PATH, CLS_PATH)

# ---------- Session state ----------
if "images" not in st.session_state:
    st.session_state.images = {}  # key -> {name,pil,boxes,preds,user_labels,canvas_json}
if "active_key" not in st.session_state:
    st.session_state.active_key = None
# Fix B tokens
if "canvas_tokens" not in st.session_state:
    st.session_state.canvas_tokens = {}  # {image_key: int}
if "_prev_active_key" not in st.session_state:
    st.session_state._prev_active_key = None

# ---------- Layout ----------
st.title("Label Assistant - crop -> classify (Top-3)")
left, right = st.columns([7, 5], gap="large")

with left:
    # Reset all
    if st.button("Clear ALL loaded images", type="primary"):
        st.session_state.images = {}
        st.session_state.active_key = None
        st.session_state.canvas_tokens = {}
        st.session_state._prev_active_key = None
        st.experimental_rerun()

    st.subheader("1) Load image(s)")
    ups = st.file_uploader(
        "Upload one or more images",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True
    )

    # Ingest uploads once
    for up in ups or []:
        key = f"{up.name}-{up.size}"
        if key not in st.session_state.images:
            pil = Image.open(up).convert("RGB")
            st.session_state.images[key] = {
                "name": up.name,
                "pil": pil,
                "boxes": [],
                "preds": [],
                "user_labels": [],
                "canvas_json": None
            }
            if st.session_state.active_key is None:
                st.session_state.active_key = key

    keys = list(st.session_state.images.keys())
    if keys:
        names = [st.session_state.images[k]["name"] for k in keys]
        sel_idx = keys.index(st.session_state.active_key) if st.session_state.active_key in keys else 0
        chosen = st.selectbox("Active image (for drawing):", names, index=sel_idx)
        st.session_state.active_key = keys[names.index(chosen)]
        item = st.session_state.images[st.session_state.active_key]
        base_img = item["pil"]

        # Fix B: per-image token, but don't bump on very first show
        ak = st.session_state.active_key
        if ak not in st.session_state.canvas_tokens:
            st.session_state.canvas_tokens[ak] = 0
        prev = st.session_state._prev_active_key
        if (prev is not None) and (prev != ak):
            st.session_state.canvas_tokens[ak] += 1  # remount only on real switch
        st.session_state._prev_active_key = ak
        canvas_token = st.session_state.canvas_tokens[ak]

        # Canvas width control
        canvas_w = st.slider("Canvas width (px)", 600, 1400, 900, 10, key=f"cw_{ak}")
        disp_w = min(canvas_w, base_img.width)
        scale  = disp_w / base_img.width
        disp_h = int(base_img.height * scale)

        # Preview original
        st.image(base_img.resize((disp_w, disp_h)), caption="Original image")

        # 2) Draw boxes (live canvas; we snapshot on Save/Lock or before Run)
        st.subheader("2) Draw one or more rectangles")
        canvas = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=3,
            stroke_color="#FF9900",
            background_image=base_img.resize((disp_w, disp_h)).copy(),
            update_streamlit=True,   # live updates so we can snapshot if needed
            height=disp_h,
            width=disp_w,
            drawing_mode="rect",
            key=f"canvas_{ak}_{canvas_token}",
            initial_drawing=item.get("canvas_json") or {}
        )

        c_lock, c_clear, c_remove = st.columns([1, 1, 1])
        lock_now  = c_lock.button("Save / Lock boxes")
        clear_it  = c_clear.button("Clear boxes (THIS image)")
        remove    = c_remove.button("Remove THIS image")

        # Save/Lock: snapshot current canvas json_data and derive boxes
        if lock_now:
            item["canvas_json"] = canvas.json_data
            new_boxes = parse_boxes_from_json(canvas.json_data, scale, base_img.width, base_img.height)
            if new_boxes:
                item["boxes"] = new_boxes
                item["user_labels"] = [None] * len(item["boxes"])
            # Force a clean remount to show saved shapes
            st.session_state.canvas_tokens[ak] += 1

        # Clear current image boxes only
        if clear_it:
            item["boxes"] = []
            item["preds"] = []
            item["user_labels"] = []
            item["canvas_json"] = None
            st.experimental_rerun()

        # Remove current image
        if remove:
            del st.session_state.images[ak]
            st.session_state.active_key = next(iter(st.session_state.images), None)
            st.experimental_rerun()

        # Build crops from saved boxes (not the live canvas) so it's stable
        boxes = item.get("boxes", [])
        crops = [base_img.crop(b) for b in boxes]
        st.write(f"Crops in this image: {len(crops)}")

        # Predict buttons
        c1, c2 = st.columns([1, 1])
        run_this = c1.button("Run THIS image")
        run_all  = c2.button("Run ALL images", disabled=(not st.session_state.images))

        # Run THIS image (auto-snapshot if user forgot to lock)
        if run_this:
            snapshot_boxes_from_live_canvas_if_needed(
                item, canvas.json_data, scale, base_img.width, base_img.height
            )
            crops = [base_img.crop(b) for b in item.get("boxes", [])]
            if crops:
                item["preds"] = predict_fn(crops, topk=3)

        # Run ALL images (active image may need snapshot)
        if run_all:
            snapshot_boxes_from_live_canvas_if_needed(
                item, canvas.json_data, scale, base_img.width, base_img.height
            )
            for kk, vv in st.session_state.images.items():
                if vv.get("boxes"):
                    pil = vv["pil"]
                    cs  = [pil.crop(b) for b in vv["boxes"]]
                    vv["preds"] = predict_fn(cs, topk=3)

with right:
    st.subheader("Predictions and labeling")
    if st.session_state.images:
        # stable ordering by filename
        by_name = sorted(st.session_state.images.items(), key=lambda kv: kv[1]["name"])
        tabs = st.tabs([v["name"] for _, v in by_name])
        for (k, v), tab in zip(by_name, tabs):
            with tab:
                if not v.get("boxes"):
                    st.info("No boxes drawn for this image.")
                    continue

                # ensure storage for user choices
                if "user_labels" not in v or len(v["user_labels"]) != len(v["boxes"]):
                    v["user_labels"] = [None] * len(v["boxes"])

                st.caption("Select a label for each crop (defaults to model Top-1 if present).")
                for i, b in enumerate(v["boxes"], start=1):
                    crop = v["pil"].crop(b)
                    cA, cB = st.columns([1, 2])
                    with cA:
                        st.image(crop, caption=f"Crop {i}: {b}", width=180)
                    with cB:
                        # default = model top-1 if available
                        default_name = None
                        if i-1 < len(v.get("preds", [])) and v["preds"]:
                            default_name = v["preds"][i-1][0][0]
                        current = v["user_labels"][i-1] or default_name
                        try:
                            default_idx = CLASS_NAMES.index(current) if current in CLASS_NAMES else 0
                        except Exception:
                            default_idx = 0
                        v["user_labels"][i-1] = st.selectbox(
                            f"Label for crop {i}",
                            CLASS_NAMES,
                            index=default_idx,
                            key=f"sel_{k}_{i}"
                        )
                        # Show model Top-3
                        if i-1 < len(v.get("preds", [])) and v["preds"]:
                            tops = v["preds"][i-1]
                            st.write("Model Top-3:")
                            for name, prob in tops:
                                st.write(f"* {name}: {prob:.3f}")

                st.divider()
                c1, c2 = st.columns([1, 1])
                if c1.button("Export THIS image", key=f"exp1_{k}"):
                    class_to_id = {name: idx for idx, name in enumerate(CLASS_NAMES)}
                    out_dir = export_one_image(v, class_to_id, EXPORT_ROOT)
                    st.success(f"Exported to: {out_dir}")
                if c2.button("Export ALL images", key=f"expall_{k}"):
                    class_to_id = {name: idx for idx, name in enumerate(CLASS_NAMES)}
                    count = 0
                    for kk, vv in st.session_state.images.items():
                        if vv.get("boxes"):
                            export_one_image(vv, class_to_id, EXPORT_ROOT)
                            count += 1
                    if count:
                        st.success(f"Exported {count} image folder(s) to: {EXPORT_ROOT}")
                    else:
                        st.info("No images with boxes to export.")
