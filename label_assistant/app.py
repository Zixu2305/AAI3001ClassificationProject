import os, io, json, time, hashlib
from typing import List, Dict
import streamlit as st
import pandas as pd
from PIL import Image
from infer_loader import load_predict_fn, load_labels, crop_from_bbox

# ---------- Config ----------
DEFAULT_LABELS_PATH = "class_map.json"    # keep this file alongside app.py
DEFAULT_CKPT_PATH   = "best.pth"          # or pull from HF Hub (see infer_loader.py)
MANIFEST_PATH       = "manifest.csv"      # local CSV; on Spaces prefer a dataset repo
TOP_K               = 3
TAU                 = 0.60    # abstain threshold (UI hint only)
TAU_HI              = 0.85    # (optional) auto-accept threshold

st.set_page_config(page_title="Label Assistant", layout="wide")
st.title("Project 1 — Label Assistant (Top-3 per crop)")

# ---------- Load model + labels ----------
labels = load_labels(DEFAULT_LABELS_PATH)
predict = load_predict_fn(ckpt_path=DEFAULT_CKPT_PATH, labels=labels)

# ---------- Sidebar ----------
st.sidebar.header("Options")
top_k = st.sidebar.slider("Top-K", 1, min(5, len(labels)), TOP_K)
tau   = st.sidebar.slider("Abstain threshold (τ)", 0.0, 1.0, TAU, 0.01)
tau_hi= st.sidebar.slider("Auto-accept (τ_hi)", 0.0, 1.0, TAU_HI, 0.01)

# ---------- Main UI ----------
img_file = st.file_uploader("Upload image", type=["jpg","jpeg","png","bmp","webp"])
bbox_json = st.text_area(
    "Paste bounding boxes JSON",
    placeholder='[{"id":"b1","x1":120,"y1":160,"x2":360,"y2":420}]'
)

if img_file:
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption=f"Image: {getattr(img_file,'name','uploaded')}", use_column_width=True)

    if st.button("Predict on all boxes") and bbox_json.strip():
        try:
            bboxes: List[Dict] = json.loads(bbox_json)
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
            st.stop()

        # Prepare crops
        crops, metas = [], []
        for b in bboxes:
            try:
                crop = crop_from_bbox(image, b)
                crops.append(crop)
                metas.append(b)
            except Exception as e:
                st.warning(f"Skipping bbox {b}: {e}")

        if not crops:
            st.warning("No valid boxes.")
            st.stop()

        # Predict in batch
        preds = predict(crops, top_k=top_k)

        # Render results per crop
        st.subheader("Review predictions")
        saved_rows = []
        for crop, meta, pr in zip(crops, metas, preds):
            c1, c2 = st.columns([1,2])
            with c1:
                st.image(crop, caption=f"crop {meta.get('id','?')}", use_column_width=True)
            with c2:
                top1_label = pr["top1"]["label"]
                top1_prob  = pr["top1"]["prob"]
                abstain    = top1_prob < tau

                st.markdown("**Top-K**")
                for t in pr["topk"]:
                    st.progress(min(1.0, t["prob"]))
                    st.write(f"{t['label']} — {t['prob']:.3f}")

                st.caption(f"Top-1 = {top1_label} ({top1_prob:.3f})  |  "
                           f"{'ABSTAIN (τ)' if abstain else 'OK'}")

                options = [top1_label] + [t["label"] for t in pr["topk"][1:]] + ["Unsure"]
                choice = st.selectbox("Confirm label", options=options, key=f"sel_{meta.get('id','?')}")

                # Save button per crop
                if st.button(f"Save {meta.get('id','?')}", key=f"save_{meta.get('id','?')}"):
                    row = {
                        "image_id": getattr(img_file, "name", "uploaded"),
                        "crop_id": meta.get("id"),
                        "x1": meta.get("x1"), "y1": meta.get("y1"),
                        "x2": meta.get("x2"), "y2": meta.get("y2"),
                        "w": (meta.get("x2") - meta.get("x1")) if None not in (meta.get("x1"), meta.get("x2")) else None,
                        "h": (meta.get("y2") - meta.get("y1")) if None not in (meta.get("y1"), meta.get("y2")) else None,
                        "pred_top1": top1_label, "pred_prob": top1_prob,
                        "topk_json": json.dumps(pr["topk"]),
                        "final_label": None if choice == "Unsure" else choice,
                        "action": "unsure" if choice == "Unsure" else ("accept" if choice==top1_label else "override"),
                        "model_version": "v1",
                        "reviewer": os.getenv("USER","local_user"),
                        "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    # append to CSV (create header if new)
                    header = not os.path.exists(MANIFEST_PATH)
                    pd.DataFrame([row]).to_csv(MANIFEST_PATH, mode="a", index=False, header=header)
                    st.success(f"Saved {meta.get('id','?')} → {row['final_label'] or 'Unsure'}")

        st.info(f"Annotations are appended to {MANIFEST_PATH}.")
else:
    st.info("Upload an image to begin.")
