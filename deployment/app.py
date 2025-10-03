# -*- coding: utf-8 -*-
import os, json
from pathlib import Path
from datetime import datetime

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from infer import load_predict_fn

# ---- Page ----
APP_NAME = "Street-View Labeling Assistant"
APP_TAGLINE = "Draw boxes → classify street-view objects (Top-3) → export COCO JSON"

st.set_page_config(page_title=APP_NAME, layout="wide")
st.markdown("<style>.block-container{max-width:1800px;padding:1rem}</style>", unsafe_allow_html=True)

st.title(APP_NAME)
st.caption(APP_TAGLINE)

# ---- Utils ----
def list_model_dirs(root="deployment"):
    items = []
    for d in sorted(os.listdir(root)):
        if d.startswith(".") or d.startswith("__"):
            continue
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        if (os.path.exists(os.path.join(p, "model_config.json"))
            and os.path.exists(os.path.join(p, "class_map.json"))
            and os.path.exists(os.path.join(p, "weights", "best.pth"))):
            items.append(d)
    return items

def parse_boxes(json_data, scale, W, H):
    out = []
    if not json_data or "objects" not in json_data:
        return out
    for o in json_data.get("objects", []):
        if o.get("type") != "rect":
            continue
        x, y = o["left"], o["top"]
        w, h = o["width"], o["height"]
        x0 = int(round(x/scale)); y0 = int(round(y/scale))
        x1 = int(round((x+w)/scale)); y1 = int(round((y+h)/scale))
        x0 = max(0, min(W-1, x0)); y0 = max(0, min(H-1, y0))
        x1 = max(1, min(W,   x1)); y1 = max(1, min(H,   y1))
        if x1 > x0 and y1 > y0:
            out.append((x0, y0, x1, y1))
    return out

def xyxy_to_xywh(b):
    x0,y0,x1,y1 = b
    return [int(x0),int(y0),int(x1-x0),int(y1-y0)]

def export_one(item, class_to_id, export_root, model_meta):
    stem = Path(item["name"]).stem
    safe = "".join(c if (c.isalnum() or c in ("-","_")) else "_" for c in stem)
    out_dir = Path(export_root)/safe
    crops_dir = out_dir/"crops"
    out_dir.mkdir(parents=True, exist_ok=True); crops_dir.mkdir(parents=True, exist_ok=True)

    # save original once
    orig_path = out_dir/"original.jpg"
    if not orig_path.exists():
        item["pil"].save(orig_path, "JPEG")

    W,H = item["pil"].width, item["pil"].height
    boxes = item.get("boxes", [])
    preds = item.get("preds", [])
    user = item.get("user_labels", [None]*len(boxes))

    cats = [{"id":i,"name":n} for i,n in enumerate(CLASS_NAMES)]

    anns = []
    for i,b in enumerate(boxes,1):
        crop = item["pil"].crop(b)
        fn = f"crop_{i:03d}.jpg"
        crop.save(crops_dir/fn, "JPEG")
        top3 = [{"name":n,"prob":float(p)} for (n,p) in preds[i-1]] if i-1 < len(preds) else []
        chosen = user[i-1] or (top3[0]["name"] if top3 else "unlabeled")
        anns.append({
            "id": i,
            "bbox": xyxy_to_xywh(b),
            "category_id": class_to_id.get(chosen, -1),
            "category_name": chosen,
            "crop_file": str(Path("crops")/fn),
            "model_top3": top3
        })

    payload = {
        "model": model_meta,
        "image": {"file_name": item["name"], "width":W, "height":H,
                  "export_time": datetime.now().isoformat(timespec="seconds")},
        "categories": cats,
        "annotations": anns
    }
    with open(out_dir/"annotations.json","w",encoding="utf-8") as f:
        json.dump(payload,f,indent=2)
    return str(out_dir)

# ---- Auto layout (no user action required) ----
# Defaults for small laptop / built-in Mac screens
CANVAS_W, COLS = 820, "stacked"
try:
    # Optional helper: pip install streamlit-js-eval
    from streamlit_js_eval import get_page_info
    info = get_page_info() or {}
    w = int(info.get("clientWidth") or info.get("windowWidth") or 0)
    if w >= 1800:         # ultrawide
        CANVAS_W, COLS = 1200, [8, 4]
    elif w >= 1350:       # 1080p-ish
        CANVAS_W, COLS = 1000, [7, 5]
    else:                 # laptops / built-in Macs
        CANVAS_W, COLS = 820, "stacked"
except Exception:
    # Helper not available — keep safe defaults above
    pass

# ---- Sidebar ----
st.sidebar.title("Model")
models = list_model_dirs("deployment")
if not models:
    st.sidebar.error("No model folders found in ./deployment")
    st.stop()
m_choice = st.sidebar.selectbox("Choose a model", models, index=0)
MODEL_DIR = os.path.join("deployment", m_choice)

CFG_PATH  = os.path.join(MODEL_DIR, "model_config.json")
CKPT_PATH = os.path.join(MODEL_DIR, "weights", "best.pth")
CLS_PATH  = os.path.join(MODEL_DIR, "class_map.json")
EXPORT_ROOT = st.sidebar.text_input("Export folder", "exports")

@st.cache_resource(show_spinner=False)
def _load(cfg, ckpt, clsf):
    pred, class_names = load_predict_fn(cfg, ckpt, clsf)
    return pred, class_names

predict_fn, CLASS_NAMES = _load(CFG_PATH, CKPT_PATH, CLS_PATH)
CLASS_TO_ID = {n:i for i,n in enumerate(CLASS_NAMES)}
MODEL_META = {
    "id": m_choice,                      # the selected model folder name
    "dir": MODEL_DIR,                    # full path to model folder
    "config": os.path.basename(CFG_PATH),
    "weights": os.path.basename(CKPT_PATH),
    "classes": CLASS_NAMES,
}

# ---- Project info (auto from class_map.json) ----
st.sidebar.markdown("---")
st.sidebar.subheader("Project")
st.sidebar.markdown(
    "This Space is tuned for **street-view imagery** "
    "(roads, sidewalks, vehicles, signs, storefronts, street furniture, etc.)."
)

st.sidebar.markdown(f"**Classes supported**: {len(CLASS_NAMES)}")
with st.sidebar.expander("Show class list", expanded=False):
    # Render as a compact list of chips
    st.write(", ".join(CLASS_NAMES))

# ---- Session ----
if "images" not in st.session_state:
    st.session_state.images = {}  # key -> {name,pil,boxes,preds,user_labels,canvas_json}
if "active_key" not in st.session_state:
    st.session_state.active_key = None

# ---- Layout ----
st.title("Label Assistant - crop -> classify (Top-3)")
if COLS == "stacked":
    left = st.container()
    right = st.container()  # will render below 'left'
else:
    left, right = st.columns(COLS, gap="large")

with left:
    # reset
    if st.button("Clear ALL loaded images", key="reset_all", type="primary"):
        st.session_state.images = {}
        st.session_state.active_key = None
        st.experimental_rerun()

    st.subheader("1) Load image(s)")
    ups = st.file_uploader("Upload one or more images", type=["jpg","jpeg","png","bmp"], accept_multiple_files=True)
    for up in ups or []:
        key = f"{up.name}-{up.size}"
        if key not in st.session_state.images:
            img = Image.open(up).convert("RGB")
            st.session_state.images[key] = {"name": up.name, "pil": img,
                                            "boxes": [], "preds": [], "user_labels": [],
                                            "canvas_json": None}
            if st.session_state.active_key is None:
                st.session_state.active_key = key

    keys = list(st.session_state.images.keys())
    if keys:
        names = [st.session_state.images[k]["name"] for k in keys]
        idx = keys.index(st.session_state.active_key) if st.session_state.active_key in keys else 0
        chosen = st.selectbox("Active image", names, index=idx, key="active_select")
        ak = keys[names.index(chosen)]
        st.session_state.active_key = ak
        item = st.session_state.images[ak]
        base = item["pil"]

        # Fit to image but cap at CANVAS_W from auto layout
        disp_w = min(base.width, CANVAS_W)
        scale  = disp_w / base.width
        disp_h = int(base.height * scale)

        st.subheader("2) Draw one or more rectangles")
        canvas = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=3,
            stroke_color="#FF9900",
            background_image=base.resize((disp_w, disp_h)).copy(),
            update_streamlit=True,             # live; we only save on Lock
            height=disp_h,
            width=disp_w,
            drawing_mode="rect",
            key=f"canvas_{ak}",                # stable key per image
            initial_drawing=item.get("canvas_json") or {}
        )

        colA, colB, colC = st.columns(3)
        if colA.button("Save / Lock boxes", key=f"lock_{ak}"):
            if canvas.json_data and canvas.json_data.get("objects"):
                item["canvas_json"] = canvas.json_data
                item["boxes"] = parse_boxes(canvas.json_data, scale, base.width, base.height)
                item["user_labels"] = [None] * len(item["boxes"])

        if colB.button("Clear boxes (THIS image)", key=f"clear_{ak}"):
            item["boxes"] = []
            item["preds"] = []
            item["user_labels"] = []
            item["canvas_json"] = None
            st.experimental_rerun()

        if colC.button("Remove THIS image", key=f"remove_{ak}"):
            del st.session_state.images[ak]
            st.session_state.active_key = next(iter(st.session_state.images), None)
            st.experimental_rerun()

        # crops from saved boxes
        crops = [base.crop(b) for b in item.get("boxes", [])]
        st.write(f"Crops in this image: {len(crops)}")

        # run buttons (enabled if boxes exist)
        b1, b2 = st.columns(2)
        can_run = len(item.get("boxes", [])) > 0
        if b1.button("Run THIS image", key=f"run_this_{ak}", disabled=not can_run):
            item["preds"] = predict_fn(crops, topk=3)
        if b2.button("Run ALL images", key="run_all", disabled=(not st.session_state.images)):
            for kk, vv in st.session_state.images.items():
                if vv.get("boxes"):
                    cs = [vv["pil"].crop(b) for b in vv["boxes"]]
                    vv["preds"] = predict_fn(cs, topk=3)

# --- Right pane renderer ---
def render_right_pane():
    st.subheader("Predictions and labeling")
    st.caption(f"Street-view classes ({len(CLASS_NAMES)}): " + ", ".join(CLASS_NAMES))
    if not st.session_state.images:
        return
    by_name = sorted(st.session_state.images.items(), key=lambda kv: kv[1]["name"])
    tabs = st.tabs([v["name"] for _, v in by_name])
    for (k, v), tab in zip(by_name, tabs):
        with tab:
            if not v.get("boxes"):
                st.info("No boxes drawn.")
                continue

            if "user_labels" not in v or len(v["user_labels"]) != len(v["boxes"]):
                v["user_labels"] = [None] * len(v["boxes"])

            for i, b in enumerate(v["boxes"], start=1):
                crop = v["pil"].crop(b)
                cA, cB = st.columns([1,2])
                with cA:
                    st.image(crop, caption=f"Crop {i}: {b}", width=180)
                with cB:
                    # default to model top1 if available
                    default = v["preds"][i-1][0][0] if (i-1 < len(v.get("preds", [])) and v["preds"]) else None
                    current = v["user_labels"][i-1] or default
                    try:
                        idx = CLASS_NAMES.index(current) if current in CLASS_NAMES else 0
                    except Exception:
                        idx = 0
                    v["user_labels"][i-1] = st.selectbox(
                        f"Label for crop {i}", CLASS_NAMES, index=idx, key=f"sel_{k}_{i}"
                    )
                    if i-1 < len(v.get("preds", [])) and v["preds"]:
                        st.write("Model Top-3:")
                        for name, prob in v["preds"][i-1]:
                            st.write(f"* {name}: {prob:.3f}")

            st.divider()
            c1, c2 = st.columns(2)
            if c1.button("Export THIS image", key=f"exp1_{k}"):
                out_dir = export_one(v, CLASS_TO_ID, EXPORT_ROOT, MODEL_META)
                st.success(f"Exported to: {out_dir}")
            if c2.button("Export ALL images", key=f"expall_{k}"):
                n = 0
                for kk, vv in st.session_state.images.items():
                    if vv.get("boxes"):
                        export_one(vv, CLASS_TO_ID, EXPORT_ROOT, MODEL_META); n += 1
                if n:
                    st.success(f"Exported {n} image folder(s) to: {EXPORT_ROOT}")
                else:
                    st.info("No images with boxes to export.")

# Render right pane (below on stacked; side-by-side on split)
with right:
    render_right_pane()
