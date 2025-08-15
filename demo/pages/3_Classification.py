import base64
import io
import os

import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Classification", layout="wide")
st.header("🏷️ Image Classification")

API_URL = os.getenv("API_URL", "http://localhost:8080")
TOPK = 5

tab_upload, tab_camera = st.tabs(["Image Upload", "Camera (manual)"])


def render_probs(topk):
    st.write("**Top predictions:**")
    for lbl, p in topk:
        st.progress(min(max(float(p), 0.0), 1.0), text=f"{lbl} — {float(p):.2%}")


# Upload
with tab_upload:
    st.subheader("Upload → /classify")
    up = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if up and st.button("Classify", type="primary"):
        orig = Image.open(up).convert("RGB")
        buf = io.BytesIO()
        orig.save(buf, format="JPEG", quality=92)
        r = requests.post(
            f"{API_URL}/classify",
            files={"file": ("upload.jpg", buf.getvalue(), "image/jpeg")},
            timeout=60,
        )
        r.raise_for_status()
        pred = r.json()  # {"topk":[["label", prob], ...]}
        col1, col2 = st.columns([0.45, 0.55])
        col1.image(orig, caption="Image", use_column_width=True)
        with col2:
            render_probs(pred.get("topk", []))
        with st.expander("Raw JSON"):
            st.json(pred)

# Camera (manual)
with tab_camera:
    st.subheader("Camera → /v1/classify:predict")
    if "cls_cam_on" not in st.session_state:
        st.session_state.cls_cam_on = False
    if "cls_cam_key" not in st.session_state:
        st.session_state.cls_cam_key = 0

    label = "⏹ Stop camera" if st.session_state.cls_cam_on else "▶ Start camera"
    if st.button(label, type="primary", key="cls_toggle"):
        st.session_state.cls_cam_on = not st.session_state.cls_cam_on
        if st.session_state.cls_cam_on:
            st.session_state.cls_cam_key += 1

    cam_slot = st.empty()
    result = st.container()

    if st.session_state.cls_cam_on:
        frame = cam_slot.camera_input(
            "Camera", key=f"cls-{st.session_state.cls_cam_key}"
        )
        if frame and st.button("Analyze last photo", key="cls_analyze"):
            orig = Image.open(frame).convert("RGB")
            buf = io.BytesIO()
            orig.save(buf, format="JPEG", quality=90)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            r = requests.post(
                f"{API_URL}/v1/classify:predict", json={"instances": [b64]}, timeout=15
            )
            r.raise_for_status()
            pred = (r.json().get("predictions") or [{}])[0]
            col1, col2 = result.columns([0.45, 0.55])
            col1.image(orig, caption="Image", use_column_width=True)
            with col2:
                render_probs(pred.get("topk", []))
            with st.expander("Raw JSON"):
                st.json(pred)
    else:
        cam_slot.empty()
