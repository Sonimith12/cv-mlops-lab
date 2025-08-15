import os, io, base64, requests, streamlit as st
from PIL import Image
from ui_utils import overlay_masks

st.set_page_config(page_title="Segmentation", layout="wide")
st.header("🧩 Instance Segmentation")

API_URL = os.getenv("API_URL", "http://localhost:8080")

tab_upload, tab_camera = st.tabs(["Image Upload", "Camera (manual)"])

# Upload
with tab_upload:
    st.subheader("Upload → /segment")
    up = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if up and st.button("Run segmentation", type="primary"):
        orig = Image.open(up).convert("RGB")
        buf = io.BytesIO()
        orig.save(buf, format="JPEG", quality=92)
        r = requests.post(
            f"{API_URL}/segment",
            files={"file": ("upload.jpg", buf.getvalue(), "image/jpeg")},
            timeout=60,
        )
        r.raise_for_status()
        pred = r.json()  # expect {"masks":[...]}
        ann = overlay_masks(orig, pred.get("masks", []), alpha=0.45)
        c1, c2 = st.columns(2)
        c1.image(orig, caption="Original", use_column_width=True)
        c2.image(ann, caption="Masks overlay", use_column_width=True)
        with st.expander("Raw JSON"):
            st.json(pred)

with tab_camera:
    st.subheader("Camera → /v1/segment:predict")
    if "seg_cam_on" not in st.session_state:
        st.session_state.seg_cam_on = False
    if "seg_cam_key" not in st.session_state:
        st.session_state.seg_cam_key = 0

    label = "⏹ Stop camera" if st.session_state.seg_cam_on else "▶ Start camera"
    if st.button(label, type="primary", key="seg_toggle"):
        st.session_state.seg_cam_on = not st.session_state.seg_cam_on
        if st.session_state.seg_cam_on:
            st.session_state.seg_cam_key += 1

    cam_slot = st.empty()
    result = st.container()

    if st.session_state.seg_cam_on:
        frame = cam_slot.camera_input(
            "Camera", key=f"seg-{st.session_state.seg_cam_key}"
        )
        if frame and st.button("Analyze last photo", key="seg_analyze"):
            orig = Image.open(frame).convert("RGB")
            buf = io.BytesIO()
            orig.save(buf, format="JPEG", quality=90)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            r = requests.post(
                f"{API_URL}/v1/segment:predict", json={"instances": [b64]}, timeout=15
            )
            r.raise_for_status()
            pred = (r.json().get("predictions") or [{}])[0]
            ann = overlay_masks(orig, pred.get("masks", []), alpha=0.45)
            c1, c2 = result.columns(2)
            c1.image(orig, caption="Original", use_column_width=True)
            c2.image(ann, caption="Masks overlay", use_column_width=True)
            with st.expander("Raw JSON"):
                st.json(pred)
    else:
        cam_slot.empty()
