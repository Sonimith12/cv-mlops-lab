# demo/pages/0_Camera_Polling.py
import os, io, base64, requests, streamlit as st
from PIL import Image
from ui_utils import draw_boxes, overlay_masks

st.set_page_config(page_title="Camera (Polling Hub)", layout="wide")
st.header("📷 Camera (Polling) — Task Hub")

API_URL = os.getenv("API_URL", "http://localhost:8080")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.35"))

task = st.selectbox("Task", ["Detection", "Segmentation", "Classification"])

key_prefix = {"Detection":"det", "Segmentation":"seg", "Classification":"cls"}[task]
on_key  = f"{key_prefix}_cam_on"
seed_key = f"{key_prefix}_cam_key"
if on_key not in st.session_state: st.session_state[on_key] = False
if seed_key not in st.session_state: st.session_state[seed_key] = 0

label = "⏹ Stop camera" if st.session_state[on_key] else "▶ Start camera"
if st.button(label, type="primary"):
    st.session_state[on_key] = not st.session_state[on_key]
    if st.session_state[on_key]:
        st.session_state[seed_key] += 1  # force fresh widget instance

cam_slot   = st.empty()
result_box = st.container()

if task == "Detection":
    endpoint = "/v1/models:predict"
elif task == "Segmentation":
    endpoint = "/v1/segment:predict"
else:
    endpoint = "/v1/classify:predict"

if st.session_state[on_key]:
    frame = cam_slot.camera_input("Camera", key=f"{key_prefix}-{st.session_state[seed_key]}")
    analyze = st.button("Analyze last photo", key=f"{key_prefix}_analyze")

    if frame and analyze:
        orig = Image.open(frame).convert("RGB")
        buf = io.BytesIO(); orig.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        try:
            r = requests.post(f"{API_URL}{endpoint}", json={"instances":[b64]}, timeout=15)
            r.raise_for_status()
            payload = r.json()
            pred = (payload.get("predictions") or [{}])[0] if "predictions" in payload else payload
        except Exception as e:
            st.error(f"API error: {e}")
            pred = {}

        if task == "Detection":
            ann = draw_boxes(orig, pred, CONF_THRESHOLD)
            c1, c2 = result_box.columns(2)
            c1.image(orig, caption="Original", use_column_width=True)
            c2.image(ann, caption="Annotated (boxes)", use_column_width=True)
        elif task == "Segmentation":
            ann = overlay_masks(orig, pred.get("masks", []), alpha=0.45)
            c1, c2 = result_box.columns(2)
            c1.image(orig, caption="Original", use_column_width=True)
            c2.image(ann, caption="Masks overlay", use_column_width=True)
        else:  # Classification
            c1, c2 = result_box.columns([0.45, 0.55])
            c1.image(orig, caption="Image", use_column_width=True)
            with c2:
                st.write("**Top predictions:**")
                for lbl, p in pred.get("topk", []):
                    st.progress(min(max(float(p),0.0),1.0), text=f"{lbl} — {float(p):.2%}")

        with st.expander("Raw JSON"):
            st.json(pred)
else:
    cam_slot.empty()
