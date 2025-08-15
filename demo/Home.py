import os, requests, streamlit as st

st.set_page_config(page_title="CV Lab — Home", page_icon="🤖", layout="wide")

st.markdown(
    """
<style>
:root { --accent:#6C63FF; --bg:#0f1116; --card:#121626; --muted:#9aa3c7; --green:#21ba45; }
.block-container { padding-top:1.2rem; padding-bottom:1rem; }
.cv-hero {
  background: linear-gradient(135deg,#181c2c 0%,#0f1116 60%);
  border-radius: 20px; padding: 24px 22px; border: 1px solid #242a3e;
  box-shadow: 0 8px 28px rgba(0,0,0,.35); color: #e9ecff;
}
.cv-badges span {
  display:inline-block; margin-right:10px; font-size:.85rem; color:#cfd4ff;
  padding:4px 10px; border-radius:999px; border:1px solid #3a4060; background:#1a1f33;
}
.cv-card {
  background: var(--card); border-radius: 16px; padding: 18px;
  border: 1px solid #232947; box-shadow: 0 6px 20px rgba(0,0,0,.25);
}
.cv-muted { color: var(--muted); font-size:.95rem; }
.cv-title { font-weight:700; font-size:1.1rem; margin-bottom:6px; }
.cv-chip { display:inline-block; padding:3px 10px; border-radius:999px; font-size:.78rem; border:1px solid #2b3152; background:#1a1f33; color:#cfd4ff; margin-right:6px; }
.cv-btn {
  display:inline-block; margin-top:10px; padding:8px 14px; border-radius:8px; font-weight:700;
  background: var(--green); color: #fff; text-decoration:none; cursor: default;
  box-shadow: 0 2px 0 rgba(0,0,0,.25);
}
.cv-meta { font-size:.85rem; color:#8d95b3; }
hr{ border: none; border-top: 1px solid #242a3e; margin: 1.2rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

API_URL = os.getenv("API_URL", "http://localhost:8080")

st.markdown(
    f"""
<div class="cv-hero">
  <h1 style="margin:0 0 8px 0;">🤖 Computer Vision Lab</h1>
  <div class="cv-badges">
    <span>FastAPI</span><span>Ultralytics YOLO</span><span>Streamlit</span><span>Vertex AI–ready</span>
  </div>
  <p class="cv-muted" style="margin-top:10px;">
    Welcome! This is a showcase of three core CV tasks — <b>Detection</b>, <b>Segmentation</b>, and <b>Classification</b> —
    plus a <b>Camera (Polling)</b> demo for quick local tests. Use this page to learn what each task does.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# display api status
st.subheader("API Status")
try:
    r = requests.get(f"{API_URL}/health", timeout=3)
    r.raise_for_status()
    data = r.json()
    st.success(
        f"Online · backend={data.get('model_backend','?')} · version={data.get('model_version','?')}"
    )
except Exception as e:
    st.error(f"API unreachable at {API_URL} — {e}")
    st.caption("Check your API container/process and the API_URL env var.")
st.divider()

c1, c2 = st.columns(2, gap="large")
c3, c4 = st.columns(2, gap="large")

with c1:
    st.markdown('<div class="cv-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="cv-title">📦 Object Detection</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="cv-muted">Find objects in an image and return bounding boxes with labels and confidences. Great for counting or localizing items (e.g., people, helmets, vehicles).</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="cv-chip">YOLOv8</span><span class="cv-chip">/predict</span><span class="cv-chip">/v1/models:predict</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="cv-meta">Response schema: <code>{"bboxes":[{x1,y1,x2,y2,conf,cls}]}</code></div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="cv-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="cv-title">🧩 Instance Segmentation</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="cv-muted">Detect objects and return precise pixel regions as polygons. Ideal when you need accurate shapes (e.g., defect areas, masks for blending/blur).</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="cv-chip">YOLOv8-seg</span><span class="cv-chip">/segment</span><span class="cv-chip">/v1/segment:predict</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="cv-meta">Response schema: <code>{"masks":[{"points":[[x,y],...], "cls","conf"}]}</code></div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown('<div class="cv-card">', unsafe_allow_html=True)
    st.markdown('<div class="cv-title">🏷️ Classification</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cv-muted">Assign a label to the whole image (e.g., dog vs. cat). Uses ImageNet-style classes by default; for scenes/selfies, consider zero-shot labels.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="cv-chip">YOLOv8-cls</span><span class="cv-chip">/classify</span><span class="cv-chip">/v1/classify:predict</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="cv-meta">Response schema: <code>{"topk":[["label", prob], ...]}</code></div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with c4:
    st.markdown('<div class="cv-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="cv-title">📷 Camera (Polling) Demo</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="cv-muted">Manual capture flow using <code>st.camera_input</code>. Start the camera, take a still, then send to the API for any task.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="cv-chip">Local demo</span><span class="cv-chip">No WebRTC</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="cv-meta">UI shows Original vs. Annotated side-by-side for detection/segmentation.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()
with st.expander("ℹ️ How to use the API (quick curl examples)"):
    st.code(
        f"""# Detection (multipart)
curl -s -X POST '{API_URL}/predict' \\
  -F 'file=@sample.jpg' | jq .

# Detection (base64, Vertex-style)
python - <<'PY'
import base64, json, requests
b64 = base64.b64encode(open("sample.jpg","rb").read()).decode("utf-8")
print(requests.post("{API_URL}/v1/models:predict", json={{"instances":[b64]}}).json())
PY

# Segmentation (multipart)
curl -s -X POST '{API_URL}/segment' \\
  -F 'file=@sample.jpg' | jq .

# Classification (multipart)
curl -s -X POST '{API_URL}/classify' \\
  -F 'file=@sample.jpg' | jq .
""",
        language="bash",
    )

st.caption(
    "This landing page is informational only. Use the sidebar Pages list to open each app, or keep this as a static intro for demos."
)
