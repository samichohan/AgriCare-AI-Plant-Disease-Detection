import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# ----------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AgriCare AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "final_model.keras"
CLASSES_PATH = BASE_DIR / "model" / "class_names.json"
CONF_THRESHOLD = 0.60   # below this the result is shown as uncertain


@st.cache_resource(show_spinner="Loading model...")
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)
    return model, classes


model, class_names = load_model_and_classes()
IMG_SIZE = tuple(model.input_shape[1:3])


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def pretty(name: str):
    """'Tomato___Late_blight' -> ('Tomato', 'Late blight', is_healthy)"""
    plant, disease = name.split("___")
    plant = plant.replace("_", " ").strip()
    disease = disease.replace("_", " ").strip()
    return plant, disease, disease.lower() == "healthy"


def preprocess(img: Image.Image) -> np.ndarray:
    img = ImageOps.exif_transpose(img).convert("RGB")      # fix phone rotation, drop alpha
    arr = np.array(img, dtype="float32")
    arr = tf.image.resize(arr, IMG_SIZE).numpy()           # same resize as in training
    return np.expand_dims(arr, 0)                          # (1, H, W, 3); scaling is inside the model


def predict(img: Image.Image) -> np.ndarray:
    return model.predict(preprocess(img), verbose=0)[0]


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
#MainMenu, footer, header {visibility: hidden;}
.stApp {background: radial-gradient(1100px 600px at 10% -10%, #123524 0%, #0B1410 55%) fixed;}
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px;}
.hero {background: linear-gradient(135deg, #16A34A 0%, #0E7C86 60%, #3B82F6 100%); border-radius: 20px; padding: 28px 34px; margin-bottom: 24px; box-shadow: 0 10px 40px rgba(22,163,74,0.25);}
.hero h1 {color: #fff; font-size: 2rem; margin: 0 0 6px 0; font-weight: 800; letter-spacing: -0.5px;}
.hero p {color: rgba(255,255,255,0.9); margin: 0; font-size: 1rem;}
.card {background: #12201A; border: 1px solid #1F3A2D; border-radius: 18px; padding: 22px; margin-bottom: 16px;}
.card h3 {color: #E8F5EC; margin: 0 0 12px 0; font-size: 1.05rem; font-weight: 700;}
.muted {color: #8FB09D; font-size: 0.85rem;}
.badge {display: inline-block; padding: 6px 16px; border-radius: 999px; font-weight: 700; font-size: 0.9rem;}
.b-ok {background: rgba(34,197,94,0.15); color: #4ADE80; border: 1px solid rgba(34,197,94,0.4);}
.b-bad {background: rgba(239,68,68,0.15); color: #F87171; border: 1px solid rgba(239,68,68,0.4);}
.b-unk {background: rgba(245,158,11,0.15); color: #FBBF24; border: 1px solid rgba(245,158,11,0.4);}
.big {font-size: 1.6rem; font-weight: 800; color: #fff; margin: 10px 0 2px 0;}
.bar-row {margin: 10px 0;}
.bar-label {display: flex; justify-content: space-between; color: #D7EBDD; font-size: 0.88rem; margin-bottom: 4px;}
.bar-bg {background: #0B1410; border-radius: 999px; height: 10px; overflow: hidden;}
.bar-fill {height: 10px; border-radius: 999px; background: linear-gradient(90deg, #16A34A, #4ADE80);}
.stat {background: #0F1B15; border: 1px solid #1F3A2D; border-radius: 12px; padding: 12px 14px; margin-bottom: 10px;}
.stat .n {color: #fff; font-weight: 800; font-size: 1.25rem;}
.stat .l {color: #8FB09D; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.8px;}
.empty {text-align: center; padding: 50px 10px;}
.empty .e {font-size: 3rem;}
.empty .t {color: #E8F5EC; font-weight: 700; margin-top: 8px;}
.empty .d {color: #8FB09D; font-size: 0.9rem; margin-top: 4px;}
.stTabs [data-baseweb="tab"] {background: #0F1B15; border-radius: 10px; padding: 8px 18px; color: #A8C7B5;}
.stTabs [aria-selected="true"] {background: #16A34A !important; color: #fff !important;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display: none;}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌿 About the Model")
    st.markdown(
        '<p class="muted">MobileNetV2 pre-trained on ImageNet and fine-tuned on the '
        "PlantVillage dataset (color images). It recognises 38 plant and disease classes.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="stat"><div class="n">96.9%</div><div class="l">Test accuracy</div></div>'
        '<div class="stat"><div class="n">96.6%</div><div class="l">Validation accuracy</div></div>'
        '<div class="stat"><div class="n">38</div><div class="l">Classes</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### ⚠️ Limitations")
    st.markdown(
        '<p class="muted">The model was trained on laboratory-style photos with plain '
        "backgrounds, so accuracy in real fields is not guaranteed. It only knows the 38 "
        "classes it was trained on and will still return one of them for any image. "
        "Results below 60% confidence are marked as uncertain.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("This tool gives a first estimate only. For treatment decisions, "
               "consult an agricultural expert.")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="hero"><h1>🌿 AgriCare AI</h1>'
    "<p>Upload a clear photo of a single leaf and get the likely plant, "
    "its health status and the top predictions with confidence scores.</p></div>",
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1], gap="large")

with left:
    tab_up, tab_cam = st.tabs(["📁 Upload", "📷 Camera"])
    with tab_up:
        uploaded = st.file_uploader("Leaf image (JPG or PNG)", type=["jpg", "jpeg", "png"])
    with tab_cam:
        camera = st.camera_input("Take a photo of the leaf")
    source = uploaded or camera

    if source is not None:
        image = Image.open(source)
        st.image(image, caption="Input image", use_container_width=True)

with right:
    if source is None:
        st.markdown(
            '<div class="card"><div class="empty"><div class="e">🍃</div>'
            '<div class="t">No image yet</div>'
            '<div class="d">Upload a leaf photo or use the camera to see the prediction.</div>'
            "</div></div>",
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Analysing leaf..."):
            probs = predict(image)

        order = np.argsort(probs)[::-1][:3]
        top = int(order[0])
        conf = float(probs[top])
        plant, disease, healthy = pretty(class_names[top])

        if conf < CONF_THRESHOLD:
            badge = '<span class="badge b-unk">Uncertain result</span>'
            headline = "Not confident enough"
            sub = ("The top prediction is below 60% confidence. The image may be unclear, "
                   "not a supported leaf, or taken in conditions unlike the training data.")
        elif healthy:
            badge = '<span class="badge b-ok">Healthy</span>'
            headline = plant
            sub = "No disease detected for this plant class."
        else:
            badge = '<span class="badge b-bad">Disease detected</span>'
            headline = f"{plant}: {disease}"
            sub = f"Plant: {plant}"

        st.markdown(
            f'<div class="card"><h3>Prediction</h3>{badge}'
            f'<div class="big">{headline}</div>'
            f'<div class="muted">{sub}</div>'
            f'<div class="muted" style="margin-top:8px;">Top confidence: <b>{conf:.1%}</b></div></div>',
            unsafe_allow_html=True,
        )

        bars = ""
        for i in order:
            p, d, h = pretty(class_names[int(i)])
            name = f"{p}: {d}"
            pct = float(probs[int(i)]) * 100
            bars += (
                f'<div class="bar-row"><div class="bar-label"><span>{name}</span><span>{pct:.1f}%</span></div>'
                f'<div class="bar-bg"><div class="bar-fill" style="width:{pct:.1f}%;"></div></div></div>'
            )
        st.markdown(f'<div class="card"><h3>Top 3 Predictions</h3>{bars}</div>', unsafe_allow_html=True)

        st.caption("This is an estimate from a model trained on laboratory-style images. "
                   "It is not a guarantee or a replacement for an expert diagnosis.")