"""
streamlit_app.py — BI-RADS 5-Architecture Ensemble Classifier (Streamlit UI)

Deploy on Streamlit Community Cloud or a Hugging Face Space with SDK=streamlit.
Model definitions, checkpoint config, and inference logic live in
model_utils.py — edit the config block at the top of that file (checkpoint
source, class names, vote weights) before deploying.
"""

import streamlit as st
from PIL import Image
from model_utils import run_ensemble, class_names, CHECKPOINT_REPO_ID, USE_LOCAL_CHECKPOINTS

MODEL_NAME = "DenseNet121 (CC + MLO)"

st.set_page_config(
    page_title="BI-RADS Ensemble Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------
# Styling
# ---------------------------------------------
st.markdown(
    """
    <style>
    .hero {
        padding: 1.75rem 2rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 { margin-bottom: 0.25rem; font-size: 1.9rem; }
    .hero p { margin: 0; opacity: 0.85; font-size: 0.95rem; }
    .pred-card {
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        background: #f3f4f6;
        border: 1px solid #e5e7eb;
    }
    .pred-label { font-size: 1.4rem; font-weight: 700; color: #111827; }
    .pred-conf { font-size: 0.95rem; color: #6b7280; }
    .disclaimer {
        font-size: 0.82rem;
        color: #9ca3af;
        border-top: 1px solid #e5e7eb;
        padding-top: 0.75rem;
        margin-top: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------
# Sidebar
# ---------------------------------------------
with st.sidebar:
    st.markdown("## 🩺 About")
    st.markdown(
        f"""
        This demo classifies mammogram pairs (CC + MLO views) into BI-RADS
        categories using a **{MODEL_NAME}** model fine-tuned with CBAM
        attention. The CC-view and MLO-view softmax probabilities are
        averaged for the final prediction.
        """
    )
    st.markdown("---")
    st.markdown("### Classes")
    for c in class_names:
        st.markdown(f"- {c}")
    st.markdown("---")
    checkpoint_source = "Local `checkpoints/` folder" if USE_LOCAL_CHECKPOINTS else f"HF model repo: `{CHECKPOINT_REPO_ID}`"
    st.caption(f"Checkpoint source: {checkpoint_source}")
    st.markdown(
        '<div class="disclaimer">Research/demo tool only. Not a medical '
        "device and not intended for clinical diagnosis or to guide real "
        "patient care.</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------
# Hero header
# ---------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>BI-RADS Intelligent Classifier</h1>
        <p>DenseNet121 (CC + MLO averaged) BI-RADS classifier</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------
# Upload section
# ---------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### CC View")
    cc_file = st.file_uploader("Upload CC view image", type=["png", "jpg", "jpeg"], key="cc")
    if cc_file:
        st.image(Image.open(cc_file), use_container_width=True)

with col2:
    st.markdown("#### MLO View")
    mlo_file = st.file_uploader("Upload MLO view image", type=["png", "jpg", "jpeg"], key="mlo")
    if mlo_file:
        st.image(Image.open(mlo_file), use_container_width=True)

run_clicked = st.button("🔍 Run Ensemble Classification", type="primary", use_container_width=True)

# ---------------------------------------------
# Inference + results
# ---------------------------------------------
if run_clicked:
    if not cc_file or not mlo_file:
        st.warning("Please upload both a CC view and an MLO view image before running.")
    else:
        with st.spinner("Running DenseNet121 CC+MLO inference..."):
            cc_image = Image.open(cc_file)
            mlo_image = Image.open(mlo_file)
            final_probs = run_ensemble(cc_image, mlo_image)

        top_class = max(final_probs, key=final_probs.get)
        top_conf = final_probs[top_class]

        st.markdown("### Result")
        st.markdown(
            f"""
            <div class="pred-card">
                <div class="pred-label">{top_class}</div>
                <div class="pred-conf">Confidence: {top_conf * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Class Probabilities (CC + MLO averaged)")
        sorted_classes = sorted(final_probs.items(), key=lambda kv: kv[1], reverse=True)
        for cls, prob in sorted_classes:
            st.write(f"**{cls}** — {prob * 100:.1f}%")
            st.progress(min(max(prob, 0.0), 1.0))
