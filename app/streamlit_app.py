import streamlit as st
import torch
from PIL import Image
import numpy as np
import os
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils import load_model, get_image_transforms, predict_image, get_fruit_info

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="FreshScan AI — Fruit Freshness Detector",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  /* ── Reset & Base ── */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  html, body, [data-testid="stAppViewContainer"], .stApp {
      font-family: 'Inter', sans-serif;
      background: #f8fafc;
      color: #1e293b;
      min-height: 100vh;
  }

  /* ── Subtle light background decoration ── */
  [data-testid="stAppViewContainer"]::before {
      content: "";
      position: fixed; inset: 0;
      background:
          radial-gradient(ellipse 70% 50% at 10% 0%,  rgba(16,185,129,.08) 0%, transparent 60%),
          radial-gradient(ellipse 60% 40% at 90% 100%, rgba(59,130,246,.06) 0%, transparent 60%);
      pointer-events: none; z-index: 0;
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, header, footer,
  [data-testid="stToolbar"],
  [data-testid="collapsedControl"] { display: none !important; }

  /* ── Main block ── */
  .block-container {
      position: relative; z-index: 1;
      max-width: 1100px !important;
      padding: 2rem 2rem 3rem !important;
      background: transparent !important;
  }

  /* ── Card ── */
  .glass {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 20px;
      padding: 1.6rem;
      box-shadow: 0 1px 8px rgba(0,0,0,.06);
      transition: box-shadow .3s ease;
  }
  .glass:hover { box-shadow: 0 4px 24px rgba(16,185,129,.12); }

  /* ── Hero header ── */
  .hero-title {
      font-size: 3rem;
      font-weight: 800;
      letter-spacing: -1px;
      background: linear-gradient(135deg, #10b981 0%, #3b82f6 55%, #8b5cf6 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      background-clip: text;
      line-height: 1.15;
      margin-bottom: .5rem;
  }
  .hero-sub {
      font-size: 1.05rem;
      color: #64748b;
      font-weight: 400;
      margin-bottom: 0;
  }

  /* ── Stat cards ── */
  .stat-card {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 16px;
      padding: 1.1rem 1rem;
      text-align: center;
      box-shadow: 0 1px 6px rgba(0,0,0,.05);
      transition: transform .25s, box-shadow .25s;
  }
  .stat-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(16,185,129,.15); }
  .stat-value   { font-size: 1.5rem; font-weight: 700; color: #10b981; }
  .stat-label   { font-size: .78rem; color: #94a3b8; margin-top: .25rem; text-transform: uppercase; letter-spacing: .05em; }

  /* ── File uploader ── */
  [data-testid="stFileUploader"] {
      background: #fff;
      border: 2px dashed #a7f3d0;
      border-radius: 16px;
      padding: 1rem;
      transition: border-color .3s;
  }
  [data-testid="stFileUploader"]:hover { border-color: #10b981; }

  /* ── Camera input ── */
  [data-testid="stCameraInput"] {
      background: #fff;
      border: 1px solid #e2e8f0;
      border-radius: 16px;
      padding: 1rem;
  }

  /* ── Buttons ── */
  .stButton > button {
      background: linear-gradient(135deg, #10b981, #059669) !important;
      color: #fff !important;
      border: none !important;
      border-radius: 12px !important;
      padding: .7rem 2rem !important;
      font-weight: 600 !important;
      font-size: 1rem !important;
      letter-spacing: .02em;
      transition: all .3s ease !important;
      box-shadow: 0 4px 14px rgba(16,185,129,.30) !important;
  }
  .stButton > button:hover {
      transform: translateY(-2px) !important;
      box-shadow: 0 8px 24px rgba(16,185,129,.40) !important;
  }

  /* ── Progress bar ── */
  .stProgress > div > div > div > div {
      background: linear-gradient(90deg, #10b981, #3b82f6) !important;
      border-radius: 999px;
  }
  .stProgress > div > div {
      background: #e2e8f0 !important;
      border-radius: 999px;
  }

  /* ── Result cards ── */
  .result-fresh {
      background: linear-gradient(135deg, #f0fdf4, #dcfce7);
      border: 1.5px solid #86efac;
      border-radius: 20px; padding: 1.6rem;
      animation: fadeSlide .5s ease;
  }
  .result-stale {
      background: linear-gradient(135deg, #fffbeb, #fef3c7);
      border: 1.5px solid #fcd34d;
      border-radius: 20px; padding: 1.6rem;
      animation: fadeSlide .5s ease;
  }
  @keyframes fadeSlide {
      from { opacity: 0; transform: translateY(12px); }
      to   { opacity: 1; transform: translateY(0); }
  }

  .result-status-fresh { font-size: 1.9rem; font-weight: 800; color: #059669; }
  .result-status-stale { font-size: 1.9rem; font-weight: 800; color: #d97706; }
  .result-name  { font-size: 1.1rem; color: #64748b; margin: .25rem 0 .6rem; }
  .result-desc  { font-size: .9rem; color: #475569; line-height: 1.6; }
  .conf-label   { font-size: .82rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .06em; margin: .8rem 0 .3rem; }
  .conf-value   { font-size: 1.4rem; font-weight: 700; color: #1e293b; }

  /* ── Top-3 pills ── */
  .top3-row     { display: flex; flex-direction: column; gap: .5rem; margin-top: .6rem; }
  .top3-item    {
      display: flex; align-items: center; justify-content: space-between;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 10px; padding: .5rem .9rem;
      font-size: .88rem;
  }
  .top3-rank    { font-weight: 700; color: #94a3b8; margin-right: .6rem; }
  .top3-name    { flex: 1; color: #334155; }
  .top3-prob    { font-weight: 600; color: #10b981; }

  /* ── Fruit pill tags ── */
  .fruit-tag {
      display: inline-block;
      background: #f1f5f9;
      border: 1px solid #e2e8f0;
      border-radius: 999px;
      padding: .3rem .85rem;
      font-size: .85rem; color: #475569;
      transition: background .2s, border-color .2s;
  }
  .fruit-tag:hover {
      background: #d1fae5;
      border-color: #6ee7b7;
      color: #065f46;
  }

  /* ── Section label ── */
  .section-label {
      font-size: .72rem;
      text-transform: uppercase;
      letter-spacing: .1em;
      color: #10b981;
      font-weight: 600;
      margin-bottom: .5rem;
  }

  /* ── Divider ── */
  hr.fancy {
      border: none;
      height: 1px;
      background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
      margin: 1.5rem 0;
  }

  /* ── Image display ── */
  [data-testid="stImage"] img {
      border-radius: 16px !important;
      border: 1px solid #e2e8f0 !important;
  }

  /* ── Spinner ── */
  [data-testid="stSpinner"] { color: #10b981 !important; }

  /* ── Error / info messages ── */
  .stAlert { border-radius: 12px !important; }

  /* ── Label text ── */
  .stSelectbox label, .stFileUploader label, .stCameraInput label {
      color: #64748b !important;
      font-size: .9rem !important;
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #f1f5f9; }
  ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─── Model Loader ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_trained_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'models', 'best_model.pth'
    )
    try:
        model = load_model(model_path, device)
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


# ─── Result Display ───────────────────────────────────────────────────────────
def display_prediction_results(predicted_class, confidence, top3_predictions, image):
    fruit_info = get_fruit_info(predicted_class)
    is_fresh   = "fresh" in predicted_class

    img_col, res_col = st.columns([1, 1], gap="large")

    with img_col:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with res_col:
        box_cls  = "result-fresh" if is_fresh else "result-stale"
        stat_cls = "result-status-fresh" if is_fresh else "result-status-stale"

        top3_html = ""
        medals = ["🥇", "🥈", "🥉"]
        for i, (cn, prob) in enumerate(top3_predictions):
            fi = get_fruit_info(cn)
            top3_html += f"""
            <div class="top3-item">
              <span class="top3-rank">{medals[i]}</span>
              <span class="top3-name">{fi['name']}</span>
              <span class="top3-prob">{prob:.1%}</span>
            </div>"""

        st.markdown(f"""
        <div class="{box_cls}">
          <div class="{stat_cls}">{fruit_info['status']}</div>
          <div class="result-name">{fruit_info['name']}</div>
          <div class="result-desc">{fruit_info['description']}</div>

          <div class="conf-label">Confidence</div>
          <div class="conf-value">{confidence:.2%}</div>
        </div>

        <hr class="fancy"/>

        <div class="section-label">Top Predictions</div>
        <div class="top3-row">{top3_html}</div>
        """, unsafe_allow_html=True)

        # progress bar below the markdown
        st.progress(confidence)


# ─── Main App ─────────────────────────────────────────────────────────────────
def main():
    model, device = load_trained_model()

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:2rem;">
      <div class="section-label">AI Fruit Inspector</div>
      <div class="hero-title">FreshScan AI</div>
      <div class="hero-sub">
        Instantly detect whether your fruit or vegetable is fresh or stale
        using state-of-the-art deep learning.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Stat cards ────────────────────────────────────────────────────────────
    if model is not None:
        c1, c2, c3, c4 = st.columns(4)
        device_label = "GPU 🚀" if device.type == "cuda" else "CPU 💻"
        stats = [
            ("99.91%",    "Model Accuracy"),
            ("12",        "Fruit Classes"),
            ("EfficientNet-B0",  "Architecture"),
            (device_label, "Running On"),
        ]
        for col, (val, lbl) in zip([c1, c2, c3, c4], stats):
            with col:
                st.markdown(f"""
                <div class="stat-card">
                  <div class="stat-value">{val}</div>
                  <div class="stat-label">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='fancy'>", unsafe_allow_html=True)

    # ── Supported fruits ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="glass" style="margin-bottom:1.5rem;">
      <div class="section-label" style="margin-bottom:.8rem;">Supported Fruits & Vegetables</div>
      <div style="display:flex;flex-wrap:wrap;gap:.5rem;">
        <span class="fruit-tag">🍎 Apple</span>
        <span class="fruit-tag">🍌 Banana</span>
        <span class="fruit-tag">🥒 Bitter Gourd</span>
        <span class="fruit-tag">🫑 Capsicum</span>
        <span class="fruit-tag">🍊 Orange</span>
        <span class="fruit-tag">🍅 Tomato</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Guard ─────────────────────────────────────────────────────────────────
    if model is None:
        st.error("⚠️ Failed to load model. Please check that `models/best_model.pth` exists.")
        return

    transform = get_image_transforms()

    # ── Method tabs ───────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["📁  Upload Image", "📸  Use Camera"])

    with tab1:
        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop your image here or click to browse",
            type=["png", "jpg", "jpeg"],
            help="Supports PNG and JPEG. Clear, well-lit photos give the best results."
        )
        if uploaded_file:
            image = Image.open(uploaded_file)
            with st.spinner("🔍 Analyzing image…"):
                try:
                    predicted_class, confidence, top3 = predict_image(model, image, device, transform)
                    st.markdown("<hr class='fancy'>", unsafe_allow_html=True)
                    st.markdown("""
                    <div style="margin-bottom:.8rem;">
                      <div class="section-label">Detection Result</div>
                      <div style="font-size:1.4rem;font-weight:700;color:#1e293b;">Analysis Result</div>
                    </div>
                    """, unsafe_allow_html=True)
                    display_prediction_results(predicted_class, confidence, top3, image)
                except Exception as e:
                    st.error(f"Prediction error: {e}")

    with tab2:
        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        _, cam_col, _ = st.columns([1, 2, 1])
        with cam_col:
            camera_image = st.camera_input("Point camera at your fruit")

        if camera_image:
            image = Image.open(camera_image)
            _, btn_col, _ = st.columns([1, 1, 1])
            with btn_col:
                analyze = st.button("🔍  Analyze Picture", type="primary", use_container_width=True)
            if analyze:
                with st.spinner("🔍 Analyzing…"):
                    try:
                        predicted_class, confidence, top3 = predict_image(model, image, device, transform)
                        st.markdown("<hr class='fancy'>", unsafe_allow_html=True)
                        st.markdown("""
                        <div style="margin-bottom:.8rem;">
                          <div class="section-label">Detection Result</div>
                          <div style="font-size:1.4rem;font-weight:700;color:#1e293b;">Analysis Result</div>
                        </div>
                        """, unsafe_allow_html=True)
                        display_prediction_results(predicted_class, confidence, top3, image)
                    except Exception as e:
                        st.error(f"Prediction error: {e}")




if __name__ == "__main__":
    main()
