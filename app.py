import os
import sys
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessor import TextPreprocessor
from extractor import LinguisticExtractor
from recommender import NewsRecommender

st.set_page_config(
    page_title="NewsSense AI // Neural Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# 1. Custom CSS: Glassmorphism, Neon Glows & Sci-Fi Palettes
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;600&display=swap');

    /* Global Dark Backdrop */
    .stApp {
        background-color: #050811 !important;
        font-family: 'Inter', sans-serif;
        color: #e2e8f0;
    }

    /* Headings with Gradient & Orbitron Font */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .main-title {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #9055ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 35px rgba(79, 172, 254, 0.45);
        margin-bottom: 0px;
    }

    .subtitle-badge {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        background: rgba(79, 172, 254, 0.1);
        border: 1px solid rgba(79, 172, 254, 0.3);
        color: #38bdf8;
        padding: 5px 14px;
        border-radius: 20px;
        margin-bottom: 25px;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.2);
    }

    /* Glassmorphism Containers */
    div[data-testid="stMetric"], .glass-card {
        background: rgba(15, 23, 42, 0.65) !important;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5) !important;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }

    div[data-testid="stMetric"]:hover, .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(56, 189, 248, 0.6) !important;
        box-shadow: 0 12px 40px rgba(56, 189, 248, 0.25) !important;
    }

    /* Metric Label and Value Styles */
    div[data-testid="stMetricLabel"] p {
        font-family: 'JetBrains Mono', monospace !important;
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    div[data-testid="stMetricValue"] div {
        font-family: 'Orbitron', sans-serif !important;
        color: #38bdf8 !important;
        font-size: 1.85rem !important;
        text-shadow: 0 0 12px rgba(56, 189, 248, 0.5);
    }

    /* Modern Text Area */
    .stTextArea textarea {
        background: rgba(13, 20, 36, 0.75) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 12px !important;
        color: #f1f5f9 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.95rem;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
    }

    .stTextArea textarea:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.35) !important;
    }

    /* Primary Execution Button with Glow Animation */
    button[kind="primary"] {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%) !important;
        border: none !important;
        color: #030712 !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        border-radius: 10px !important;
        padding: 10px 30px !important;
        box-shadow: 0 0 25px rgba(79, 172, 254, 0.4) !important;
        transition: all 0.3s ease-in-out !important;
    }

    button[kind="primary"]:hover {
        transform: scale(1.03) !important;
        box-shadow: 0 0 40px rgba(0, 242, 254, 0.8) !important;
        color: #000 !important;
    }

    /* Sci-Fi Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.5);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #94a3b8;
        border-radius: 8px;
        padding: 8px 18px;
        background: transparent;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(56, 189, 248, 0.15) !important;
        color: #38bdf8 !important;
        font-weight: 600;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.25);
    }

    /* Badges & Entity Pills */
    .entity-tag {
        display: inline-block;
        padding: 4px 10px;
        margin: 4px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
        background: rgba(147, 51, 234, 0.2);
        color: #c084fc;
        border: 1px solid rgba(147, 51, 234, 0.4);
    }

    .chunk-tag {
        display: inline-block;
        padding: 4px 10px;
        margin: 4px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
        background: rgba(14, 165, 233, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(14, 165, 233, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Interactive Background Particle & Neural Web Mesh
# ---------------------------------------------------------
components.html("""
<canvas id="neural-canvas" style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; pointer-events: none; z-index: -1;"></canvas>
<script>
    const canvas = document.getElementById('neural-canvas');
    const ctx = canvas.getContext('2d');
    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;

    window.addEventListener('resize', () => {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    });

    const particles = [];
    const particleCount = 45;

    for (let i = 0; i < particleCount; i++) {
        particles.push({
            x: Math.random() * width,
            y: Math.random() * height,
            vx: (Math.random() - 0.5) * 0.7,
            vy: (Math.random() - 0.5) * 0.7,
            radius: Math.random() * 2 + 1
        });
    }

    function animate() {
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = "rgba(56, 189, 248, 0.4)";
        ctx.strokeStyle = "rgba(56, 189, 248, 0.1)";

        for (let i = 0; i < particles.length; i++) {
            let p = particles[i];
            p.x += p.vx;
            p.y += p.vy;

            if (p.x < 0 || p.x > width) p.vx *= -1;
            if (p.y < 0 || p.y > height) p.vy *= -1;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            ctx.fill();

            for (let j = i + 1; j < particles.length; j++) {
                let p2 = particles[j];
                let dist = Math.hypot(p.x - p2.x, p.y - p2.y);
                if (dist < 130) {
                    ctx.beginPath();
                    ctx.moveTo(p.x, p.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(animate);
    }
    animate();
</script>
""", height=0)

# ---------------------------------------------------------
# 3. Model & Pipeline Artifacts Loader
# ---------------------------------------------------------
@st.cache_resource
def load_components():
    classifier = joblib.load(os.path.join("models", "classifier.pkl"))
    vectorizer = joblib.load(os.path.join("models", "tfidf_vectorizer.pkl"))
    preprocessor = TextPreprocessor()
    extractor = LinguisticExtractor()
    recommender = NewsRecommender()
    return classifier, vectorizer, preprocessor, extractor, recommender

clf, vectorizer, preprocessor, extractor, recommender = load_components()

# ---------------------------------------------------------
# 4. Header UI
# ---------------------------------------------------------
st.markdown('<div class="main-title">NEWSSENSE AI</div>', unsafe_allow_html=True)
st.markdown('<span class="subtitle-badge">CYBERNETIC LINGUISTIC PIPELINE & SEMANTIC RETRIEVAL MATRIX</span>', unsafe_allow_html=True)

sample_texts = {
    "Custom Input (Type/Paste Anything)": "",
    "Cybersecurity / Tech Intelligence": "Computer security experts warned consumers about malicious spyware software circulating via email attachments. Software developers and broadband internet providers are updating firewall systems to protect PC users from downloading rogue programs.",
    "Central Banking / High Finance": "The central bank decided to raise interest rates to tackle rising inflation figures and economic growth concerns. European stock markets dropped following the announcement, with banking equities seeing heavy losses.",
    "Premier League Championship": "Arsenal sealed a 2-0 victory against Chelsea at Highbury yesterday afternoon. Thierry Henry scored both goals in the second half with brilliant finishes, sending the football club to the top of the Premier League table."
}

selected_sample = st.selectbox("Select Intelligence Preset:", list(sample_texts.keys()))
default_val = sample_texts[selected_sample] if selected_sample != "Custom Input (Type/Paste Anything)" else ""

input_text = st.text_area("Input Stream Buffer:", value=default_val, height=140, placeholder="Stream raw text telemetry here...")

if st.button("RUN NEURAL ANALYSIS", type="primary"):
    if not input_text.strip():
        st.warning("Buffer empty. Please inject text stream.")
    else:
        # Preprocessing & Inference
        cleaned = preprocessor.tokenize_and_lemmatize(input_text)
        vec = vectorizer.transform([cleaned])
        pred_label = clf.predict(vec)[0]

        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(vec)[0]
            classes = clf.classes_
        else:
            probs = [1.0]
            classes = [pred_label]

        ling_data = extractor.extract_linguistics(input_text)
        recommendations = recommender.recommend(input_text, top_k=3)

        st.markdown("<br>", unsafe_allow_html=True)

        # Top Futuristic Metric Cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Predicted Domain", pred_label.upper())
        with c2:
            st.metric("Entity Nodes", len(ling_data["entities"]))
        with c3:
            st.metric("Syntactic Chunks", len(ling_data["noun_chunks"]))

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs([
            "⚡ TOPIC INFERENCE MATRIX",
            "🏷️ ENTITY & CHUNK TELEMETRY",
            "🧬 SYNTACTIC & POS PARSER",
            "🌐 VECTOR COSINE PROXIMITY"
        ])

        with tab1:
            col_chart, col_proc = st.columns([3, 2])
            with col_chart:
                prob_df = pd.DataFrame({"Category": classes, "Confidence": probs})
                prob_df = prob_df.sort_values(by="Confidence", ascending=True)

                fig = px.bar(
                    prob_df, x="Confidence", y="Category", orientation='h',
                    text_auto='.1%',
                    color="Confidence",
                    color_continuous_scale=[[0, '#0f172a'], [0.5, '#38bdf8'], [1, '#00f2fe']]
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94a3b8', family="JetBrains Mono"),
                    margin=dict(l=20, r=20, t=30, b=20),
                    coloraxis_showscale=False,
                    xaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)", zerolinecolor="rgba(255, 255, 255, 0.1)"),
                    yaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)")
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_proc:
                st.markdown("#### Corpus Normalization Trace")
                st.markdown("**Raw Input Stream:**")
                st.caption(input_text[:220] + "..." if len(input_text) > 220 else input_text)
                st.markdown("**Lemmatized Morphological Form:**")
                st.code(cleaned if cleaned else "No valid tokens persisted filter.", language="text")

        with tab2:
            c_ent, c_chunk = st.columns(2)
            with c_ent:
                st.markdown("#### Named Entity Matrix (NER)")
                if ling_data["entities"]:
                    for ent in ling_data["entities"]:
                        st.markdown(f"<span class='entity-tag'><b>{ent['text']}</b> <span style='opacity:0.7;'>({ent['label']})</span></span>", unsafe_allow_html=True)
                else:
                    st.info("No categorical entities registered.")

            with c_chunk:
                st.markdown("#### Noun Chunks (Shallow Parse)")
                if ling_data["noun_chunks"]:
                    for chunk in ling_data["noun_chunks"]:
                        st.markdown(f"<span class='chunk-tag'>{chunk}</span>", unsafe_allow_html=True)
                else:
                    st.info("No noun chunk dependencies detected.")

        with tab3:
            st.markdown("#### Part-of-Speech Structural Histogram")
            if ling_data["pos_distribution"]:
                pos_df = pd.DataFrame(list(ling_data["pos_distribution"].items()), columns=["POS Tag", "Count"]).sort_values(by="Count", ascending=False)
                fig_pos = px.bar(
                    pos_df, x="POS Tag", y="Count",
                    color="Count",
                    color_continuous_scale=[[0, '#818cf8'], [1, '#c084fc']]
                )
                fig_pos.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94a3b8', family="JetBrains Mono"),
                    margin=dict(l=20, r=20, t=20, b=20),
                    coloraxis_showscale=False,
                    xaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)"),
                    yaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)")
                )
                st.plotly_chart(fig_pos, use_container_width=True)

            with st.expander("🔬 Unfold Full Syntactic Dependency Graph Tokens"):
                st.dataframe(pd.DataFrame(ling_data["token_analysis"]), use_container_width=True)

        with tab4:
            st.markdown("#### Vector Cosine Semantic Neighbors")
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"""
                <div class="glass-card" style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="color: #38bdf8; font-family: 'Orbitron'; font-weight: 700;">#{i} DOMAIN: {rec['category'].upper()}</span>
                        <span style="background: rgba(56, 189, 248, 0.2); color: #38bdf8; padding: 2px 10px; border-radius: 12px; font-family: 'JetBrains Mono'; font-size: 0.8rem;">COSINE SIMILARITY: {rec['similarity_score']}</span>
                    </div>
                    <div style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.5;">{rec['snippet']}</div>
                </div>
                """, unsafe_allow_html=True)
