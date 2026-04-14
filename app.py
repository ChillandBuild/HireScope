"""
HireScope — India Job Market Intelligence Dashboard
Streamlit app with two pages:
  Page 1: Market Overview (forecasts, charts, SHAP)
  Page 2: Career Intelligence (resume upload, AI agents)
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import load_all
from src.aggregator import aggregate_weekly
from src.forecaster import forecast_all_skills, get_forecast_summary
from src.feature_engineer import compute_features, compute_demand_scores
from src.evaluator import explain_predictions, get_shap_df

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HireScope — India Job Market Intelligence",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" />
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* Force white background everywhere */
.stApp, .main, [data-testid="stAppViewContainer"] {
    background-color: #ffffff !important;
}

.main .block-container {
    padding-top: 1.5rem;
    max-width: 1200px;
}

/* Header styling */
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
    letter-spacing: -0.5px;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #64748b;
    font-weight: 400;
    margin-bottom: 1.5rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    border: none;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.25);
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.2;
}

.metric-label {
    font-size: 0.8rem;
    color: rgba(255, 255, 255, 0.8);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 500;
}

/* Section headers */
.section-header {
    font-size: 1.2rem;
    font-weight: 600;
    color: #1e293b;
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid rgba(99, 102, 241, 0.3);
}

/* Agent card */
.agent-card {
    background: #f8f9fc;
    border-radius: 12px;
    padding: 1.2rem;
    border: 1px solid #e2e8f0;
    margin-bottom: 1rem;
}

.agent-card h4 {
    color: #1e293b;
}

/* Status badges */
.badge-rising {
    background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
    color: #065f46;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}

.badge-declining {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    color: #991b1b;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}

.badge-stable {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    color: #92400e;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Sidebar - light theme */
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8f9fc 0%, #eef0f7 100%);
}

div[data-testid="stSidebar"] [data-testid="stMarkdown"] {
    color: #1e293b;
}

/* Hide default streamlit footer */
footer {visibility: hidden;}

/* Ensure Material Symbols icons render as icons, not text */
.material-symbols-rounded,
[data-testid="collapsedControl"] span,
button[kind="header"] span,
[data-testid="stFileUploader"] span[class*="Icon"],
[data-testid="baseButton-header"] span {
    font-family: 'Material Symbols Rounded' !important;
    font-style: normal;
    font-weight: normal;
    font-size: 24px !important;
    line-height: 1;
    letter-spacing: normal;
    text-transform: none;
    display: inline-block;
    white-space: nowrap;
    word-wrap: normal;
    direction: ltr;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
    font-feature-settings: 'liga';
}

/* Fix expander text overlap */
div[data-testid="stExpander"] summary {
    font-size: 0.95rem;
    font-weight: 500;
    color: #1e293b;
    padding: 0.5rem 0;
}

div[data-testid="stExpander"] summary span {
    display: inline-block;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
</style>
""", unsafe_allow_html=True)


# ── data loading with caching ──────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading 820K+ job postings...")
def load_data():
    """Load and process all data (cached)."""
    df = load_all()
    weekly, top_skills = aggregate_weekly(df)
    return df, weekly, top_skills


@st.cache_data(show_spinner="Running Prophet forecasts...")
def run_forecasts(_weekly, top_skills):
    """Run Prophet forecasts for all skills (cached)."""
    import warnings
    warnings.filterwarnings("ignore")
    forecasts = forecast_all_skills(_weekly, top_skills)
    summary = get_forecast_summary(forecasts)
    return forecasts, summary


@st.cache_data(show_spinner="Computing demand scores...")
def compute_scores(_weekly, top_skills):
    """Compute XGBoost demand scores (cached)."""
    features = compute_features(_weekly, top_skills)
    scores, model = compute_demand_scores(features)
    return features, scores, model


@st.cache_data(show_spinner="Generating SHAP explanations...")
def compute_explanations(_model, _features):
    """Generate SHAP explanations (cached)."""
    explanations = explain_predictions(_model, _features)
    shap_df, feature_cols = get_shap_df(_model, _features)
    return explanations, shap_df, feature_cols


# ── load everything ────────────────────────────────────────────────────────────
df, weekly, top_skills = load_data()
forecasts, forecast_summary = run_forecasts(weekly, top_skills)
features, scores, model = compute_scores(weekly, top_skills)
explanations, shap_df, feature_cols = compute_explanations(model, features)

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔭 HireScope")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Market Overview", "🎯 Career Intelligence"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(f"**Data:** {len(df):,} postings")
    st.markdown(f"**Skills tracked:** {len(top_skills)}")
    st.markdown(f"**Sources:** LinkedIn + Naukri")

    # API key status
    groq_key = os.getenv("GROQ_API_KEY", "")
    has_groq = groq_key and groq_key != "your_groq_key_here"
    if has_groq:
        st.success("✅ Groq API connected")
    else:
        st.warning("⚠️ No Groq API key — AI agents disabled")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Market Overview":
    # ── hero header ────────────────────────────────────────────────────────
    st.markdown('<div class="hero-title">HireScope</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">India Job Market Intelligence — Powered by 820K+ Real Job Postings</div>', unsafe_allow_html=True)

    # ── metric cards ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df['job_id'].nunique():,}</div>
            <div class="metric-label">Jobs Analysed</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(top_skills)}</div>
            <div class="metric-label">Skills Tracked</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        date_range = df["date"].dropna()
        last_date = date_range.max().strftime("%d %b %Y") if not date_range.empty else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{last_date}</div>
            <div class="metric-label">Last Updated</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">90 Days</div>
            <div class="metric-label">Forecast Horizon</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── rising and declining skills ────────────────────────────────────────
    if not forecast_summary.empty:
        col_rise, col_fall = st.columns(2)

        with col_rise:
            st.markdown('<div class="section-header">📈 Top 10 Rising Skills</div>', unsafe_allow_html=True)
            rising = forecast_summary[forecast_summary["trend"] == "RISING"].head(10)
            if not rising.empty:
                fig_rise = px.bar(
                    rising,
                    x="change_pct",
                    y="skill",
                    orientation="h",
                    color="change_pct",
                    color_continuous_scale=["#065f46", "#10b981", "#6ee7b7"],
                    labels={"change_pct": "Growth %", "skill": ""},
                )
                fig_rise.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#1e293b", family="Inter"),
                    showlegend=False,
                    coloraxis_showscale=False,
                    height=380,
                    margin=dict(l=0, r=20, t=10, b=10),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_rise, use_container_width=True)
            else:
                st.info("No rising skills detected in current forecast period.")

        with col_fall:
            st.markdown('<div class="section-header">📉 Top 5 Declining Skills</div>', unsafe_allow_html=True)
            declining = forecast_summary[forecast_summary["trend"] == "DECLINING"].tail(5)
            if not declining.empty:
                fig_fall = px.bar(
                    declining,
                    x="change_pct",
                    y="skill",
                    orientation="h",
                    color="change_pct",
                    color_continuous_scale=["#fca5a5", "#dc2626", "#7f1d1d"],
                    labels={"change_pct": "Decline %", "skill": ""},
                )
                fig_fall.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#1e293b", family="Inter"),
                    showlegend=False,
                    coloraxis_showscale=False,
                    height=380,
                    margin=dict(l=0, r=20, t=10, b=10),
                )
                st.plotly_chart(fig_fall, use_container_width=True)
            else:
                st.info("No declining skills detected in current forecast period.")

    # ── skill selector + forecast chart ────────────────────────────────────
    st.markdown('<div class="section-header">🔮 90-Day Skill Forecast</div>', unsafe_allow_html=True)

    available_skills = list(forecasts.keys())
    if available_skills:
        selected_skill = st.selectbox(
            "Select a skill to view its forecast",
            available_skills,
            index=0,
        )

        if selected_skill in forecasts:
            fc = forecasts[selected_skill]

            fig_forecast = go.Figure()

            # actual values
            actuals = fc.dropna(subset=["actual"])
            fig_forecast.add_trace(go.Scatter(
                x=actuals["ds"],
                y=actuals["actual"],
                mode="lines+markers",
                name="Actual",
                line=dict(color="#818cf8", width=3),
                marker=dict(size=8, color="#818cf8"),
            ))

            # forecast
            fig_forecast.add_trace(go.Scatter(
                x=fc["ds"],
                y=fc["yhat"],
                mode="lines",
                name="Forecast",
                line=dict(color="#f472b6", width=2, dash="dash"),
            ))

            # confidence interval
            fig_forecast.add_trace(go.Scatter(
                x=pd.concat([fc["ds"], fc["ds"][::-1]]),
                y=pd.concat([fc["yhat_upper"], fc["yhat_lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(244, 114, 182, 0.1)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Confidence Band",
                showlegend=True,
            ))

            fig_forecast.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#1e293b", family="Inter"),
                xaxis=dict(title="", gridcolor="rgba(148,163,184,0.15)"),
                yaxis=dict(title="Weekly Postings", gridcolor="rgba(148,163,184,0.15)"),
                height=400,
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

    # ── SHAP explanation ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">🧠 Why Is This Skill Trending?</div>', unsafe_allow_html=True)

    if available_skills:
        shap_skill = st.selectbox(
            "Select skill for SHAP explanation",
            available_skills,
            index=0,
            key="shap_skill",
        )

        if shap_skill in explanations:
            exp = explanations[shap_skill]

            # build waterfall-style bar chart
            features_list = [e["feature"].replace("_", " ").title() for e in exp]
            values = [e["contribution"] for e in exp]
            colors = ["#10b981" if v > 0 else "#ef4444" for v in values]

            fig_shap = go.Figure(go.Bar(
                x=values,
                y=features_list,
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.2f}" for v in values],
                textposition="outside",
                textfont=dict(color="#1e293b"),
            ))
            fig_shap.update_layout(
                title=f"Feature contributions for {shap_skill}",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#1e293b", family="Inter"),
                height=300,
                margin=dict(l=0, r=60, t=40, b=10),
                xaxis=dict(title="SHAP Value (impact on prediction)",
                           gridcolor="rgba(148,163,184,0.15)"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_shap, use_container_width=True)

    # ── demand scores table ────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Full Demand Score Rankings</div>', unsafe_allow_html=True)

    if isinstance(scores, tuple):
        scores_df = scores[0] if isinstance(scores, tuple) else scores
    else:
        scores_df = scores

    display_cols = ["skill", "demand_score", "trend", "confidence", "posting_count"]
    valid_cols = [c for c in display_cols if c in scores_df.columns]

    if valid_cols:
        styled_df = scores_df[valid_cols].copy()
        styled_df = styled_df.rename(columns={
            "skill": "Skill",
            "demand_score": "Demand Score",
            "trend": "Trend",
            "confidence": "Confidence",
            "posting_count": "Weekly Posts",
        })
        st.dataframe(styled_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: CAREER INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Career Intelligence":
    st.markdown('<div class="hero-title">Career Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Upload your resume and get AI-powered career guidance</div>', unsafe_allow_html=True)

    # check if crewai is available
    try:
        import crewai
        has_crewai = True
    except ImportError:
        has_crewai = False

    if not has_crewai:
        st.info("""
        **AI Agents are not available.**  
        Install agent dependencies: `pip install -r requirements-agents.txt`  
        The Market Overview page with forecasts and SHAP explanations works fully!
        """)
    elif not has_groq:
        st.warning("""
        ⚠️ **AI Agents require a Groq API key.**  
        Add your key to `.env` file: `GROQ_API_KEY=your_key_here`  
        The Market Overview page works without it.
        """)

    # ── resume upload + gap analysis ───────────────────────────────────────
    st.markdown('<div class="section-header">📄 Resume Analysis</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload your resume (PDF)",
        type=["pdf"],
        help="We'll extract skills and compare against market demand",
    )

    resume_text = ""
    if uploaded_file is not None:
        try:
            import fitz  # pymupdf
            pdf_bytes = uploaded_file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            resume_text = ""
            for pdf_page in doc:
                resume_text += pdf_page.get_text()
            doc.close()
            st.success(f"✅ Resume loaded: {len(resume_text)} characters extracted")

            with st.expander("Preview extracted text"):
                st.text(resume_text[:2000])
        except ImportError:
            st.error("PyMuPDF not installed. Run: `pip install pymupdf`")
        except Exception as e:
            st.error(f"Error reading PDF: {e}")

    col_gap, col_opp = st.columns(2)

    with col_gap:
        st.markdown("""
        <div class="agent-card">
            <h4>🔍 Career Gap Analyst</h4>
            <p style="color: #64748b; font-size: 0.85rem;">
                Compares your skills against the top 20 rising market skills 
                and identifies priority learning areas.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Run Gap Analysis", disabled=not (has_crewai and has_groq and resume_text),
                      use_container_width=True):
            with st.spinner("🔍 Agent analysing your resume..."):
                try:
                    from agents.gap_analyser import run_gap_analysis

                    # prepare rising skills data
                    rising_data = []
                    if isinstance(scores, tuple):
                        scores_df = scores[0]
                    else:
                        scores_df = scores

                    for _, row in scores_df.iterrows():
                        rising_data.append({
                            "skill": row.get("skill", ""),
                            "demand_score": row.get("demand_score", 0),
                            "trend": row.get("trend", "STABLE"),
                            "confidence": row.get("confidence", "MEDIUM"),
                        })

                    result = run_gap_analysis(resume_text, rising_data)
                    st.session_state["gap_report"] = result
                except Exception as e:
                    st.error(f"Agent error: {e}")

    with col_opp:
        st.markdown("""
        <div class="agent-card">
            <h4>🔭 Opportunity Scout</h4>
            <p style="color: #64748b; font-size: 0.85rem;">
                Finds the top 5 job opportunities with the best timing 
                based on market demand trends.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Scout Opportunities", disabled=not (has_crewai and has_groq and resume_text),
                      use_container_width=True):
            with st.spinner("🔭 Scouting best opportunities..."):
                try:
                    from agents.opportunity_scout import run_opportunity_scouting

                    gap_report = st.session_state.get("gap_report", "No gap analysis yet.")
                    student_skills = [s.strip() for s in resume_text.split() if len(s) > 3][:20]

                    fc_data = []
                    if not forecast_summary.empty:
                        for _, row in forecast_summary.iterrows():
                            fc_data.append({
                                "skill": row.get("skill", ""),
                                "trend": row.get("trend", "STABLE"),
                                "change_pct": row.get("change_pct", 0),
                                "confidence": "HIGH",
                            })

                    result = run_opportunity_scouting(student_skills, gap_report, fc_data)
                    st.session_state["opportunities"] = result
                except Exception as e:
                    st.error(f"Agent error: {e}")

    # display results
    if "gap_report" in st.session_state:
        st.markdown('<div class="section-header">📊 Gap Analysis Report</div>', unsafe_allow_html=True)
        st.markdown(st.session_state["gap_report"])

    if "opportunities" in st.session_state:
        st.markdown('<div class="section-header">🎯 Top Opportunities</div>', unsafe_allow_html=True)
        st.markdown(st.session_state["opportunities"])

    # ── strategy advisor ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📝 Application Strategy Advisor</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="agent-card">
        <h4>🎯 Strategy Advisor</h4>
        <p style="color: #64748b; font-size: 0.85rem;">
            Paste a job description below. The AI will tell you whether to 
            APPLY NOW, WAIT, or SKIP — based on your skills + market timing.
        </p>
    </div>
    """, unsafe_allow_html=True)

    jd_text = st.text_area(
        "Paste job description here",
        height=200,
        placeholder="Paste the full job description from LinkedIn, Naukri, etc.",
    )

    if st.button("🎯 Get Strategy Advice", disabled=not (has_crewai and has_groq and jd_text),
                  use_container_width=True):
        with st.spinner("🎯 Analysing job description..."):
            try:
                from agents.strategist import run_strategy_analysis

                student_skills = []
                if resume_text:
                    student_skills = [s.strip() for s in resume_text.split() if len(s) > 3][:20]

                fc_data = []
                if not forecast_summary.empty:
                    for _, row in forecast_summary.iterrows():
                        fc_data.append({
                            "skill": row.get("skill", ""),
                            "trend": row.get("trend", "STABLE"),
                            "change_pct": row.get("change_pct", 0),
                        })

                result = run_strategy_analysis(jd_text, student_skills, fc_data)
                st.session_state["strategy"] = result
            except Exception as e:
                st.error(f"Agent error: {e}")

    if "strategy" in st.session_state:
        st.markdown('<div class="section-header">📋 Strategy Recommendation</div>', unsafe_allow_html=True)
        st.markdown(st.session_state["strategy"])
