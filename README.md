# HireScope — India Job Market Intelligence

A job market forecasting engine for Indian students that aggregates LinkedIn & Naukri job postings, forecasts skill demand 90 days out, and provides personalised career intelligence through AI agents.

## What It Does

- **Aggregates** ~80K+ job postings from LinkedIn and Naukri into weekly skill demand counts
- **Forecasts** which skills will be in demand 90 days from now using Prophet + XGBoost
- **Explains** predictions using SHAP (which features drive each skill's forecast?)
- **3 AI Agents** (powered by CrewAI + Groq Llama 3.3 70B):
  - **Career Gap Analyst** — compares your resume to rising skills
  - **Opportunity Scout** — finds best-timed job opportunities
  - **Strategy Advisor** — tells you when to apply for a specific JD
- **Streamlit Dashboard** — interactive charts, resume upload, real-time analysis

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Groq API key (optional — dashboard works without it)
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 3. Run the dashboard
streamlit run app.py
```

## Project Structure

```
hirescope/
├── app.py                     → Streamlit dashboard
├── data/                      → CSV data files
├── src/
│   ├── data_loader.py         → Load + clean all CSVs
│   ├── aggregator.py          → Weekly skill demand counts
│   ├── forecaster.py          → Prophet + XGBoost models
│   ├── feature_engineer.py    → XGBoost feature extraction
│   └── evaluator.py           → SHAP explainability
├── agents/
│   ├── gap_analyser.py        → Career gap agent
│   ├── opportunity_scout.py   → Job scout agent
│   └── strategist.py          → Application timing agent
├── crew.py                    → CrewAI orchestration
├── requirements.txt
├── .env
└── README.md
```

## Tech Stack

| Component | Tool |
|-----------|------|
| Data Processing | pandas, numpy |
| Forecasting | Prophet, XGBoost |
| Explainability | SHAP |
| AI Agents | CrewAI + Groq (Llama 3.3 70B) |
| Resume Parsing | PyMuPDF |
| Dashboard | Streamlit + Plotly |
| Config | python-dotenv |

## Data Sources

- **LinkedIn Jobs** (~12K postings) — `job_postings.csv`, `job_skills.csv`, `job_summary.csv`
- **Naukri India** (~87K postings) — 3 CSV files covering Data Science & Analytics roles

## Notes

- The dashboard works fully without a Groq API key — forecasts and charts are always available
- AI agent features (resume analysis, opportunity scouting, strategy advice) require a valid `GROQ_API_KEY` in `.env`
- Built for Mac M2 / Apple Silicon
