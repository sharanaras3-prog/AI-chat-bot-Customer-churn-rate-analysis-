# 🤖 AI Churn Analysis Dashboard

## Business Problem
Customer churn increases **30–60 days after** AI marks support tickets as *"Successfully Resolved"* — even when the issue was not genuinely resolved.

## What This Dashboard Does
- **Descriptive Analytics**: Distribution of churn by plan, industry, CSAT, NPS, region, AI model version, and time
- **Diagnostic Analytics**: Root cause analysis — reopen count, follow-up contacts, escalation patterns, revenue at risk
- **Correlation Analysis**: Full Pearson correlation heatmap + ranked churn predictors
- **Business Validation**: End-to-end hypothesis validation + sales pipeline summary

## Files
| File | Description |
|------|-------------|
| `app.py` | Main Streamlit dashboard (all-in-one, no utils folder) |
| `churn_data.csv` | 500-row synthetic dataset |
| `AI_Churn_Analysis.xlsx` | 7-sheet Excel workbook (raw data, cleaning log, descriptive stats, correlation matrix, pivot analytics, charts, insights) |
| `requirements.txt` | Python dependencies |

## Deploy on Streamlit Cloud
1. Fork / upload this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select your repo → Branch: `main` → Main file: `app.py`
4. Click **Deploy**

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dashboard Tabs
| Tab | Analytics Type | Key Charts |
|-----|---------------|------------|
| 🏠 Overview | Summary KPIs | Churn window distribution, AI label vs genuine resolution |
| 📊 Descriptive | Who & What | Plan/Industry/CSAT/NPS/Region/Monthly trend |
| 🔬 Diagnostic | Why | Reopen count, follow-up contacts, escalation paradox, revenue lost |
| 📈 Correlation | Relationships | Full heatmap, scatter plots, ranked predictor table |
| 💡 Insights | Validation | Hypothesis results, sales pipeline, ROI case |
| 📋 Raw Data | Explorer | Filterable data table with CSV download |

## Key Findings
- **38%** of AI-resolved tickets were NOT genuinely resolved
- **31-60 day** window is the peak churn period (validates hypothesis)
- **3× churn multiplier** for unresolved vs resolved tickets
- **Reopen count r=0.48** — strongest churn predictor
- **AI confidence score r=0.03** with genuine resolution — AI is overconfident and unreliable

## Data Dictionary
| Column | Description |
|--------|-------------|
| `Genuine_Resolution_Flag` | Hidden truth: was ticket actually resolved? (0=No, 1=Yes) |
| `AI_Resolution_Status` | Always "Successfully Resolved" — intentional AI bias |
| `AI_Confidence_Score` | AI self-reported confidence (30–99) — shown to be unreliable |
| `Churn_Window` | When churn occurred: 0-30d, 31-60d, 61-90d, or No Churn |
| `CSAT_Score` | Customer satisfaction 1–5 |
| `Reopen_Count` | Number of times ticket was reopened |
| `Followup_Contacts` | Support contacts after AI "resolution" |
| `Revenue_Lost_Annual_USD` | Annualised revenue lost (0 if not churned) |
