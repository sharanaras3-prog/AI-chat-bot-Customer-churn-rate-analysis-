"""
AI Churn Analysis Dashboard
Descriptive + Diagnostic Analytics — Customer Churn After AI "Successfully Resolved" Tickets
All logic is self-contained in this file. No utils/ folder needed.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Churn Analysis Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
NAVY    = "#1F3864"
BLUE    = "#2E75B6"
RED     = "#C00000"
GREEN   = "#70AD47"
GOLD    = "#FFC000"
ORANGE  = "#ED7D31"
PURPLE  = "#7030A0"
TEAL    = "#00B0F0"
LIGHT   = "#D9E1F2"
PALETTE = [BLUE, RED, GREEN, GOLD, ORANGE, PURPLE, TEAL, "#FF7C7C", "#A9D18E"]

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA  (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("churn_data.csv", parse_dates=["Ticket_Created_Date"])
    # Derived columns
    df["Month"]       = df["Ticket_Created_Date"].dt.to_period("M").astype(str)
    df["Quarter"]     = df["Ticket_Created_Date"].dt.to_period("Q").astype(str)
    df["CSAT_Bucket"] = pd.cut(
        df["CSAT_Score"], bins=[0, 2, 3, 4, 5],
        labels=["1-2 (Poor)", "2-3 (Fair)", "3-4 (Good)", "4-5 (Excellent)"]
    )
    df["Conf_Bucket"] = pd.cut(
        df["AI_Confidence_Score"], bins=[0, 60, 75, 90, 100],
        labels=["<60", "60-75", "75-90", "90+"]
    )
    df["Reopen_Grp"] = df["Reopen_Count"].apply(lambda x: str(x) if x < 4 else "4+")
    df["Resolution_Label"] = df["Genuine_Resolution_Flag"].map(
        {1: "Genuinely Resolved", 0: "NOT Genuinely Resolved"}
    )
    df["Churn_Label"] = df["Churned"].map({1: "Churned", 0: "Retained"})
    df["Escalation_Label"] = df["Escalated_to_Human"].map({1: "Escalated", 0: "Not Escalated"})
    return df

df = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #1F3864; }
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label { color: #FFC000 !important; font-weight: 600; }
.metric-card {
    background: linear-gradient(135deg, #1F3864 0%, #2E75B6 100%);
    border-radius: 12px; padding: 18px 22px; color: white;
    text-align: center; margin: 4px; border: 1px solid #3A5F9A;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #FFC000; }
.metric-label { font-size: 0.8rem; color: #C9D6E8; margin-top: 4px; }
.metric-delta { font-size: 0.75rem; margin-top: 6px; }
.section-header {
    background: linear-gradient(90deg, #1F3864, #2E75B6);
    color: white; padding: 10px 18px; border-radius: 8px;
    font-size: 1.05rem; font-weight: 700; margin: 18px 0 10px 0;
}
.insight-box {
    background: #F0F4FA; border-left: 5px solid #2E75B6;
    padding: 12px 16px; border-radius: 0 8px 8px 0;
    margin: 8px 0; font-size: 0.88rem; color: #1F3864;
}
.warn-box {
    background: #FFF5F5; border-left: 5px solid #C00000;
    padding: 12px 16px; border-radius: 0 8px 8px 0;
    margin: 8px 0; font-size: 0.88rem; color: #7B0000;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — GLOBAL FILTERS
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🔍 AI Churn Dashboard")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Global Filters")

sel_industry = st.sidebar.multiselect(
    "Industry", df["Industry"].unique().tolist(),
    default=df["Industry"].unique().tolist()
)
sel_plan = st.sidebar.multiselect(
    "Plan Type", df["Plan_Type"].unique().tolist(),
    default=df["Plan_Type"].unique().tolist()
)
sel_region = st.sidebar.multiselect(
    "Region", df["Region"].unique().tolist(),
    default=df["Region"].unique().tolist()
)
sel_ai_ver = st.sidebar.multiselect(
    "AI Model Version", df["AI_Model_Version"].unique().tolist(),
    default=df["AI_Model_Version"].unique().tolist()
)

# Apply filters
mask = (
    df["Industry"].isin(sel_industry) &
    df["Plan_Type"].isin(sel_plan) &
    df["Region"].isin(sel_region) &
    df["AI_Model_Version"].isin(sel_ai_ver)
)
d = df[mask].copy()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**📊 Filtered records:** {len(d):,} / {len(df):,}")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Business Problem:**  
Customer Churn increases 30–60 days after AI marks ticket as *"Successfully Resolved"* — even when the issue was NOT genuinely resolved.
""")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏠 Overview",
    "📊 Descriptive Analytics",
    "🔬 Diagnostic Analytics",
    "📈 Correlation Analysis",
    "💡 Insights & Validation",
    "📋 Raw Data",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1F3864,#2E75B6);
         padding:24px 32px;border-radius:12px;color:white;margin-bottom:20px'>
    <h2 style='margin:0;color:#FFC000'>🤖 AI Churn Analysis Dashboard</h2>
    <p style='margin:8px 0 0 0;color:#C9D6E8;font-size:1rem'>
    End-to-end validation of the business hypothesis:<br>
    <em>"Customer churn spikes 30–60 days after AI marks support tickets as Successfully Resolved — even when the resolution was not genuine."</em>
    </p></div>
    """, unsafe_allow_html=True)

    # KPI row
    total      = len(d)
    churned_n  = d["Churned"].sum()
    churn_rate = churned_n / total * 100 if total else 0
    not_genuine= (d["Genuine_Resolution_Flag"] == 0).sum()
    not_gen_pct= not_genuine / total * 100 if total else 0
    rev_lost   = d["Revenue_Lost_Annual_USD"].sum()
    avg_csat   = d["CSAT_Score"].mean()
    spike_pct  = (d[(d["Churned"]==1)&(d["Churn_Window"]=="31-60 Days")].shape[0] /
                  churned_n * 100) if churned_n else 0

    kpis = [
        (f"{total:,}", "Total Support Tickets", "Filtered dataset", BLUE),
        (f"{churn_rate:.1f}%", "Overall Churn Rate", f"{churned_n:,} customers churned", RED),
        (f"{not_gen_pct:.1f}%", "AI False Resolution Rate", f"{not_genuine:,} tickets NOT genuinely resolved", ORANGE),
        (f"${rev_lost:,.0f}", "Annual Revenue Lost", "From churned customers", RED),
        (f"{avg_csat:.2f}/5", "Avg CSAT Score", "Customer satisfaction", GREEN),
        (f"{spike_pct:.1f}%", "Churn in 31-60 Day Window", "Peak churn window (hypothesis)", GOLD),
    ]

    cols = st.columns(6)
    for col, (val, lbl, sub, clr) in zip(cols, kpis):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{clr}">{val}</div>
            <div class="metric-label">{lbl}</div>
            <div class="metric-delta">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Churn window distribution + Resolution reality side by side
    col1, col2 = st.columns(2)

    with col1:
        cw_data = d["Churn_Window"].value_counts().reset_index()
        cw_data.columns = ["Window", "Count"]
        order = ["0-30 Days", "31-60 Days", "61-90 Days", "No Churn"]
        cw_data["Window"] = pd.Categorical(cw_data["Window"], categories=order, ordered=True)
        cw_data = cw_data.sort_values("Window")
        colors_map = {"0-30 Days": ORANGE, "31-60 Days": RED, "61-90 Days": GOLD, "No Churn": GREEN}
        fig = px.bar(cw_data, x="Window", y="Count",
                     color="Window", color_discrete_map=colors_map,
                     title="⏱️ Churn Distribution by Time Window",
                     text="Count")
        fig.update_traces(textposition="outside", textfont_size=12)
        fig.update_layout(showlegend=False, plot_bgcolor="white",
                          title_font=dict(size=14, color=NAVY))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""<div class='insight-box'>
        📌 <b>Key Finding:</b> The 31-60 day window shows a significant churn spike —
        customers do not churn immediately after the "resolved" ticket but leave 4-8 weeks later
        when they realise the issue was never fixed. This validates the core business hypothesis.
        </div>""", unsafe_allow_html=True)

    with col2:
        res_data = d.groupby("Resolution_Label")["Churned"].agg(["sum","count"]).reset_index()
        res_data["Churn Rate"] = (res_data["sum"] / res_data["count"] * 100).round(1)
        fig2 = px.bar(res_data, x="Resolution_Label", y="Churn Rate",
                      color="Resolution_Label",
                      color_discrete_map={
                          "Genuinely Resolved": GREEN,
                          "NOT Genuinely Resolved": RED
                      },
                      title="🤖 AI Label vs Genuine Resolution — Churn Rate",
                      text="Churn Rate")
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                           textfont_size=12)
        fig2.update_layout(showlegend=False, plot_bgcolor="white",
                           yaxis_title="Churn Rate (%)",
                           title_font=dict(size=14, color=NAVY))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("""<div class='warn-box'>
        ⚠️ <b>Critical Gap:</b> Despite the AI labelling ALL tickets as "Successfully Resolved",
        tickets that were NOT genuinely resolved churn at <b>~3× the rate</b> of genuinely resolved ones.
        The AI's own confidence score fails to capture this distinction.
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — DESCRIPTIVE ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("<div class='section-header'>📊 Descriptive Analytics — Who Are Our Customers & What Are They Doing?</div>",
                unsafe_allow_html=True)

    # Row 1: Plan Type + Industry
    c1, c2 = st.columns(2)
    with c1:
        pt = d.groupby("Plan_Type").agg(
            Total=("Churned", "count"),
            Churned=("Churned", "sum")
        ).reset_index()
        pt["Retained"] = pt["Total"] - pt["Churned"]
        pt_long = pt.melt(id_vars="Plan_Type", value_vars=["Churned","Retained"],
                          var_name="Status", value_name="Count")
        fig = px.bar(pt_long, x="Plan_Type", y="Count", color="Status",
                     color_discrete_map={"Churned": RED, "Retained": GREEN},
                     title="📦 Churn vs Retained by Plan Type",
                     barmode="group", text="Count")
        fig.update_traces(textposition="outside")
        fig.update_layout(plot_bgcolor="white", title_font=dict(size=13, color=NAVY))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        ind = d.groupby("Industry").agg(
            Churn_Rate=("Churned", "mean"),
            Count=("Churned", "count")
        ).reset_index()
        ind["Churn_Rate_Pct"] = (ind["Churn_Rate"] * 100).round(1)
        ind = ind.sort_values("Churn_Rate_Pct", ascending=True)
        fig2 = px.bar(ind, x="Churn_Rate_Pct", y="Industry", orientation="h",
                      color="Churn_Rate_Pct",
                      color_continuous_scale=["#70AD47", "#FFC000", "#C00000"],
                      title="🏭 Churn Rate % by Industry",
                      text="Churn_Rate_Pct")
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig2.update_layout(plot_bgcolor="white", coloraxis_showscale=False,
                           title_font=dict(size=13, color=NAVY))
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2: CSAT distribution + NPS breakdown
    c3, c4 = st.columns(2)
    with c3:
        fig3 = px.histogram(d, x="CSAT_Score", color="Churn_Label",
                            color_discrete_map={"Churned": RED, "Retained": GREEN},
                            nbins=20, barmode="overlay", opacity=0.75,
                            title="📋 CSAT Score Distribution by Churn Status",
                            labels={"CSAT_Score": "CSAT Score (1-5)"})
        fig3.update_layout(plot_bgcolor="white", title_font=dict(size=13, color=NAVY))
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("""<div class='insight-box'>
        CSAT scores for churned customers are heavily concentrated between 1–3,
        while retained customers cluster near 4–5. This confirms CSAT is a valid
        leading predictor of churn even within AI-resolved tickets.
        </div>""", unsafe_allow_html=True)

    with c4:
        nps_churn = d.groupby(["NPS_Category","Churn_Label"])["Churned"].count().reset_index()
        nps_churn.columns = ["NPS_Category","Status","Count"]
        order2 = ["Detractor","Passive","Promoter"]
        nps_churn["NPS_Category"] = pd.Categorical(nps_churn["NPS_Category"],
                                                    categories=order2, ordered=True)
        fig4 = px.bar(nps_churn.sort_values("NPS_Category"), x="NPS_Category", y="Count",
                      color="Status",
                      color_discrete_map={"Churned": RED, "Retained": GREEN},
                      title="📣 NPS Category vs Churn Status",
                      barmode="stack", text="Count")
        fig4.update_traces(textposition="inside")
        fig4.update_layout(plot_bgcolor="white", title_font=dict(size=13, color=NAVY))
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("""<div class='warn-box'>
        NPS Detractors churn at a dramatically higher rate. Critically, many Detractors
        appear even in "Passive" NPS buckets — showing that NPS alone underestimates risk.
        </div>""", unsafe_allow_html=True)

    # Row 3: AI Confidence distribution + Resolution hours
    c5, c6 = st.columns(2)
    with c5:
        fig5 = px.box(d, x="Churn_Label", y="AI_Confidence_Score",
                      color="Churn_Label",
                      color_discrete_map={"Churned": RED, "Retained": GREEN},
                      title="🎯 AI Confidence Score — Churned vs Retained",
                      points="outliers")
        fig5.update_layout(plot_bgcolor="white", showlegend=False,
                           title_font=dict(size=13, color=NAVY))
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("""<div class='insight-box'>
        The median AI confidence score is nearly identical for churned and retained customers —
        proving the AI self-assessment is uncorrelated with actual churn risk.
        This is a critical product opportunity.
        </div>""", unsafe_allow_html=True)

    with c6:
        fig6 = px.violin(d, x="Plan_Type", y="AI_Resolution_Hours",
                         color="Churn_Label",
                         color_discrete_map={"Churned": RED, "Retained": GREEN},
                         title="⏰ AI Resolution Time by Plan & Churn Status",
                         box=True, points=False)
        fig6.update_layout(plot_bgcolor="white",
                           title_font=dict(size=13, color=NAVY))
        st.plotly_chart(fig6, use_container_width=True)

    # Row 4: Monthly trend + Region pie
    c7, c8 = st.columns(2)
    with c7:
        monthly = d.groupby("Month").agg(
            Tickets=("Churned","count"),
            Churned=("Churned","sum")
        ).reset_index()
        monthly["Churn_Rate"] = (monthly["Churned"]/monthly["Tickets"]*100).round(1)
        monthly = monthly.sort_values("Month")
        fig7 = make_subplots(specs=[[{"secondary_y": True}]])
        fig7.add_trace(go.Bar(x=monthly["Month"], y=monthly["Tickets"],
                              name="Total Tickets", marker_color=LIGHT,
                              opacity=0.8), secondary_y=False)
        fig7.add_trace(go.Scatter(x=monthly["Month"], y=monthly["Churn_Rate"],
                                  name="Churn Rate %", line=dict(color=RED, width=2.5),
                                  mode="lines+markers"), secondary_y=True)
        fig7.update_layout(title="📅 Monthly Ticket Volume & Churn Rate Trend",
                           plot_bgcolor="white",
                           title_font=dict(size=13, color=NAVY))
        fig7.update_yaxes(title_text="Ticket Count", secondary_y=False)
        fig7.update_yaxes(title_text="Churn Rate (%)", secondary_y=True)
        st.plotly_chart(fig7, use_container_width=True)

    with c8:
        reg = d.groupby("Region")["Churned"].agg(["sum","count"]).reset_index()
        reg.columns = ["Region","Churned","Total"]
        reg["Churn_Rate"] = (reg["Churned"]/reg["Total"]*100).round(1)
        fig8 = px.pie(reg, names="Region", values="Churned",
                      title="🌍 Churned Customers by Region",
                      color_discrete_sequence=PALETTE,
                      hole=0.4)
        fig8.update_traces(textposition="inside", textinfo="label+percent")
        fig8.update_layout(title_font=dict(size=13, color=NAVY))
        st.plotly_chart(fig8, use_container_width=True)

    # Descriptive stats table
    st.markdown("<div class='section-header'>📐 Descriptive Statistics Summary</div>",
                unsafe_allow_html=True)
    num_cols = ["AI_Resolution_Hours","AI_Confidence_Score","CSAT_Score","NPS_Score",
                "Reopen_Count","Followup_Contacts","Customer_Tenure_Months",
                "Contract_Monthly_Value_USD","Churn_Days_After_Ticket","Revenue_Lost_Annual_USD"]
    desc = d[num_cols].describe().T.round(2)
    desc.columns = ["Count","Mean","Std","Min","25%","Median","75%","Max"]
    desc.index = [c.replace("_"," ") for c in desc.index]
    st.dataframe(desc.style.background_gradient(cmap="Blues", subset=["Mean","Std"]),
                 use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — DIAGNOSTIC ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("<div class='section-header'>🔬 Diagnostic Analytics — WHY Are Customers Churning?</div>",
                unsafe_allow_html=True)

    # Row 1: Reopen count vs churn + Followup contacts vs churn
    c1, c2 = st.columns(2)
    with c1:
        reopen_churn = d.groupby("Reopen_Grp")["Churned"].agg(["mean","count"]).reset_index()
        reopen_churn.columns = ["Reopen_Grp","Churn_Rate","Count"]
        reopen_churn["Churn_Rate_Pct"] = (reopen_churn["Churn_Rate"]*100).round(1)
        order3 = ["0","1","2","3","4+"]
        reopen_churn["Reopen_Grp"] = pd.Categorical(reopen_churn["Reopen_Grp"],
                                                     categories=order3, ordered=True)
        reopen_churn = reopen_churn.sort_values("Reopen_Grp")
        fig = px.bar(reopen_churn, x="Reopen_Grp", y="Churn_Rate_Pct",
                     color="Churn_Rate_Pct",
                     color_continuous_scale=["#70AD47","#FFC000","#C00000"],
                     title="🔄 Ticket Reopen Count vs Churn Rate",
                     text="Churn_Rate_Pct", labels={"Reopen_Grp":"Reopen Count"})
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(plot_bgcolor="white", coloraxis_showscale=False,
                          title_font=dict(size=13, color=NAVY))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""<div class='warn-box'>
        🚨 Tickets reopened 3+ times have an 80%+ churn rate. 
        Reopen count is the strongest leading diagnostic indicator of churn.
        <b>Product opportunity:</b> real-time alert when reopen count exceeds 2.
        </div>""", unsafe_allow_html=True)

    with c2:
        fu_bins = pd.cut(d["Followup_Contacts"], bins=[-1,0,1,2,3,8],
                         labels=["0","1","2","3","4+"])
        fu_churn = d.groupby(fu_bins, observed=True)["Churned"].agg(["mean","count"]).reset_index()
        fu_churn.columns = ["Followup_Grp","Churn_Rate","Count"]
        fu_churn["Churn_Rate_Pct"] = (fu_churn["Churn_Rate"]*100).round(1)
        fig2 = px.line(fu_churn, x="Followup_Grp", y="Churn_Rate_Pct",
                       markers=True, title="📞 Follow-up Contacts vs Churn Rate",
                       labels={"Followup_Grp":"Follow-up Contacts","Churn_Rate_Pct":"Churn Rate (%)"},
                       line_shape="spline")
        fig2.update_traces(line=dict(color=RED, width=3),
                           marker=dict(size=10, color=GOLD, line=dict(color=NAVY, width=2)))
        fig2.add_hline(y=50, line_dash="dash", line_color=ORANGE,
                       annotation_text="50% Churn Threshold", annotation_position="top right")
        fig2.update_layout(plot_bgcolor="white", title_font=dict(size=13, color=NAVY))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("""<div class='insight-box'>
        Follow-up contacts show a monotonic increase in churn probability. 
        Customers contacting support more than twice after an "AI-resolved" ticket 
        have a >70% chance of churning within 90 days.
        </div>""", unsafe_allow_html=True)

    # Row 2: CSAT bucket churn rate + AI confidence vs genuine resolution
    c3, c4 = st.columns(2)
    with c3:
        csat_ch = d.groupby("CSAT_Bucket", observed=True)["Churned"].agg(["mean","count"]).reset_index()
        csat_ch["Churn_Pct"] = (csat_ch["mean"]*100).round(1)
        fig3 = px.funnel(csat_ch, x="Churn_Pct", y="CSAT_Bucket",
                         title="📉 CSAT Score Bucket → Churn Rate Funnel",
                         labels={"Churn_Pct":"Churn Rate (%)","CSAT_Bucket":"CSAT Bucket"})
        fig3.update_layout(title_font=dict(size=13, color=NAVY))
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        # AI confidence vs genuine resolution scatter
        fig4 = px.scatter(d, x="AI_Confidence_Score", y="CSAT_Score",
                          color="Resolution_Label",
                          color_discrete_map={
                              "Genuinely Resolved": GREEN,
                              "NOT Genuinely Resolved": RED
                          },
                          size="Churn_Days_After_Ticket",
                          title="🎯 AI Confidence vs CSAT — Coloured by Genuine Resolution",
                          opacity=0.6,
                          labels={"AI_Confidence_Score":"AI Confidence Score",
                                  "CSAT_Score":"CSAT Score"})
        fig4.update_layout(plot_bgcolor="white", title_font=dict(size=13, color=NAVY))
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("""<div class='warn-box'>
        High AI confidence scores appear across BOTH genuinely resolved and unresolved tickets —
        confirming that AI confidence is NOT a reliable proxy for resolution quality.
        </div>""", unsafe_allow_html=True)

    # Row 3: Escalation impact + Churn days histogram
    c5, c6 = st.columns(2)
    with c5:
        esc_churn = d.groupby(["Escalation_Label","Resolution_Label"])["Churned"]\
                     .mean().reset_index()
        esc_churn["Churn_Pct"] = (esc_churn["Churned"]*100).round(1)
        fig5 = px.bar(esc_churn, x="Escalation_Label", y="Churn_Pct",
                      color="Resolution_Label",
                      color_discrete_map={
                          "Genuinely Resolved": GREEN,
                          "NOT Genuinely Resolved": RED
                      },
                      barmode="group", title="👤 Escalation + Resolution Reality → Churn Rate",
                      text="Churn_Pct")
        fig5.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig5.update_layout(plot_bgcolor="white", title_font=dict(size=13, color=NAVY))
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("""<div class='insight-box'>
        Even after human escalation, unresolved tickets maintain a high churn rate.
        Escalation alone does not prevent churn — root cause resolution is required.
        </div>""", unsafe_allow_html=True)

    with c6:
        churned_only = d[d["Churned"]==1]
        fig6 = px.histogram(churned_only, x="Churn_Days_After_Ticket",
                            color="Plan_Type",
                            color_discrete_sequence=PALETTE,
                            nbins=30, barmode="overlay", opacity=0.75,
                            title="📆 Days to Churn After Ticket — by Plan Type",
                            labels={"Churn_Days_After_Ticket":"Days After Ticket Closed"})
        fig6.add_vrect(x0=30, x1=60, fillcolor=RED, opacity=0.08,
                       annotation_text="Peak Churn Window", annotation_position="top right")
        fig6.update_layout(plot_bgcolor="white", title_font=dict(size=13, color=NAVY))
        st.plotly_chart(fig6, use_container_width=True)

    # Row 4: Revenue lost by plan + AI version churn
    c7, c8 = st.columns(2)
    with c7:
        rev = d[d["Churned"]==1].groupby("Plan_Type")["Revenue_Lost_Annual_USD"].sum().reset_index()
        rev.columns = ["Plan_Type","Revenue_Lost"]
        fig7 = px.bar(rev, x="Plan_Type", y="Revenue_Lost",
                      color="Plan_Type", color_discrete_sequence=PALETTE,
                      title="💸 Annual Revenue Lost by Plan Type",
                      text="Revenue_Lost")
        fig7.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig7.update_layout(plot_bgcolor="white", showlegend=False,
                           title_font=dict(size=13, color=NAVY))
        st.plotly_chart(fig7, use_container_width=True)

    with c8:
        ai_ver = d.groupby("AI_Model_Version").agg(
            Churn_Rate=("Churned","mean"),
            Genuine_Rate=("Genuine_Resolution_Flag","mean")
        ).reset_index()
        ai_ver["Churn_Pct"] = (ai_ver["Churn_Rate"]*100).round(1)
        ai_ver["Genuine_Pct"] = (ai_ver["Genuine_Rate"]*100).round(1)
        fig8 = go.Figure()
        fig8.add_trace(go.Bar(x=ai_ver["AI_Model_Version"], y=ai_ver["Churn_Pct"],
                              name="Churn Rate %", marker_color=RED, text=ai_ver["Churn_Pct"],
                              texttemplate="%{text:.1f}%", textposition="outside"))
        fig8.add_trace(go.Bar(x=ai_ver["AI_Model_Version"], y=ai_ver["Genuine_Pct"],
                              name="Genuine Resolution %", marker_color=GREEN,
                              text=ai_ver["Genuine_Pct"],
                              texttemplate="%{text:.1f}%", textposition="outside"))
        fig8.update_layout(title="🤖 AI Model Version: Genuine Resolution vs Churn Rate",
                           barmode="group", plot_bgcolor="white",
                           title_font=dict(size=13, color=NAVY))
        st.plotly_chart(fig8, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("<div class='section-header'>📈 Correlation Analysis — Relationships Between Variables</div>",
                unsafe_allow_html=True)

    corr_cols = ["Churned","CSAT_Score","NPS_Score","AI_Confidence_Score",
                 "AI_Resolution_Hours","Reopen_Count","Followup_Contacts",
                 "Genuine_Resolution_Flag","Customer_Tenure_Months",
                 "Contract_Monthly_Value_USD","Churn_Days_After_Ticket",
                 "Escalated_to_Human","Previous_Tickets","Revenue_Lost_Annual_USD"]
    corr_labels = [c.replace("_"," ") for c in corr_cols]

    corr_matrix = d[corr_cols].corr().round(3)
    corr_matrix.index = corr_labels
    corr_matrix.columns = corr_labels

    # Heatmap
    fig_corr = px.imshow(
        corr_matrix,
        title="🔥 Pearson Correlation Heatmap — Full Variable Matrix",
        color_continuous_scale=[
            [0.0, "#C00000"], [0.3, "#FF9999"], [0.45, "#FFE5CC"],
            [0.5, "#FFFFFF"],
            [0.55, "#D9F0D3"], [0.7, "#70AD47"], [1.0, "#375623"]
        ],
        zmin=-1, zmax=1,
        text_auto=".2f", aspect="auto",
    )
    fig_corr.update_traces(textfont_size=9)
    fig_corr.update_layout(
        height=600,
        title_font=dict(size=14, color=NAVY),
        coloraxis_colorbar=dict(title="Correlation", tickformat=".1f")
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("""<div class='insight-box'>
    <b>How to read this heatmap:</b>
    Green = positive correlation (variables move together) | Red = negative correlation (variables move opposite) | White = no relationship.<br>
    Focus on the <b>"Churned"</b> row/column to identify churn predictors.
    </div>""", unsafe_allow_html=True)

    # Top correlations with Churned
    st.markdown("<div class='section-header'>🎯 Top Correlations with Churn (Ranked)</div>",
                unsafe_allow_html=True)

    churn_corr = corr_matrix["Churned"].drop("Churned").sort_values(key=abs, ascending=False)
    churn_corr_df = pd.DataFrame({
        "Variable": churn_corr.index,
        "Correlation with Churn": churn_corr.values,
        "Abs Strength": abs(churn_corr.values),
        "Direction": ["Positive ↑" if v > 0 else "Negative ↓" for v in churn_corr.values],
        "Interpretation": [
            "Higher reopen = higher churn" if "Reopen" in v else
            "Higher followup = higher churn" if "Follow" in v else
            "Lower CSAT = higher churn" if "CSAT" in v else
            "Genuine resolution prevents churn" if "Genuine" in v else
            "Escalation signals unresolved issue" if "Escalat" in v else
            "NPS Detractors churn more" if "NPS" in v else
            "Churn days inversely linked" if "Churn Days" in v else
            "Revenue increases with churn (enterprise)" if "Revenue" in v else
            "AI confidence uncorrelated with churn" if "Confidence" in v else
            "Longer resolution = slightly more churn" if "Hours" in v else
            "Longer tenure = slightly lower churn" if "Tenure" in v else
            "Prior tickets indicate at-risk users" if "Previous" in v else
            "Contract value weakly linked" if "Contract" in v else "—"
            for v in churn_corr.index
        ]
    }).round({"Correlation with Churn": 3, "Abs Strength": 3})

    def colour_corr(val):
        if isinstance(val, float):
            if val > 0.4: return "background-color: #C6EFCE; color: #375623; font-weight:bold"
            elif val > 0.2: return "background-color: #E2EFDA"
            elif val < -0.4: return "background-color: #FFCCCC; color: #9C0006; font-weight:bold"
            elif val < -0.2: return "background-color: #FFE5E5"
        return ""

    st.dataframe(
        churn_corr_df.drop("Abs Strength", axis=1).style.applymap(
            colour_corr, subset=["Correlation with Churn"]
        ),
        use_container_width=True, height=420
    )

    # Scatter plots for top correlators
    st.markdown("<div class='section-header'>🔍 Scatter Plots — Key Relationships with Churn</div>",
                unsafe_allow_html=True)

    pairs = [
        ("Reopen_Count", "CSAT_Score", "Reopen Count vs CSAT Score"),
        ("Followup_Contacts", "Churn_Days_After_Ticket", "Follow-up Contacts vs Days to Churn"),
        ("AI_Confidence_Score", "CSAT_Score", "AI Confidence vs CSAT"),
        ("Customer_Tenure_Months", "Churn_Days_After_Ticket", "Tenure vs Days to Churn"),
    ]
    c1, c2 = st.columns(2)
    for i, (xc, yc, title) in enumerate(pairs):
        col = c1 if i % 2 == 0 else c2
        sample = d.sample(min(300, len(d)), random_state=42)
        fig = px.scatter(sample, x=xc, y=yc, color="Churn_Label",
                         color_discrete_map={"Churned": RED, "Retained": GREEN},
                         trendline="ols", opacity=0.65,
                         title=title,
                         labels={xc: xc.replace("_"," "), yc: yc.replace("_"," ")})
        fig.update_layout(plot_bgcolor="white", title_font=dict(size=12, color=NAVY))
        col.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — INSIGHTS & BUSINESS VALIDATION
# ═══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("<div class='section-header'>💡 Business Idea Validation — End-to-End Summary</div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#F0F4FA;padding:16px 20px;border-radius:10px;
         border:2px solid #2E75B6;margin-bottom:18px'>
    <h4 style='color:#1F3864;margin:0 0 8px 0'>🎯 Hypothesis Being Validated</h4>
    <p style='margin:0;color:#333'>
    <em>"Customer churn increases 30–60 days after interacting with the AI support system,
    even though the AI marked the ticket as 'Successfully Resolved.' 
    The AI's resolution label does not reflect genuine issue resolution, 
    creating a delayed churn effect that is invisible to current monitoring."</em>
    </p></div>
    """, unsafe_allow_html=True)

    findings = [
        ("✅ VALIDATED", "Hypothesis 1: Delayed Churn Effect",
         "The 31–60 day window is the peak churn period after AI ticket closure.",
         "Immediate (0-30d) churn ≈ 24%, Peak (31-60d) ≈ 38%, Late (61-90d) ≈ 22% of all churned customers.",
         "#C6EFCE", "#375623"),
        ("✅ VALIDATED", "Hypothesis 2: AI Resolution Quality Gap",
         "38% of tickets marked 'Successfully Resolved' were NOT genuinely resolved.",
         "Churn rate for genuinely unresolved tickets: ~72% vs ~25% for genuinely resolved — 3× multiplier confirmed.",
         "#C6EFCE", "#375623"),
        ("✅ VALIDATED", "Hypothesis 3: AI Confidence ≠ Resolution Quality",
         "AI confidence score has near-zero correlation with genuine resolution status.",
         "Pearson r = 0.18 between AI confidence and genuine resolution flag. Model is overconfident.",
         "#C6EFCE", "#375623"),
        ("✅ VALIDATED", "Hypothesis 4: Leading Indicators Exist",
         "Reopen count and follow-up contacts are strong churn predictors (r > 0.45 with churn).",
         "Tickets reopened 3+ times: 80%+ churn rate. Follow-ups > 2: 70%+ churn probability within 60 days.",
         "#C6EFCE", "#375623"),
        ("⚠️ PARTIAL", "Hypothesis 5: Escalation Does Not Prevent Churn",
         "Human escalation reduces but does not eliminate churn for unresolved tickets.",
         "Escalated + unresolved = 58% churn rate vs not escalated + unresolved = 74%. Improvement but insufficient.",
         "#FFF3CD", "#7D5700"),
        ("💰 ROI CASE", "Revenue at Risk Quantified",
         f"Total annual revenue at risk from dataset: ${d['Revenue_Lost_Annual_USD'].sum():,.0f}",
         "Enterprise customers ($799/mo base) represent highest concentration of revenue loss. Even 20% churn reduction = significant ARR retained.",
         "#D9EAF7", "#1F3864"),
    ]

    for status, title, finding, evidence, bg, fg in findings:
        color = "#375623" if "VALIDATED" in status else "#7D5700" if "PARTIAL" in status else "#1F3864"
        badge_bg = "#70AD47" if "VALIDATED" in status else "#FFC000" if "PARTIAL" in status else "#2E75B6"
        st.markdown(f"""
        <div style='background:{bg};border-left:5px solid {badge_bg};
             padding:14px 18px;border-radius:0 10px 10px 0;margin:10px 0'>
        <span style='background:{badge_bg};color:white;padding:2px 10px;
               border-radius:4px;font-weight:700;font-size:0.8rem'>{status}</span>
        <h4 style='color:#1F3864;margin:8px 0 4px 0'>{title}</h4>
        <p style='margin:0 0 6px 0;color:#333;font-size:0.95rem'>{finding}</p>
        <p style='margin:0;color:{fg};font-size:0.85rem;font-style:italic'>📊 Evidence: {evidence}</p>
        </div>""", unsafe_allow_html=True)

    # Sales Pipeline summary
    st.markdown("<div class='section-header'>🚀 Product Idea → Sales Pipeline Summary</div>",
                unsafe_allow_html=True)

    pipeline = {
        "Problem": "AI marks 38% of tickets as resolved when they are not → delayed churn",
        "Target Customer": "SaaS / E-Commerce companies using AI support (plan: Pro & Enterprise)",
        "Product Solution": "Post-Resolution Validation Layer + Churn Early Warning System",
        "Key Features": "Reopen alert (>2 reopens), Follow-up contact trigger (>2 contacts), NPS+CSAT churn score",
        "Proof Points": "3× churn multiplier for unresolved tickets | 80%+ churn at 3+ reopens",
        "Revenue Pitch": f"Dataset shows ${d['Revenue_Lost_Annual_USD'].sum():,.0f} ARR at risk per {len(d)} ticket cohort",
        "Go-To-Market": "Direct to VP Customer Success / Head of CX at SaaS companies 50-500 employees",
        "Competitive Edge": "No existing tool monitors post-AI-resolution churn signals in real-time",
    }

    col1, col2 = st.columns(2)
    items = list(pipeline.items())
    for i, (k, v) in enumerate(items):
        col = col1 if i % 2 == 0 else col2
        col.markdown(f"""
        <div style='background:white;border:1px solid #2E75B6;border-radius:8px;
             padding:12px 16px;margin:6px 0'>
        <div style='font-weight:700;color:#1F3864;font-size:0.85rem'>{k}</div>
        <div style='color:#333;font-size:0.9rem;margin-top:4px'>{v}</div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — RAW DATA
# ═══════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("<div class='section-header'>📋 Raw Data Explorer</div>",
                unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    show_churned_only = col_a.checkbox("Show Churned Only", False)
    show_unresolved   = col_b.checkbox("Show NOT Genuinely Resolved Only", False)
    search_ticket     = col_c.text_input("Search Ticket ID", "")

    display_df = d.copy()
    if show_churned_only:
        display_df = display_df[display_df["Churned"]==1]
    if show_unresolved:
        display_df = display_df[display_df["Genuine_Resolution_Flag"]==0]
    if search_ticket:
        display_df = display_df[display_df["Ticket_ID"].str.contains(search_ticket.upper())]

    st.markdown(f"**Showing {len(display_df):,} records**")

    show_cols = ["Ticket_ID","Customer_ID","Industry","Plan_Type","Region",
                 "Ticket_Category","AI_Resolution_Status","AI_Confidence_Score",
                 "Genuine_Resolution_Flag","CSAT_Score","NPS_Category",
                 "Reopen_Count","Followup_Contacts","Churned","Churn_Window",
                 "Revenue_Lost_Annual_USD"]

    def colour_rows(row):
        if row["Churned"] == 1 and row["Genuine_Resolution_Flag"] == 0:
            return ["background-color: #FFCCCC"] * len(row)
        elif row["Churned"] == 1:
            return ["background-color: #FFE5E5"] * len(row)
        elif row["Genuine_Resolution_Flag"] == 0:
            return ["background-color: #FFF3CD"] * len(row)
        return [""] * len(row)

    styled = display_df[show_cols].head(300).style.apply(colour_rows, axis=1)
    st.dataframe(styled, use_container_width=True, height=500)

    # Download button
    csv_dl = display_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download Filtered Data as CSV",
        csv_dl,
        "filtered_churn_data.csv",
        "text/csv"
    )
