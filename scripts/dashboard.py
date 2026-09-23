"""Streamlit dashboard for the Fraud Risk Agent.

Usage:
    streamlit run scripts/dashboard.py

Modes:
    - Live: connects to FRAUD_DATABASE_URL (set in .env or env vars)
    - Demo: generates synthetic data when no DB is available
"""

from __future__ import annotations

import datetime
import json
import random
import uuid

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Fraud Risk Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _generate_demo_data(n: int = 200) -> pd.DataFrame:
    """Generate synthetic fraud assessment data for demo mode."""
    rng = random.Random(42)
    verdicts = [None, None, None, "confirmed_fraud", "false_positive", "escalated"]
    signals_pool = [
        "return_rate_high", "high_risk_reasons", "cancel_after_ship",
        "promo_cancel_pattern", "linked_abuse", "return_value_ratio_high",
        "late_window_returns", "cancel_reorder_loop", "linked_accounts",
        "category_adjusted_rate", "rapid_returns",
    ]

    rows = []
    base = datetime.datetime(2026, 8, 1, tzinfo=datetime.UTC)
    for _i in range(n):
        score = max(0, min(100, int(rng.gauss(38, 25))))
        if score <= 39:
            risk_level = "low"
        elif score <= 69:
            risk_level = "medium"
        else:
            risk_level = "high"

        if risk_level == "low":
            action = "allow"
        elif risk_level == "medium":
            action = rng.choice(["inspect_return", "hold_refund_for_review"])
        else:
            action = rng.choice(["hold_refund_for_review", "escalate_to_analyst"])

        num_signals = min(score // 15, 5)
        signals = rng.sample(signals_pool, k=min(num_signals, len(signals_pool)))

        created = base + datetime.timedelta(
            days=rng.randint(0, 53), hours=rng.randint(0, 23), minutes=rng.randint(0, 59),
        )

        rows.append({
            "id": str(uuid.uuid4()),
            "customer_id": f"C-{rng.randint(1000, 9999)}",
            "assessment_id": str(uuid.uuid4()),
            "score": score,
            "risk_level": risk_level,
            "recommended_action": action,
            "confidence": round(rng.uniform(0.3, 0.98), 2),
            "signals": json.dumps([{"name": s} for s in signals]),
            "analyst_verdict": rng.choice(verdicts),
            "rules_version": "v1",
            "created_at": created,
            "batch_run_id": str(uuid.uuid4()) if rng.random() > 0.3 else None,
        })

    return pd.DataFrame(rows)


@st.cache_data(ttl=30)
def load_data() -> tuple[pd.DataFrame, bool]:
    """Load fraud cases from the database, falling back to demo data."""
    import os

    if not os.environ.get("FRAUD_DATABASE_URL"):
        return _generate_demo_data(), False

    try:
        import asyncio

        from sqlalchemy import text

        from fraud_risk_agent.data.connection import get_fraud_session

        async def _fetch():
            async with get_fraud_session() as session:
                result = await session.execute(text(
                    "SELECT id, customer_id, assessment_id, score, risk_level, "
                    "recommended_action, confidence, signals, analyst_verdict, "
                    "rules_version, created_at, batch_run_id "
                    "FROM fraud_cases ORDER BY created_at DESC LIMIT 1000"
                ))
                return [dict(row._mapping) for row in result.fetchall()]

        rows = asyncio.run(_fetch())
        if rows:
            return pd.DataFrame(rows), True
        return _generate_demo_data(), False
    except Exception:
        return _generate_demo_data(), False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Fraud Risk Agent")
st.sidebar.markdown("---")

df, is_live = load_data()
mode_label = "Live (DB)" if is_live else "Demo (synthetic data)"
st.sidebar.info(f"Mode: **{mode_label}**")

if not is_live:
    st.sidebar.caption(
        "No database connected. Showing synthetic data. "
        "Start the DB with `docker compose up db` and run `make migrate`."
    )

st.sidebar.markdown("---")
st.sidebar.markdown("**Filters**")

risk_filter = st.sidebar.multiselect(
    "Risk Level", ["low", "medium", "high"], default=["low", "medium", "high"],
)
action_filter = st.sidebar.multiselect(
    "Recommended Action",
    df["recommended_action"].unique().tolist(),
    default=df["recommended_action"].unique().tolist(),
)

score_range = st.sidebar.slider("Score Range", 0, 100, (0, 100))

filtered = df[
    (df["risk_level"].isin(risk_filter))
    & (df["recommended_action"].isin(action_filter))
    & (df["score"] >= score_range[0])
    & (df["score"] <= score_range[1])
]

if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ---------------------------------------------------------------------------
# Header metrics
# ---------------------------------------------------------------------------

st.title("Fraud Risk Dashboard")
st.caption(f"{len(filtered)} assessments shown  |  {mode_label}")

col1, col2, col3, col4, col5 = st.columns(5)

total = len(filtered)
high_risk = len(filtered[filtered["risk_level"] == "high"])
medium_risk = len(filtered[filtered["risk_level"] == "medium"])
avg_score = filtered["score"].mean() if total > 0 else 0
avg_confidence = filtered["confidence"].mean() if total > 0 else 0

col1.metric("Total Assessments", total)
col2.metric("High Risk", high_risk, delta=f"{high_risk/total*100:.0f}%" if total else "0%")
col3.metric("Medium Risk", medium_risk, delta=f"{medium_risk/total*100:.0f}%" if total else "0%")
col4.metric("Avg Score", f"{avg_score:.1f}")
col5.metric("Avg Confidence", f"{avg_confidence:.0%}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Charts row 1
# ---------------------------------------------------------------------------

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Score Distribution")
    fig_hist = px.histogram(
        filtered, x="score", nbins=20,
        color="risk_level",
        color_discrete_map={"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"},
        labels={"score": "Risk Score", "count": "Count"},
    )
    fig_hist.update_layout(
        bargap=0.05, height=350, margin=dict(t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with chart_col2:
    st.subheader("Risk Level Breakdown")
    risk_counts = filtered["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["risk_level", "count"]
    fig_pie = px.pie(
        risk_counts, values="count", names="risk_level",
        color="risk_level",
        color_discrete_map={"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"},
        hole=0.4,
    )
    fig_pie.update_layout(height=350, margin=dict(t=10, b=40))
    st.plotly_chart(fig_pie, use_container_width=True)

# ---------------------------------------------------------------------------
# Charts row 2
# ---------------------------------------------------------------------------

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.subheader("Recommended Actions")
    action_counts = filtered["recommended_action"].value_counts().reset_index()
    action_counts.columns = ["action", "count"]
    fig_bar = px.bar(
        action_counts, x="action", y="count",
        color="action",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"action": "Action", "count": "Count"},
    )
    fig_bar.update_layout(
        height=350, margin=dict(t=10, b=40), showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with chart_col4:
    st.subheader("Assessments Over Time")
    if "created_at" in filtered.columns and total > 0:
        daily = filtered.copy()
        daily["date"] = pd.to_datetime(daily["created_at"]).dt.date
        daily_counts = daily.groupby(["date", "risk_level"]).size().reset_index(name="count")
        fig_time = px.area(
            daily_counts, x="date", y="count", color="risk_level",
            color_discrete_map={"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"},
            labels={"date": "Date", "count": "Assessments"},
        )
        fig_time.update_layout(height=350, margin=dict(t=10, b=40))
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("No time data available")

st.markdown("---")

# ---------------------------------------------------------------------------
# Signal frequency
# ---------------------------------------------------------------------------

st.subheader("Top Fraud Signals")

all_signals = []
for _, row in filtered.iterrows():
    try:
        sigs = json.loads(row["signals"]) if isinstance(row["signals"], str) else row["signals"]
        for s in sigs:
            if isinstance(s, dict):
                all_signals.append(s.get("name", "unknown"))
    except (json.JSONDecodeError, TypeError):
        pass

if all_signals:
    sig_counts = pd.Series(all_signals).value_counts().head(10).reset_index()
    sig_counts.columns = ["signal", "count"]
    fig_signals = px.bar(
        sig_counts, x="count", y="signal", orientation="h",
        color="count", color_continuous_scale="OrRd",
        labels={"signal": "Signal", "count": "Frequency"},
    )
    fig_signals.update_layout(
        height=max(300, len(sig_counts) * 35),
        margin=dict(t=10, b=20, l=200),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    st.plotly_chart(fig_signals, use_container_width=True)
else:
    st.info("No signal data available in filtered results")

st.markdown("---")

# ---------------------------------------------------------------------------
# Analyst verdicts
# ---------------------------------------------------------------------------

verdict_col1, verdict_col2 = st.columns(2)

with verdict_col1:
    st.subheader("Analyst Verdicts")
    verdicts = filtered[filtered["analyst_verdict"].notna()]
    if len(verdicts) > 0:
        verdict_counts = verdicts["analyst_verdict"].value_counts().reset_index()
        verdict_counts.columns = ["verdict", "count"]
        fig_verdict = px.pie(
            verdict_counts, values="count", names="verdict",
            color="verdict",
            color_discrete_map={
                "confirmed_fraud": "#e74c3c",
                "false_positive": "#2ecc71",
                "escalated": "#f39c12",
            },
            hole=0.4,
        )
        fig_verdict.update_layout(height=300, margin=dict(t=10, b=20))
        st.plotly_chart(fig_verdict, use_container_width=True)
    else:
        st.info("No analyst verdicts recorded yet")

with verdict_col2:
    st.subheader("Batch vs On-Demand")
    batch_count = len(filtered[filtered["batch_run_id"].notna()])
    on_demand = len(filtered[filtered["batch_run_id"].isna()])
    source_df = pd.DataFrame({
        "source": ["Batch (nightly)", "On-demand (A2A)"],
        "count": [batch_count, on_demand],
    })
    fig_source = px.pie(
        source_df, values="count", names="source",
        color_discrete_sequence=["#3498db", "#9b59b6"],
        hole=0.4,
    )
    fig_source.update_layout(height=300, margin=dict(t=10, b=20))
    st.plotly_chart(fig_source, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Recent assessments table
# ---------------------------------------------------------------------------

st.subheader("Recent Assessments")

display_cols = [
    "customer_id", "score", "risk_level", "recommended_action",
    "confidence", "analyst_verdict", "created_at",
]
available_cols = [c for c in display_cols if c in filtered.columns]

table_df = filtered[available_cols].head(50).copy()
if "confidence" in table_df.columns:
    table_df["confidence"] = table_df["confidence"].apply(lambda x: f"{x:.0%}")
if "created_at" in table_df.columns:
    table_df["created_at"] = pd.to_datetime(table_df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")

st.dataframe(
    table_df,
    use_container_width=True,
    height=400,
    column_config={
        "score": st.column_config.ProgressColumn(
            "Score", min_value=0, max_value=100, format="%d",
        ),
        "risk_level": st.column_config.TextColumn("Risk Level"),
        "recommended_action": st.column_config.TextColumn("Action"),
    },
)

# ---------------------------------------------------------------------------
# On-demand assessment form
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Run On-Demand Assessment")

with st.form("assess_form"):
    form_col1, form_col2 = st.columns(2)
    with form_col1:
        customer_id = st.text_input("Customer ID", placeholder="C-1234")
    with form_col2:
        context = st.selectbox(
            "Context",
            ["refund_request", "return_request", "support_contact"],
        )
    submitted = st.form_submit_button("Run Assessment")

if submitted and customer_id:
    try:
        import asyncio

        from fraud_risk_agent.agent.graph import run_assessment

        with st.spinner(f"Assessing {customer_id}..."):
            result = asyncio.run(run_assessment(customer_id=customer_id, context=context))

        if isinstance(result, dict):
            st.success(
                f"Score: **{result.get('score', 'N/A')}** | "
                f"Risk: **{result.get('risk_level', 'N/A')}** | "
                f"Action: **{result.get('recommended_action', 'N/A')}**"
            )
            with st.expander("Full Assessment"):
                st.json(result)
        else:
            st.warning("Assessment returned no result")
    except Exception as e:
        st.error(
            f"Assessment failed: {e}\n\n"
            "Make sure the database is running and environment variables are set."
        )
elif submitted:
    st.warning("Enter a customer ID")
