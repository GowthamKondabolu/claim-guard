from __future__ import annotations

from html import escape
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

from claimguard.work_queue import (
    available_reason_codes,
    build_demo_work_queue,
    filter_work_queue,
    prepare_work_queue,
    summarize_work_queue,
)

st.set_page_config(
    page_title="ClaimGuard Investigator Work Queue",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 85% 5%, rgba(52, 211, 153, 0.09), transparent 25%),
            radial-gradient(circle at 5% 35%, rgba(56, 189, 248, 0.07), transparent 22%),
            #07111f;
    }
    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    .hero {
        padding: 1.6rem 1.8rem;
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            rgba(17, 28, 46, 0.96),
            rgba(9, 20, 35, 0.92)
        );
        box-shadow: 0 18px 55px rgba(0, 0, 0, 0.24);
        margin-bottom: 1.3rem;
    }
    .eyebrow {
        color: #34d399;
        font-size: 0.76rem;
        font-weight: 750;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }
    .hero h1 {
        color: #f8fafc;
        font-size: clamp(2rem, 4vw, 3.2rem);
        line-height: 1.05;
        margin: 0;
    }
    .hero p {
        color: #aebed1;
        max-width: 850px;
        font-size: 1.02rem;
        margin: 0.8rem 0 0;
    }
    div[data-testid="stMetric"] {
        background: rgba(17, 28, 46, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 14px;
        padding: 0.9rem 1rem;
    }
    div[data-testid="stMetric"] label {
        color: #91a4ba;
    }
    div[data-testid="stMetricValue"] {
        color: #f8fafc;
    }
    .reason-badge {
        display: inline-block;
        padding: 0.34rem 0.58rem;
        margin: 0.18rem 0.28rem 0.18rem 0;
        border-radius: 999px;
        border: 1px solid rgba(52, 211, 153, 0.35);
        background: rgba(52, 211, 153, 0.10);
        color: #8ff3ce;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .detail-card {
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: rgba(17, 28, 46, 0.68);
        margin-top: 0.75rem;
    }
    .detail-card h3 {
        margin-top: 0;
        color: #f8fafc;
    }
    .disclaimer {
        color: #8193a8;
        font-size: 0.82rem;
        border-top: 1px solid rgba(148, 163, 184, 0.14);
        margin-top: 2rem;
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] {
        background: #0a1524;
        border-right: 1px solid rgba(148, 163, 184, 0.13);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_demo_queue(rows: int) -> pd.DataFrame:
    return build_demo_work_queue(rows=rows, seed=42)


@st.cache_data(show_spinner=False)
def load_uploaded_queue(payload: bytes) -> pd.DataFrame:
    frame = pd.read_csv(BytesIO(payload), low_memory=False)
    return prepare_work_queue(frame)


st.sidebar.markdown("## 🛡️ ClaimGuard")
st.sidebar.caption("Investigator controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload a scored ClaimGuard CSV",
    type=["csv"],
    help="The app uses generated, privacy-safe demo claims when no file is uploaded.",
)
demo_rows = st.sidebar.slider(
    "Generated demo claims",
    min_value=500,
    max_value=5_000,
    value=2_000,
    step=500,
    disabled=uploaded_file is not None,
)

try:
    if uploaded_file is not None:
        queue = load_uploaded_queue(uploaded_file.getvalue())
        data_source = uploaded_file.name
    else:
        with st.spinner("Generating and scoring privacy-safe demo claims..."):
            queue = load_demo_queue(demo_rows)
        data_source = f"Generated synthetic demo · {demo_rows:,} claims"
except (TypeError, ValueError) as exc:
    st.error(f"Unable to prepare the work queue: {exc}")
    st.stop()

st.sidebar.divider()
st.sidebar.markdown("### Queue filters")

claim_type_options = sorted(queue["claim_type"].dropna().unique().tolist())
selected_claim_types = st.sidebar.multiselect(
    "Claim types",
    options=claim_type_options,
    default=claim_type_options,
)
flagged_only = st.sidebar.toggle("Flagged claims only", value=True)
minimum_score = st.sidebar.slider(
    "Minimum ensemble score",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.01,
)
reason_options = available_reason_codes(queue)
selected_reasons = st.sidebar.multiselect(
    "Triggered reasons",
    options=reason_options,
    format_func=lambda code: code.replace("_", " ").title(),
)

filtered = filter_work_queue(
    queue,
    claim_types=selected_claim_types,
    minimum_score=minimum_score,
    flagged_only=flagged_only,
    reason_codes=selected_reasons,
)

st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">Payment integrity · Human review</div>
        <h1>Investigator Work Queue</h1>
        <p>
            Prioritize unusual healthcare claims using calibrated model scores,
            transparent business rules, and a model-dominant ensemble.
            Every signal supports review—it does not determine fraud.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.caption(f"Data source: {data_source}")

summary = summarize_work_queue(filtered)
metric_columns = st.columns(5)
metric_columns[0].metric("Visible claims", f"{summary['claims']:,}")
metric_columns[1].metric("Flagged", f"{summary['flagged_claims']:,}")
metric_columns[2].metric(
    "Paid amount in review",
    f"${summary['review_paid_amount']:,.0f}",
)
metric_columns[3].metric(
    "Average ensemble score",
    f"{summary['average_ensemble_score']:.3f}",
)

if "flagged_injected_scenarios" in summary:
    metric_columns[4].metric(
        "Demo scenarios visible",
        f"{summary['flagged_injected_scenarios']:,}",
    )
else:
    metric_columns[4].metric("Model version", "0.2.0")

if filtered.empty:
    st.warning("No claims match the selected filters. Adjust the sidebar controls.")
    st.stop()

queue_tab, analysis_tab, about_tab = st.tabs(
    ["Review queue", "Score analysis", "About this demo"]
)

with queue_tab:
    st.subheader("Prioritized claims")
    st.caption("Select a row to inspect its score composition and review signals.")

    display_columns = [
        "claim_id",
        "claim_type",
        "paid_amount",
        "model_score_percentile",
        "rule_score",
        "ensemble_score",
        "is_flagged",
        "reason_codes_display",
    ]
    display_queue = filtered[display_columns].rename(
        columns={
            "claim_id": "Claim ID",
            "claim_type": "Claim type",
            "paid_amount": "Paid amount",
            "model_score_percentile": "Model percentile",
            "rule_score": "Rule score",
            "ensemble_score": "Ensemble score",
            "is_flagged": "Flagged",
            "reason_codes_display": "Review signals",
        }
    )

    selection = st.dataframe(
        display_queue,
        hide_index=True,
        width="stretch",
        height=430,
        key="investigator_queue",
        on_select="rerun",
        selection_mode="single-row-required",
        column_config={
            "Paid amount": st.column_config.NumberColumn(format="$%.2f"),
            "Model percentile": st.column_config.ProgressColumn(
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
            ),
            "Rule score": st.column_config.ProgressColumn(
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
            ),
            "Ensemble score": st.column_config.ProgressColumn(
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
            ),
            "Flagged": st.column_config.CheckboxColumn(),
        },
    )

    selected_positions = selection.selection.rows
    if selected_positions:
        selected = filtered.iloc[selected_positions[0]]
        st.markdown(
            f"""
            <div class="detail-card">
                <div class="eyebrow">Selected claim</div>
                <h3>{escape(str(selected['claim_id']))}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        detail_metrics = st.columns(4)
        detail_metrics[0].metric(
            "Paid amount",
            f"${float(selected['paid_amount']):,.2f}",
        )
        detail_metrics[1].metric(
            "Model percentile",
            f"{float(selected['model_score_percentile']):.3f}",
        )
        detail_metrics[2].metric(
            "Rule score",
            f"{float(selected['rule_score']):.3f}",
        )
        detail_metrics[3].metric(
            "Ensemble score",
            f"{float(selected['ensemble_score']):.3f}",
        )

        reasons = selected["rule_reason_codes"] or ["MULTIVARIATE_OUTLIER"]
        badges = "".join(
            (
                '<span class="reason-badge">'
                f"{escape(reason.replace('_', ' ').title())}"
                "</span>"
            )
            for reason in reasons
        )
        st.markdown("#### Review signals")
        st.markdown(badges, unsafe_allow_html=True)

        detail_fields = [
            ("Claim type", "claim_type"),
            ("Service date", "service_date"),
            ("Provider", "provider_id"),
            ("Beneficiary", "beneficiary_id"),
            ("Units", "units"),
            ("Provider claims · 30d", "provider_claim_count_30d"),
            ("Beneficiary claims · 30d", "beneficiary_claim_count_30d"),
            ("Provider paid z-score", "provider_paid_zscore"),
            ("Duplicate indicator", "duplicate_indicator"),
            ("Weekend service", "weekend_service"),
        ]
        available_details = [
            {"Field": label, "Value": str(selected[column])}
            for label, column in detail_fields
            if column in selected.index
        ]
        if available_details:
            st.dataframe(
                pd.DataFrame(available_details),
                hide_index=True,
                width="stretch",
            )

    st.download_button(
        "Download filtered queue",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="claimguard_investigator_queue.csv",
        mime="text/csv",
        width="stretch",
    )

with analysis_tab:
    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.subheader("Model and rule agreement")
        st.caption(
            "Upper-right claims receive strong signals from both scoring paths."
        )
        st.scatter_chart(
            filtered,
            x="model_score_percentile",
            y="rule_score",
            color="claim_type",
            size="paid_amount",
            height=390,
        )

    with chart_right:
        st.subheader("Ensemble-score distribution")
        score_bands = pd.cut(
            filtered["ensemble_score"],
            bins=np.linspace(0.0, 1.0, 11),
            include_lowest=True,
        )
        distribution = (
            score_bands.value_counts(sort=False)
            .rename_axis("Score band")
            .reset_index(name="Claims")
        )
        distribution["Score band"] = distribution["Score band"].astype(str)
        st.bar_chart(
            distribution,
            x="Score band",
            y="Claims",
            height=390,
        )

    st.subheader("Claim-type mix")
    claim_mix = (
        filtered.groupby("claim_type")
        .size()
        .sort_values(ascending=False)
        .rename_axis("Claim type")
        .reset_index(name="Claims")
    )
    st.bar_chart(
        claim_mix,
        x="Claim type",
        y="Claims",
        height=320,
    )

with about_tab:
    st.subheader("How ClaimGuard prioritizes review")
    st.markdown(
        """
        1. **Validate and enrich claims** using the Pandas or parity-tested
           PySpark feature pipeline.
        2. **Score multivariate anomalies** with Isolation Forest.
        3. **Apply transparent rules** for cost, units, duplicates, and
           provider or beneficiary utilization.
        4. **Calibrate and combine signals** using a 90/10 model-rule ensemble.
        5. **Apply review capacity** and present reason codes to an investigator.
        """
    )
    st.info(
        "Synthetic labels are used only for offline evaluation. "
        "They are never model-training features or fraud determinations."
    )
    st.markdown(
        """
        **Designed for:** portfolio demonstration, model review, and
        payment-integrity workflow exploration.

        **Not designed for:** payment denial, clinical decisions, provider
        sanctions, or use with protected health information without approved
        security and governance controls.
        """
    )

st.markdown(
    """
    <div class="disclaimer">
        ClaimGuard is an educational decision-support project. A flagged claim
        is not evidence of fraud, waste, or abuse. Every claim requires
        qualified human review.
    </div>
    """,
    unsafe_allow_html=True,
)