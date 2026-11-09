"""Streamlit Dashboard for Cloud Autoscaling Simulation and Prediction.

Stage E: Presents results for:
- Workload generation (Stage A)
- Random Forest workload prediction (Stage B)
- Infrastructure simulation (Stage C)
- Reactive vs. Predictive autoscaling evaluation (Stage D3)
"""

from typing import Dict, List
import pandas as pd
import streamlit as st

from dashboard.charts import (
    create_autoscaling_cpu_chart,
    create_autoscaling_overload_chart,
    create_autoscaling_server_usage_chart,
    create_infrastructure_telemetry_chart,
    create_prediction_chart,
    create_prediction_error_chart,
    create_strategy_comparison_timeline,
    create_workload_chart,
)
from dashboard.data_loader import (
    SCENARIOS,
    STRATEGIES,
    compute_prediction_metrics,
    compute_workload_summary_metrics,
    load_autoscaling_decisions,
    load_autoscaling_metrics,
    load_autoscaling_summary,
    load_infrastructure_metrics,
    load_prediction_data,
    load_workload_data,
)

# Page configuration
st.set_page_config(
    page_title="Cloud Autoscaling & Workload Prediction Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_sidebar() -> str:
    """Render sidebar controls and return the selected scenario."""
    st.sidebar.title("⚡ Cloud Autoscaler")
    st.sidebar.caption("Dynamic Resource Allocation for Live Broadcasting")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Workload Scenario")
    selected_scenario = st.sidebar.selectbox(
        "Select Traffic Scenario:",
        options=SCENARIOS,
        index=0,
        help="Choose one of the 4 live broadcasting benchmark traffic scenarios.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **System Specifications:**
        - **Nominal Capacity:** 1,000 viewers/server
        - **Target CPU Utilization:** 70.0%
        - **Scale-Down CPU Threshold:** 30.0%
        - **Cooldown Duration:** 2 minutes
        - **Prediction Lead Horizon:** 5 minutes ($t + 5$)
        - **Server Allocation Range:** 2 – 30 servers
        """
    )
    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **Simulation Note:**\n"
        "CPU utilization above 100% represents an uncapped demand-to-capacity "
        "ratio in this simulation."
    )
    return selected_scenario


def render_workload_overview(scenario: str) -> None:
    """Section A: Workload Overview."""
    st.header("Section A: Workload Overview")
    st.markdown(
        f"Real-time viewer traffic progression for **{scenario}** across a 120-minute live broadcast simulation."
    )

    try:
        df_workload = load_workload_data(scenario=scenario)
    except FileNotFoundError as err:
        st.error(f"❌ {err}")
        return

    metrics = compute_workload_summary_metrics(df_workload)

    # Metric summary cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Peak Concurrent Viewers",
            value=f"{metrics['peak_viewers']:,}",
        )
    with col2:
        st.metric(
            label="Average Concurrent Viewers",
            value=f"{metrics['avg_viewers']:,.0f}",
        )
    with col3:
        st.metric(
            label="Total Cumulative Viewers",
            value=f"{metrics['total_viewers']:,}",
            help="Sum of concurrent viewer minutes observed across the 120-minute broadcast.",
        )
    with col4:
        st.metric(
            label="Baseline (Min) Viewers",
            value=f"{metrics['min_viewers']:,}",
        )

    # Plotly interactive time-series chart
    fig_workload = create_workload_chart(df_workload, scenario)
    st.plotly_chart(fig_workload, use_container_width=True)

    with st.expander("🔍 View Raw Workload Data Table"):
        st.dataframe(df_workload, use_container_width=True, height=220)


def render_workload_prediction(scenario: str) -> None:
    """Section B: Workload Prediction."""
    st.header("Section B: Workload Prediction Engine")
    st.markdown(
        f"Evaluation of the **Random Forest Regressor** against the **Persistence Baseline** for **{scenario}**.\n\n"
        "**Prediction Horizon:** **5 minutes ahead ($t + 5$)**."
    )

    try:
        df_pred = load_prediction_data(scenario=scenario)
        df_pred_all = load_prediction_data(scenario=None)
    except FileNotFoundError as err:
        st.error(f"❌ {err}")
        return

    # Scenario-specific metrics
    pred_metrics = compute_prediction_metrics(df_pred)
    overall_metrics = compute_prediction_metrics(df_pred_all)

    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Random Forest MAE",
            value=f"{pred_metrics['rf_mae']:.1f}",
            delta=(
                f"{pred_metrics['mae_reduction_pct']:+.1f}% vs Baseline"
                if pred_metrics["mae_reduction_pct"] != 0
                else None
            ),
            delta_color="normal" if pred_metrics["mae_reduction_pct"] >= 0 else "inverse",
            help="Mean Absolute Error on 5-minute lead forecast",
        )
    with col2:
        st.metric(
            label="Random Forest RMSE",
            value=f"{pred_metrics['rf_rmse']:.1f}",
            delta=(
                f"{pred_metrics['rmse_reduction_pct']:+.1f}% vs Baseline"
                if pred_metrics["rmse_reduction_pct"] != 0
                else None
            ),
            delta_color="normal" if pred_metrics["rmse_reduction_pct"] >= 0 else "inverse",
            help="Root Mean Squared Error on 5-minute lead forecast",
        )
    with col3:
        st.metric(
            label="Persistence Baseline MAE",
            value=f"{pred_metrics['base_mae']:.1f}",
            help="Baseline assumes V(t+5) = V(t)",
        )
    with col4:
        st.metric(
            label="Persistence Baseline RMSE",
            value=f"{pred_metrics['base_rmse']:.1f}",
        )

    # Prediction chart
    fig_pred = create_prediction_chart(df_pred, scenario)
    st.plotly_chart(fig_pred, use_container_width=True)

    # Prediction error breakdown
    fig_err = create_prediction_error_chart(df_pred, scenario)
    st.plotly_chart(fig_err, use_container_width=True)

    # Overall cross-scenario summary table
    st.subheader("Cross-Scenario Prediction Performance Benchmark")
    scenario_rows = []
    for sc in SCENARIOS:
        sc_df = df_pred_all[df_pred_all["scenario"] == sc]
        m = compute_prediction_metrics(sc_df)
        scenario_rows.append(
            {
                "Scenario": sc,
                "RF MAE": f"{m['rf_mae']:.2f}",
                "Baseline MAE": f"{m['base_mae']:.2f}",
                "MAE Reduction (%)": f"{m['mae_reduction_pct']:+.1f}%",
                "RF RMSE": f"{m['rf_rmse']:.2f}",
                "Baseline RMSE": f"{m['base_rmse']:.2f}",
                "RMSE Reduction (%)": f"{m['rmse_reduction_pct']:+.1f}%",
            }
        )
    scenario_rows.append(
        {
            "Scenario": "Overall (All Scenarios)",
            "RF MAE": f"{overall_metrics['rf_mae']:.2f}",
            "Baseline MAE": f"{overall_metrics['base_mae']:.2f}",
            "MAE Reduction (%)": f"{overall_metrics['mae_reduction_pct']:+.1f}%",
            "RF RMSE": f"{overall_metrics['rf_rmse']:.2f}",
            "Baseline RMSE": f"{overall_metrics['base_rmse']:.2f}",
            "RMSE Reduction (%)": f"{overall_metrics['rmse_reduction_pct']:+.1f}%",
        }
    )
    st.table(pd.DataFrame(scenario_rows))


def render_autoscaling_comparison(scenario: str) -> None:
    """Section C: Autoscaling Comparison."""
    st.header("Section C: Autoscaling Policy Comparison (Reactive vs Predictive)")
    st.markdown(
        "Head-to-head benchmark comparing **Reactive (threshold-driven)** and "
        "**Predictive (machine learning-driven)** autoscaling across infrastructure stability, "
        "responsiveness, and compute costs."
    )

    try:
        df_summary = load_autoscaling_summary()
        df_metrics = load_autoscaling_metrics(scenario=scenario)
    except FileNotFoundError as err:
        st.error(f"❌ {err}")
        return

    # Scenario head-to-head KPI cards
    st.subheader(f"Head-to-Head Performance — {scenario}")
    r_sum = df_summary[(df_summary["scenario"] == scenario) & (df_summary["strategy"] == "Reactive")].iloc[0]
    p_sum = df_summary[(df_summary["scenario"] == scenario) & (df_summary["strategy"] == "Predictive")].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Overloaded Intervals (min)",
            value=f"{int(p_sum['total_overload_intervals'])} m (Predictive)",
            delta=f"{int(p_sum['total_overload_intervals'] - r_sum['total_overload_intervals'])} vs Reactive ({int(r_sum['total_overload_intervals'])} m)",
            delta_color="inverse",
            help="Total simulation minutes where CPU > 100% or Memory > 100%",
        )
    with col2:
        st.metric(
            label="Average CPU Utilization",
            value=f"{p_sum['average_cpu_utilization']:.1f}% (Predictive)",
            delta=f"{p_sum['average_cpu_utilization'] - r_sum['average_cpu_utilization']:+.1f}% vs Reactive ({r_sum['average_cpu_utilization']:.1f}%)",
            delta_color="normal",
            help="Mean CPU utilization across all 120 intervals (Target: 70%)",
        )
    with col3:
        st.metric(
            label="Maximum CPU Utilization",
            value=f"{p_sum['maximum_cpu_utilization']:.1f}% (Predictive)",
            delta=f"{p_sum['maximum_cpu_utilization'] - r_sum['maximum_cpu_utilization']:+.1f}% vs Reactive ({r_sum['maximum_cpu_utilization']:.1f}%)",
            delta_color="inverse",
            help="Peak uncapped demand-to-capacity CPU ratio recorded",
        )
    with col4:
        st.metric(
            label="Total Server-Minutes (Cost)",
            value=f"{int(p_sum['total_server_minutes']):,} (Predictive)",
            delta=f"{int(p_sum['total_server_minutes'] - r_sum['total_server_minutes']):+,} vs Reactive ({int(r_sum['total_server_minutes']):,})",
            delta_color="inverse",
            help="Total active server minutes allocated",
        )

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric(
            label="Average Active Servers",
            value=f"{p_sum['average_active_servers']:.2f}",
            delta=f"{p_sum['average_active_servers'] - r_sum['average_active_servers']:+.2f} vs Reactive ({r_sum['average_active_servers']:.2f})",
            delta_color="off",
        )
    with col6:
        st.metric(
            label="Scale-Up Operations",
            value=f"{int(p_sum['scale_up_operations'])}",
            delta=f"{int(p_sum['scale_up_operations'] - r_sum['scale_up_operations'])} vs Reactive ({int(r_sum['scale_up_operations'])})",
            delta_color="inverse",
            help="Number of server pool expansion actions executed",
        )
    with col7:
        st.metric(
            label="Scale-Down Operations",
            value=f"{int(p_sum['scale_down_operations'])}",
            delta=f"{int(p_sum['scale_down_operations'] - r_sum['scale_down_operations'])} vs Reactive ({int(r_sum['scale_down_operations'])})",
            delta_color="off",
            help="Number of server pool decommissioning actions executed",
        )
    with col8:
        st.metric(
            label="Total Scaling Actions (Churn)",
            value=f"{int(p_sum['total_scaling_operations'])}",
            delta=f"{int(p_sum['total_scaling_operations'] - r_sum['total_scaling_operations'])} vs Reactive ({int(r_sum['total_scaling_operations'])})",
            delta_color="inverse",
            help="Total scaling operations (measure of operational churn)",
        )

    # Comparative Timeline
    st.subheader(f"Dynamic Scaling Timeline — {scenario}")
    fig_timeline = create_strategy_comparison_timeline(df_metrics, scenario)
    st.plotly_chart(fig_timeline, use_container_width=True)

    # Comparative charts across all 4 scenarios
    st.subheader("Global Benchmark Visualizations (All Scenarios)")
    fig_overload = create_autoscaling_overload_chart(df_summary)
    st.plotly_chart(fig_overload, use_container_width=True)

    col_chart_left, col_chart_right = st.columns(2)
    with col_chart_left:
        fig_cpu = create_autoscaling_cpu_chart(df_summary)
        st.plotly_chart(fig_cpu, use_container_width=True)
    with col_chart_right:
        fig_srv = create_autoscaling_server_usage_chart(df_summary)
        st.plotly_chart(fig_srv, use_container_width=True)

    # Summary table
    with st.expander("📊 Complete Stage D3 Summary Table"):
        st.dataframe(df_summary, use_container_width=True)


def render_infrastructure_monitoring(scenario: str) -> None:
    """Section D: Infrastructure Monitoring."""
    st.header("Section D: Infrastructure Monitoring & Telemetry")
    st.markdown(
        "Continuous operational monitoring of simulated cluster performance: "
        "CPU utilization, memory usage, active server count, and overload incidents."
    )

    # Mandatory simulation disclaimer
    st.warning(
        "⚠️ **Simulation Scale & Capacity Note:**\n\n"
        "**CPU utilization above 100% represents an uncapped demand-to-capacity ratio in this simulation.**\n"
        "Because the simulation operates without dropping incoming viewer connections, "
        "utilization reflects true required compute demand relative to active server capacity."
    )

    col_strat, col_view = st.columns([2, 2])
    with col_strat:
        selected_strategy = st.radio(
            "Select Autoscaling Strategy to Inspect:",
            options=STRATEGIES,
            index=0,
            horizontal=True,
        )

    try:
        df_metrics = load_autoscaling_metrics(scenario=scenario, strategy=selected_strategy)
    except FileNotFoundError as err:
        st.error(f"❌ {err}")
        return

    # Telemetry KPIs
    avg_cpu = df_metrics["cpu_utilization"].mean()
    max_cpu = df_metrics["cpu_utilization"].max()
    avg_mem = df_metrics["memory_utilization"].mean()
    overload_count = df_metrics["overload_status"].sum()
    curr_servers = df_metrics["active_servers"].iloc[-1]

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Average CPU", f"{avg_cpu:.1f}%")
    with col2:
        st.metric("Peak CPU", f"{max_cpu:.1f}%")
    with col3:
        st.metric("Average Memory", f"{avg_mem:.1f}%")
    with col4:
        st.metric("Overloaded Minutes", f"{overload_count} / 120")
    with col5:
        st.metric("Final Active Servers", f"{curr_servers}")

    # Interactive infrastructure telemetry chart
    fig_infra = create_infrastructure_telemetry_chart(df_metrics, scenario, selected_strategy)
    st.plotly_chart(fig_infra, use_container_width=True)

    # Operational decision events log
    with st.expander(f"📋 Scaling Decision Event Logs ({selected_strategy} — {scenario})"):
        try:
            df_decisions = load_autoscaling_decisions(scenario=scenario, strategy=selected_strategy)
            st.dataframe(df_decisions, use_container_width=True, height=280)
        except FileNotFoundError:
            st.info("Decision logs file not available.")


def main() -> None:
    """Main application layout."""
    selected_scenario = render_sidebar()

    # Header
    st.title("⚡ Dynamic Cloud Resource Allocation & Auto-Scaling Dashboard")
    st.markdown(
        "Interactive dashboard for simulated cloud infrastructure management, "
        "workload forecasting, and autoscaling policy benchmarking during live broadcast events."
    )

    # 4 Main Sections as Tabs
    tab_workload, tab_prediction, tab_comparison, tab_monitoring = st.tabs(
        [
            "📈 A. Workload Overview",
            "🤖 B. Workload Prediction",
            "⚖️ C. Autoscaling Comparison",
            "🖥️ D. Infrastructure Monitoring",
        ]
    )

    with tab_workload:
        render_workload_overview(selected_scenario)

    with tab_prediction:
        render_workload_prediction(selected_scenario)

    with tab_comparison:
        render_autoscaling_comparison(selected_scenario)

    with tab_monitoring:
        render_infrastructure_monitoring(selected_scenario)


if __name__ == "__main__":
    main()
