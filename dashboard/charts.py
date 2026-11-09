"""Plotly interactive chart builders for the Streamlit dashboard."""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Clean corporate color palette
COLOR_PRIMARY = "#1f77b4"      # Blue (Actual)
COLOR_PREDICTIVE = "#2ca02c"   # Green (Predictive / RF)
COLOR_REACTIVE = "#d62728"     # Red / Crimson (Reactive)
COLOR_BASELINE = "#ff7f0e"     # Orange (Persistence Baseline)
COLOR_MEMORY = "#9467bd"       # Purple (Memory)
COLOR_OVERLOAD = "#e377c2"     # Pink / Magenta
COLOR_BACKGROUND = "#fafafa"


def create_workload_chart(df: pd.DataFrame, scenario_name: str) -> go.Figure:
    """Create an interactive time-series chart of audience viewer traffic.

    Args:
        df: DataFrame containing minute, viewers, and optionally event_active.
        scenario_name: Name of the scenario to display in title.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["minute"],
            y=df["viewers"],
            mode="lines+markers",
            name="Concurrent Viewers",
            line=dict(color=COLOR_PRIMARY, width=2.5),
            marker=dict(size=4),
            hovertemplate="Minute %{x}<br>Viewers: %{y:,}<extra></extra>",
        )
    )

    # Shaded event active window if event_active is present
    if "event_active" in df.columns and df["event_active"].any():
        active_minutes = df[df["event_active"]]["minute"]
        min_active = active_minutes.min()
        max_active = active_minutes.max()
        fig.add_vrect(
            x0=min_active,
            x1=max_active,
            fillcolor="rgba(255, 127, 14, 0.15)",
            layer="below",
            line_width=1,
            line_dash="dot",
            line_color="rgba(255, 127, 14, 0.6)",
            annotation_text="Live Event Active Window",
            annotation_position="top left",
        )

    fig.update_layout(
        title=dict(
            text=f"Audience Viewer Dynamics — {scenario_name}",
            font=dict(size=16),
        ),
        xaxis=dict(title="Simulation Time (Minutes)", gridcolor="#e5e5e5"),
        yaxis=dict(title="Concurrent Viewers", gridcolor="#e5e5e5"),
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_prediction_chart(df: pd.DataFrame, scenario_name: str) -> go.Figure:
    """Create interactive actual vs predicted vs baseline viewer chart.

    Args:
        df: DataFrame with minute, actual_viewers, predicted_viewers, baseline_viewers.
        scenario_name: Scenario name.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["minute"],
            y=df["actual_viewers"],
            mode="lines+markers",
            name="Actual Viewers (t+5)",
            line=dict(color=COLOR_PRIMARY, width=2.5),
            marker=dict(size=6),
            hovertemplate="Minute %{x}<br>Actual: %{y:,}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["minute"],
            y=df["predicted_viewers"],
            mode="lines+markers",
            name="Random Forest Forecast (t+5)",
            line=dict(color=COLOR_PREDICTIVE, width=2.5, dash="dash"),
            marker=dict(size=6, symbol="diamond"),
            hovertemplate="Minute %{x}<br>RF Forecast: %{y:,.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["minute"],
            y=df["baseline_viewers"],
            mode="lines+markers",
            name="Persistence Baseline (t)",
            line=dict(color=COLOR_BASELINE, width=2, dash="dot"),
            marker=dict(size=5, symbol="circle-open"),
            hovertemplate="Minute %{x}<br>Baseline: %{y:,}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Workload Prediction Comparison (5-Minute Lead Horizon: t + 5) — {scenario_name}",
            font=dict(size=16),
        ),
        xaxis=dict(title="Simulation Minute (Test Horizon: 93–114)", gridcolor="#e5e5e5"),
        yaxis=dict(title="Viewers", gridcolor="#e5e5e5"),
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_prediction_error_chart(df: pd.DataFrame, scenario_name: str) -> go.Figure:
    """Create absolute prediction error comparison chart.

    Args:
        df: DataFrame with error_rf, error_baseline, and minute.
        scenario_name: Scenario name.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["minute"],
            y=df["error_rf"],
            name="Random Forest Absolute Error",
            marker_color=COLOR_PREDICTIVE,
            opacity=0.8,
            hovertemplate="Minute %{x}<br>RF Error: %{y:,}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            x=df["minute"],
            y=df["error_baseline"],
            name="Baseline Absolute Error",
            marker_color=COLOR_BASELINE,
            opacity=0.6,
            hovertemplate="Minute %{x}<br>Baseline Error: %{y:,}<extra></extra>",
        )
    )

    fig.update_layout(
        barmode="group",
        title=dict(
            text=f"Absolute Forecast Error Comparison (|Actual - Forecast|) — {scenario_name}",
            font=dict(size=15),
        ),
        xaxis=dict(title="Simulation Minute", gridcolor="#e5e5e5"),
        yaxis=dict(title="Absolute Error (Viewers)", gridcolor="#e5e5e5"),
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_autoscaling_overload_chart(df_summary: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing overload intervals and overload percentage across scenarios."""
    fig = go.Figure()

    scenarios = df_summary["scenario"].unique().tolist()
    reactive_overloads = []
    predictive_overloads = []
    reactive_pct = []
    predictive_pct = []

    for sc in scenarios:
        r_row = df_summary[(df_summary["scenario"] == sc) & (df_summary["strategy"] == "Reactive")]
        p_row = df_summary[(df_summary["scenario"] == sc) & (df_summary["strategy"] == "Predictive")]

        r_val = r_row["total_overload_intervals"].values[0] if not r_row.empty else 0
        p_val = p_row["total_overload_intervals"].values[0] if not p_row.empty else 0
        r_p = r_row["overload_percentage"].values[0] if not r_row.empty else 0.0
        p_p = p_row["overload_percentage"].values[0] if not p_row.empty else 0.0

        reactive_overloads.append(r_val)
        predictive_overloads.append(p_val)
        reactive_pct.append(r_p)
        predictive_pct.append(p_p)

    fig.add_trace(
        go.Bar(
            x=scenarios,
            y=reactive_overloads,
            name="Reactive Strategy",
            marker_color=COLOR_REACTIVE,
            text=[f"{v}m ({p:.1f}%)" for v, p in zip(reactive_overloads, reactive_pct)],
            textposition="auto",
            hovertemplate="%{x}<br>Reactive: %{y} intervals<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            x=scenarios,
            y=predictive_overloads,
            name="Predictive Strategy",
            marker_color=COLOR_PREDICTIVE,
            text=[f"{v}m ({p:.1f}%)" for v, p in zip(predictive_overloads, predictive_pct)],
            textposition="auto",
            hovertemplate="%{x}<br>Predictive: %{y} intervals<extra></extra>",
        )
    )

    fig.update_layout(
        barmode="group",
        title=dict(text="Total Overloaded Intervals by Scenario", font=dict(size=16)),
        xaxis=dict(gridcolor="#e5e5e5"),
        yaxis=dict(title="Overload Intervals (Minutes out of 120)", gridcolor="#e5e5e5"),
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_autoscaling_cpu_chart(df_summary: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing Average and Maximum CPU utilization with threshold markers."""
    fig = go.Figure()

    scenarios = df_summary["scenario"].unique().tolist()
    r_avg = []
    p_avg = []
    r_max = []
    p_max = []

    for sc in scenarios:
        r_row = df_summary[(df_summary["scenario"] == sc) & (df_summary["strategy"] == "Reactive")]
        p_row = df_summary[(df_summary["scenario"] == sc) & (df_summary["strategy"] == "Predictive")]

        r_avg.append(r_row["average_cpu_utilization"].values[0] if not r_row.empty else 0.0)
        p_avg.append(p_row["average_cpu_utilization"].values[0] if not p_row.empty else 0.0)
        r_max.append(r_row["maximum_cpu_utilization"].values[0] if not r_row.empty else 0.0)
        p_max.append(p_row["maximum_cpu_utilization"].values[0] if not p_row.empty else 0.0)

    fig.add_trace(
        go.Bar(
            x=scenarios,
            y=r_avg,
            name="Reactive Avg CPU (%)",
            marker_color=COLOR_REACTIVE,
            text=[f"{v:.1f}%" for v in r_avg],
            textposition="auto",
            hovertemplate="%{x}<br>Reactive Avg CPU: %{y:.1f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            x=scenarios,
            y=p_avg,
            name="Predictive Avg CPU (%)",
            marker_color=COLOR_PREDICTIVE,
            text=[f"{v:.1f}%" for v in p_avg],
            textposition="auto",
            hovertemplate="%{x}<br>Predictive Avg CPU: %{y:.1f}%<extra></extra>",
        )
    )

    # Add threshold reference lines
    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="#d9534f",
        annotation_text="Scale-Up Target (70%)",
        annotation_position="top right",
    )
    fig.add_hline(
        y=30,
        line_dash="dot",
        line_color="#5bc0de",
        annotation_text="Scale-Down Target (30%)",
        annotation_position="bottom right",
    )
    fig.add_hline(
        y=100,
        line_dash="solid",
        line_color="#333333",
        annotation_text="Rated Capacity (100%)",
        annotation_position="top left",
    )

    fig.update_layout(
        barmode="group",
        title=dict(text="Average CPU Utilization Comparison vs Policy Targets", font=dict(size=16)),
        xaxis=dict(gridcolor="#e5e5e5"),
        yaxis=dict(title="CPU Utilization (%)", gridcolor="#e5e5e5"),
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_autoscaling_server_usage_chart(df_summary: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing Total Server-Minutes and Average Server Count."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Total Server-Minutes (Infrastructure Cost)", "Average Active Server Pool Size"),
    )

    scenarios = df_summary["scenario"].unique().tolist()
    r_srv_min = []
    p_srv_min = []
    r_avg_srv = []
    p_avg_srv = []

    for sc in scenarios:
        r_row = df_summary[(df_summary["scenario"] == sc) & (df_summary["strategy"] == "Reactive")]
        p_row = df_summary[(df_summary["scenario"] == sc) & (df_summary["strategy"] == "Predictive")]

        r_srv_min.append(r_row["total_server_minutes"].values[0] if not r_row.empty else 0)
        p_srv_min.append(p_row["total_server_minutes"].values[0] if not p_row.empty else 0)
        r_avg_srv.append(r_row["average_active_servers"].values[0] if not r_row.empty else 0.0)
        p_avg_srv.append(p_row["average_active_servers"].values[0] if not p_row.empty else 0.0)

    # Subplot 1: Server-Minutes
    fig.add_trace(
        go.Bar(
            x=scenarios,
            y=r_srv_min,
            name="Reactive Server-Mins",
            marker_color=COLOR_REACTIVE,
            text=[f"{v:,}" for v in r_srv_min],
            textposition="auto",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=scenarios,
            y=p_srv_min,
            name="Predictive Server-Mins",
            marker_color=COLOR_PREDICTIVE,
            text=[f"{v:,}" for v in p_srv_min],
            textposition="auto",
        ),
        row=1,
        col=1,
    )

    # Subplot 2: Average Active Servers
    fig.add_trace(
        go.Bar(
            x=scenarios,
            y=r_avg_srv,
            name="Reactive Avg Servers",
            marker_color=COLOR_REACTIVE,
            text=[f"{v:.1f}" for v in r_avg_srv],
            textposition="auto",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=scenarios,
            y=p_avg_srv,
            name="Predictive Avg Servers",
            marker_color=COLOR_PREDICTIVE,
            text=[f"{v:.1f}" for v in p_avg_srv],
            textposition="auto",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        barmode="group",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1),
    )
    return fig


def create_strategy_comparison_timeline(df_metrics: pd.DataFrame, scenario_name: str) -> go.Figure:
    """Create a 3-panel stacked timeline comparing Reactive and Predictive autoscaling dynamics.

    Args:
        df_metrics: DataFrame with both Reactive and Predictive metrics for one scenario.
        scenario_name: Name of the scenario.

    Returns:
        Plotly Figure.
    """
    df_reactive = df_metrics[df_metrics["strategy"] == "Reactive"].sort_values("minute")
    df_predictive = df_metrics[df_metrics["strategy"] == "Predictive"].sort_values("minute")

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "1. Incoming Audience Demand (Viewers)",
            "2. Active Server Allocation (N(t)) & Capacity",
            "3. Average CPU Utilization (%)",
        ),
    )

    # Panel 1: Viewer Traffic
    fig.add_trace(
        go.Scatter(
            x=df_reactive["minute"],
            y=df_reactive["actual_viewers"],
            mode="lines",
            name="Audience Demand",
            line=dict(color=COLOR_PRIMARY, width=2),
            hovertemplate="Minute %{x}: %{y:,} viewers<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Panel 2: Server Allocation
    fig.add_trace(
        go.Scatter(
            x=df_reactive["minute"],
            y=df_reactive["active_servers"],
            mode="lines+markers",
            name="Reactive Servers",
            line=dict(color=COLOR_REACTIVE, width=2),
            marker=dict(size=4),
            hovertemplate="Minute %{x}: %{y} servers (Reactive)<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_predictive["minute"],
            y=df_predictive["active_servers"],
            mode="lines+markers",
            name="Predictive Servers",
            line=dict(color=COLOR_PREDICTIVE, width=2, dash="dash"),
            marker=dict(size=4),
            hovertemplate="Minute %{x}: %{y} servers (Predictive)<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Add scaling action markers on server allocation panel
    r_ups = df_reactive[df_reactive["scaling_action"] == "scale_up"]
    if not r_ups.empty:
        fig.add_trace(
            go.Scatter(
                x=r_ups["minute"],
                y=r_ups["active_servers"],
                mode="markers",
                name="Reactive Scale-Up",
                marker=dict(symbol="triangle-up", size=9, color=COLOR_REACTIVE),
                showlegend=False,
                hovertemplate="Minute %{x}: Scale-Up to %{y} (Reactive)<extra></extra>",
            ),
            row=2,
            col=1,
        )

    p_ups = df_predictive[df_predictive["scaling_action"] == "scale_up"]
    if not p_ups.empty:
        fig.add_trace(
            go.Scatter(
                x=p_ups["minute"],
                y=p_ups["active_servers"],
                mode="markers",
                name="Predictive Scale-Up",
                marker=dict(symbol="triangle-up", size=9, color=COLOR_PREDICTIVE),
                showlegend=False,
                hovertemplate="Minute %{x}: Scale-Up to %{y} (Predictive)<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # Panel 3: CPU Utilization
    fig.add_trace(
        go.Scatter(
            x=df_reactive["minute"],
            y=df_reactive["cpu_utilization"],
            mode="lines",
            name="Reactive CPU (%)",
            line=dict(color=COLOR_REACTIVE, width=2),
            hovertemplate="Minute %{x}: %{y:.1f}% CPU (Reactive)<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_predictive["minute"],
            y=df_predictive["cpu_utilization"],
            mode="lines",
            name="Predictive CPU (%)",
            line=dict(color=COLOR_PREDICTIVE, width=2, dash="dash"),
            hovertemplate="Minute %{x}: %{y:.1f}% CPU (Predictive)<extra></extra>",
        ),
        row=3,
        col=1,
    )

    # Horizontal guide lines on Panel 3
    fig.add_hline(y=70, line_dash="dash", line_color="#d9534f", row=3, col=1)
    fig.add_hline(y=100, line_dash="solid", line_color="#333333", row=3, col=1)

    fig.update_layout(
        height=720,
        title=dict(
            text=f"Autoscaling Dynamics Timeline Comparison — {scenario_name}",
            font=dict(size=16),
        ),
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title="Simulation Minute", row=3, col=1, gridcolor="#e5e5e5")
    fig.update_yaxes(title="Viewers", row=1, col=1, gridcolor="#e5e5e5")
    fig.update_yaxes(title="Active Servers", row=2, col=1, gridcolor="#e5e5e5")
    fig.update_yaxes(title="CPU (%)", row=3, col=1, gridcolor="#e5e5e5")

    return fig


def create_infrastructure_telemetry_chart(
    df: pd.DataFrame, scenario_name: str, strategy_name: str = "Reactive"
) -> go.Figure:
    """Interactive dual-panel chart for Infrastructure Monitoring (Section D).

    Shows:
    1. CPU and Memory Utilization over time with 70% target, 100% capacity line, and overload markers.
    2. Active Servers and Scaling Actions over time.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            f"Resource Utilization & Overload Tracking ({strategy_name} Strategy)",
            f"Server Allocation & Operational Events ({strategy_name} Strategy)",
        ),
    )

    # Panel 1: CPU Utilization
    fig.add_trace(
        go.Scatter(
            x=df["minute"],
            y=df["cpu_utilization"],
            mode="lines",
            name="CPU Utilization (%)",
            line=dict(color=COLOR_PRIMARY, width=2),
            hovertemplate="Minute %{x}<br>CPU: %{y:.1f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Memory Utilization
    fig.add_trace(
        go.Scatter(
            x=df["minute"],
            y=df["memory_utilization"],
            mode="lines",
            name="Memory Utilization (%)",
            line=dict(color=COLOR_MEMORY, width=1.5, dash="dot"),
            hovertemplate="Minute %{x}<br>Memory: %{y:.1f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Overload status highlights
    overloaded_pts = df[df["overload_status"] == True]
    if not overloaded_pts.empty:
        fig.add_trace(
            go.Scatter(
                x=overloaded_pts["minute"],
                y=overloaded_pts["cpu_utilization"],
                mode="markers",
                name="Overload Event (CPU > 100%)",
                marker=dict(symbol="x", size=8, color="#d62728", line=dict(width=1.5)),
                hovertemplate="Minute %{x}: OVERLOADED (CPU: %{y:.1f}%)<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Target and capacity lines
    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="#d9534f",
        annotation_text="Target (70%)",
        annotation_position="top right",
        row=1,
        col=1,
    )
    fig.add_hline(
        y=100,
        line_dash="solid",
        line_color="#333333",
        annotation_text="Rated Capacity (100%)",
        annotation_position="top left",
        row=1,
        col=1,
    )

    # Panel 2: Active Servers
    fig.add_trace(
        go.Scatter(
            x=df["minute"],
            y=df["active_servers"],
            mode="lines+markers",
            name="Active Servers",
            line=dict(color="#2ca02c", width=2.5),
            marker=dict(size=4),
            hovertemplate="Minute %{x}<br>Active Servers: %{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Scale-up / Scale-down actions
    scale_ups = df[df["scaling_action"] == "scale_up"]
    if not scale_ups.empty:
        fig.add_trace(
            go.Scatter(
                x=scale_ups["minute"],
                y=scale_ups["active_servers"],
                mode="markers",
                name="Scale-Up Action",
                marker=dict(symbol="triangle-up", size=10, color="#2ca02c"),
                hovertemplate="Minute %{x}: Scale-Up to %{y} servers<extra></extra>",
            ),
            row=2,
            col=1,
        )

    scale_downs = df[df["scaling_action"] == "scale_down"]
    if not scale_downs.empty:
        fig.add_trace(
            go.Scatter(
                x=scale_downs["minute"],
                y=scale_downs["active_servers"],
                mode="markers",
                name="Scale-Down Action",
                marker=dict(symbol="triangle-down", size=10, color="#ff7f0e"),
                hovertemplate="Minute %{x}: Scale-Down to %{y} servers<extra></extra>",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        height=620,
        title=dict(
            text=f"Infrastructure Telemetry & Scaling Behavior — {scenario_name} ({strategy_name})",
            font=dict(size=16),
        ),
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title="Simulation Minute", row=2, col=1, gridcolor="#e5e5e5")
    fig.update_yaxes(title="Utilization (%)", row=1, col=1, gridcolor="#e5e5e5")
    fig.update_yaxes(title="Server Nodes", row=2, col=1, gridcolor="#e5e5e5")

    return fig
