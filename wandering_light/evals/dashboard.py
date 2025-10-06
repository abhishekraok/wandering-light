import json
import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st


@st.cache_data
def load_evaluation_results() -> pd.DataFrame:
    """Load all evaluation results from the results directories."""
    results = []

    # Search only in root level results directory
    search_dirs = [
        "results",  # Root level results directory
    ]

    for results_dir in search_dirs:
        if not os.path.exists(results_dir):
            continue

        # Scan all directories (including nested ones for saved)
        for root, _dirs, files in os.walk(results_dir):
            if "summary.json" not in files:
                continue

            summary_file = os.path.join(root, "summary.json")

            try:
                with open(summary_file) as f:
                    summary = json.load(f)

                # Calculate relative path from results directory
                rel_path = os.path.relpath(root, "results")
                rel_path = "results" if rel_path == "." else f"results/{rel_path}"

                # Extract data for each solver in this run
                for solver_name, solver_results in summary.get("results", {}).items():
                    result_row = {
                        "timestamp": summary.get("timestamp", os.path.basename(root)),
                        "relative_path": rel_path,
                        "run_dir": root,
                        "solver": solver_name,
                        "eval_file": summary.get("eval_file", "Unknown"),
                        "num_samples": summary.get("num_samples"),
                        "budget": summary.get("budget", 1),
                        "success_rate": solver_results.get("success_rate", 0),
                        "success_count": solver_results.get("success_count", 0),
                        "total_samples": solver_results.get("total_samples", 0),
                        "avg_solution_length": solver_results.get(
                            "avg_solution_length", 0
                        ),
                        "failure_count": len(solver_results.get("failures", [])),
                        "command": summary.get("command", ""),
                        "model_name": extract_model_name(summary.get("command", "")),
                    }

                    # Parse timestamp for better sorting
                    try:
                        result_row["datetime"] = datetime.strptime(
                            result_row["timestamp"], "%Y%m%d_%H%M%S"
                        )
                    except:
                        result_row["datetime"] = datetime.now()

                    results.append(result_row)

            except Exception as e:
                st.warning(f"Error loading {summary_file}: {e}")
                continue

    return pd.DataFrame(results)


def format_eval_file(path: str) -> str:
    """Extract just the filename from eval file path."""
    return os.path.basename(path) if path else "Unknown"


def extract_model_name(command: str) -> str:
    """Extract model name from command string."""
    if not command:
        return ""

    # Look for --model-name or --model_name parameter
    import re

    pattern = r"--model[-_]name[=\s]+([^\s]+)"
    match = re.search(pattern, command)
    if match:
        return match.group(1)
    return ""


def main():
    st.set_page_config(page_title="Evaluation Dashboard", page_icon="📊", layout="wide")

    st.title("📊 Evaluation Results Dashboard")
    st.markdown("View and analyze metrics across multiple evaluation runs")

    # Load data
    with st.spinner("Loading evaluation results..."):
        df = load_evaluation_results()

    if df.empty:
        st.warning(
            "No evaluation results found in `results/`. Run some evaluations first!"
        )
        return

    st.success(f"Loaded {len(df)} evaluation runs")

    # Sidebar filters
    st.sidebar.header("🔍 Filters")

    # Solver filter
    solvers = ["All", *sorted(df["solver"].unique().tolist())]
    selected_solver = st.sidebar.selectbox("Solver", solvers)

    # Eval file filter
    eval_files = [
        "All",
        *sorted(df["eval_file"].apply(format_eval_file).unique().tolist()),
    ]
    selected_eval_file = st.sidebar.selectbox("Eval File", eval_files)

    # Budget filter
    budgets = ["All", *sorted(df["budget"].unique().tolist())]
    selected_budget = st.sidebar.selectbox("Budget", budgets)

    # Model name filter
    model_names = ["All", *sorted([name for name in df["model_name"].unique() if name])]
    selected_model_name = st.sidebar.selectbox("Model Name", model_names)

    # Date range filter
    if len(df) > 0:
        min_date = df["datetime"].min().date()
        max_date = df["datetime"].max().date()
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

    # Apply filters
    filtered_df = df.copy()

    if selected_solver != "All":
        filtered_df = filtered_df[filtered_df["solver"] == selected_solver]

    if selected_eval_file != "All":
        filtered_df = filtered_df[
            filtered_df["eval_file"].apply(format_eval_file) == selected_eval_file
        ]

    if selected_budget != "All":
        filtered_df = filtered_df[filtered_df["budget"] == selected_budget]

    if selected_model_name != "All":
        filtered_df = filtered_df[filtered_df["model_name"] == selected_model_name]

    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df["datetime"].dt.date >= start_date)
            & (filtered_df["datetime"].dt.date <= end_date)
        ]

    # Metrics overview
    if not filtered_df.empty:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Runs", len(filtered_df))

        with col2:
            avg_success_rate = filtered_df["success_rate"].mean()
            st.metric("Avg Success Rate", f"{avg_success_rate:.1%}")

        with col3:
            avg_length = filtered_df["avg_solution_length"].mean()
            st.metric("Avg Solution Length", f"{avg_length:.2f}")

        with col4:
            total_samples = filtered_df["total_samples"].sum()
            st.metric("Total Samples", f"{total_samples:,}")

    # Charts
    if not filtered_df.empty and len(filtered_df) > 1:
        st.header("📈 Visualizations")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # Success rate over time
            fig_time = px.line(
                filtered_df.sort_values("datetime"),
                x="datetime",
                y="success_rate",
                color="solver",
                title="Success Rate Over Time",
                labels={"success_rate": "Success Rate", "datetime": "Date"},
            )
            fig_time.update_layout(yaxis={"tickformat": ".1%"})
            st.plotly_chart(fig_time, use_container_width=True)

        with chart_col2:
            # Success rate by solver
            solver_stats = (
                filtered_df.groupby("solver")
                .agg({"success_rate": "mean", "total_samples": "sum"})
                .reset_index()
            )

            fig_solver = px.bar(
                solver_stats,
                x="solver",
                y="success_rate",
                title="Average Success Rate by Solver",
                labels={"success_rate": "Avg Success Rate", "solver": "Solver"},
            )
            fig_solver.update_layout(yaxis={"tickformat": ".1%"})
            st.plotly_chart(fig_solver, use_container_width=True)

        # Solution length distribution
        fig_length = px.histogram(
            filtered_df[filtered_df["avg_solution_length"] > 0],
            x="avg_solution_length",
            nbins=20,
            title="Solution Length Distribution",
            labels={
                "avg_solution_length": "Average Solution Length",
                "count": "Number of Runs",
            },
        )
        st.plotly_chart(fig_length, use_container_width=True)

    # Results table
    st.header("📋 Results Table")

    if not filtered_df.empty:
        # Prepare display dataframe
        display_df = filtered_df.copy()
        display_df["eval_file_short"] = display_df["eval_file"].apply(format_eval_file)
        display_df["success_rate_pct"] = (display_df["success_rate"] * 100).round(1)
        display_df["avg_solution_length"] = display_df["avg_solution_length"].round(2)

        # Select and order columns for display
        display_columns = [
            "relative_path",
            "solver",
            "eval_file_short",
            "model_name",
            "budget",
            "success_rate_pct",
            "success_count",
            "total_samples",
            "avg_solution_length",
            "failure_count",
        ]

        display_df = display_df[display_columns].sort_values(
            "relative_path", ascending=False
        )

        # Rename columns for better display
        column_names = {
            "relative_path": "Path",
            "solver": "Solver",
            "eval_file_short": "Eval File",
            "model_name": "Model Name",
            "budget": "Budget",
            "success_rate_pct": "Success Rate (%)",
            "success_count": "Successes",
            "total_samples": "Total Samples",
            "avg_solution_length": "Avg Length",
            "failure_count": "Failures",
        }

        display_df = display_df.rename(columns=column_names)

        # Add click-to-view functionality
        st.markdown("📌 **Click on a row to view detailed results**")

        selected_rows = st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        # Show selected run details
        if selected_rows["selection"]["rows"]:
            selected_idx = selected_rows["selection"]["rows"][0]
            selected_run = filtered_df.iloc[selected_idx]

            st.header(f"🔍 Run Details: {selected_run['relative_path']}")

            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.subheader("Run Information")
                st.write(f"**Solver:** {selected_run['solver']}")
                st.write(
                    f"**Eval File:** {format_eval_file(selected_run['eval_file'])}"
                )
                if selected_run["model_name"]:
                    st.write(f"**Model Name:** {selected_run['model_name']}")
                st.write(f"**Budget:** {selected_run['budget']}")
                st.write(f"**Total Samples:** {selected_run['total_samples']}")
                if selected_run["command"]:
                    st.write(f"**Command:** `{selected_run['command']}`")

            with detail_col2:
                st.subheader("Performance Metrics")
                st.write(f"**Success Rate:** {selected_run['success_rate']:.1%}")
                st.write(f"**Successes:** {selected_run['success_count']}")
                st.write(f"**Failures:** {selected_run['failure_count']}")
                st.write(
                    f"**Avg Solution Length:** {selected_run['avg_solution_length']:.2f}"
                )

            # Link to detailed viewer
            viewer_path = "eval_viewer.html"
            if os.path.exists(viewer_path):
                st.info(
                    f"💡 **Tip:** Use `{viewer_path}` to view sample-by-sample details for this run"
                )

    else:
        st.info("No results match the current filters.")

    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()
