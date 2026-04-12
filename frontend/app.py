from __future__ import annotations

import os
import altair as alt
import httpx
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def _render_forecast(artifacts: dict) -> tuple:
    """Render forecast chart with historical data overlay, prediction bands, and baseline."""
    point_df = pd.DataFrame(artifacts.get("point_forecast", []))
    interval_df = pd.DataFrame(artifacts.get("prediction_intervals", []))
    baseline_df = pd.DataFrame(artifacts.get("baseline", []))
    anomaly_df = pd.DataFrame(artifacts.get("anomalies", []))

    for frame in (point_df, interval_df, baseline_df, anomaly_df):
        if not frame.empty and "period" in frame.columns:
            frame["period"] = pd.to_datetime(frame["period"])

    layers = []

    # Historical + baseline as gray dashed line
    if not baseline_df.empty and "value" in baseline_df.columns:
        history_line = alt.Chart(baseline_df).mark_line(color="#9E9E9E", strokeDash=[4, 3]).encode(
            x="period:T",
            y="value:Q",
            tooltip=[alt.Tooltip("period:T", title="Date"), alt.Tooltip("value:Q", title="Value")]
        )
        layers.append(history_line)

    # Prediction interval band
    if not interval_df.empty and "low_80" in interval_df.columns:
        band = alt.Chart(interval_df).mark_area(opacity=0.2, color="#2E8B57").encode(
            x="period:T",
            y="low_80:Q",
            y2="high_80:Q",
        )
        layers.append(band)

    # Point forecast line
    if not point_df.empty:
        point = alt.Chart(point_df).mark_line(color="#2196F3", point=True).encode(
            x="period:T",
            y="value:Q",
            tooltip=[alt.Tooltip("period:T", title="Date"), alt.Tooltip("value:Q", title="Forecast")]
        )
        layers.append(point)

    # Anomaly markers
    if not anomaly_df.empty and "actual" in anomaly_df.columns:
        anomaly_dots = alt.Chart(anomaly_df).mark_circle(size=120, color="red").encode(
            x="period:T",
            y="actual:Q",
            tooltip=[alt.Tooltip("period:T", title="Date"), alt.Tooltip("actual:Q", title="Value"), alt.Tooltip("severity:N")]
        )
        layers.append(anomaly_dots)

    chart = alt.layer(*layers).properties(height=350, title="Forecast") if layers else None
    return chart, point_df if not point_df.empty else None


def _render_scenario(artifacts: dict) -> tuple:
    """Render scenario comparison chart with baseline and scenario lines + bands."""
    base_df = pd.DataFrame(artifacts.get("baseline_forecast", []))
    scen_df = pd.DataFrame(artifacts.get("scenario_forecast", []))
    base_int = pd.DataFrame(artifacts.get("baseline_intervals", []))
    scen_int = pd.DataFrame(artifacts.get("scenario_intervals", []))

    for frame in (base_df, scen_df, base_int, scen_int):
        if not frame.empty and "period" in frame.columns:
            frame["period"] = pd.to_datetime(frame["period"])

    layers = []

    # Baseline band
    if not base_int.empty and "low_80" in base_int.columns:
        layers.append(alt.Chart(base_int).mark_area(opacity=0.15, color="#2196F3").encode(
            x="period:T", y="low_80:Q", y2="high_80:Q",
        ))

    # Scenario band
    if not scen_int.empty and "low_80" in scen_int.columns:
        layers.append(alt.Chart(scen_int).mark_area(opacity=0.15, color="#FF9800").encode(
            x="period:T", y="low_80:Q", y2="high_80:Q",
        ))

    # Baseline line
    if not base_df.empty:
        base_df["label"] = "Baseline"
        layers.append(alt.Chart(base_df).mark_line(color="#2196F3", point=True).encode(
            x="period:T", y="value:Q",
            tooltip=[alt.Tooltip("period:T"), alt.Tooltip("value:Q", title="Baseline")]
        ))

    # Scenario line
    if not scen_df.empty:
        scen_df["label"] = "Scenario"
        layers.append(alt.Chart(scen_df).mark_line(color="#FF9800", point=True, strokeDash=[6, 3]).encode(
            x="period:T", y="value:Q",
            tooltip=[alt.Tooltip("period:T"), alt.Tooltip("value:Q", title="Scenario")]
        ))

    chart = alt.layer(*layers).properties(height=350, title="Scenario Comparison") if layers else None
    combined = None
    if not base_df.empty and not scen_df.empty:
        combined = base_df[["period", "value"]].rename(columns={"value": "baseline"}).merge(
            scen_df[["period", "value"]].rename(columns={"value": "scenario"}), on="period", how="outer"
        )
    return chart, combined


def _render_anomaly(artifacts: dict) -> tuple:
    """Render anomaly chart with history and red markers for anomalous points."""
    history_df = pd.DataFrame(artifacts.get("baseline", []))
    anomaly_df = pd.DataFrame(artifacts.get("anomalies", []))

    for frame in (history_df, anomaly_df):
        if not frame.empty and "period" in frame.columns:
            frame["period"] = pd.to_datetime(frame["period"])

    layers = []
    if not history_df.empty and "value" in history_df.columns:
        layers.append(alt.Chart(history_df).mark_line(color="#607D8B", point=False).encode(
            x="period:T", y="value:Q",
            tooltip=[alt.Tooltip("period:T", title="Date"), alt.Tooltip("value:Q", title="Value")]
        ))

    if not anomaly_df.empty and "actual" in anomaly_df.columns:
        layers.append(alt.Chart(anomaly_df).mark_circle(size=150, color="red").encode(
            x="period:T", y="actual:Q",
            tooltip=[
                alt.Tooltip("period:T", title="Date"),
                alt.Tooltip("actual:Q", title="Value"),
                alt.Tooltip("severity:N", title="Severity"),
                alt.Tooltip("direction:N", title="Type"),
                alt.Tooltip("explanation:N", title="Detail"),
            ]
        ))

    chart = alt.layer(*layers).properties(height=350, title="Anomaly Detection") if layers else None
    return chart, anomaly_df if not anomaly_df.empty else None


# ── Streamlit App ──────────────────────────────────────────────────────

st.set_page_config(page_title="AI Forecast Assistant", layout="wide")
st.title("AI Predictive Forecasting")

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("1. Connect Data")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    if st.button("Upload & Connect") and uploaded_file is not None:
        with st.spinner("Uploading and analyzing schema..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            resp = httpx.post(f"{API_URL}/api/data/upload", files=files, timeout=60)
            if resp.status_code == 200:
                st.success(resp.json()["message"])
                st.session_state.data_loaded = True
            else:
                st.error(f"Error: {resp.text}")

# Always show chat — the backend will respond appropriately if no data is loaded
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("chart") is not None:
            st.altair_chart(message["chart"])
        if message.get("dataframe") is not None:
            st.dataframe(message["dataframe"])

if prompt := st.chat_input("Ask a question (e.g., 'Forecast bitcoin price for next 4 weeks')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")

        payload = {
            "question": prompt,
            "mode": "auto"
        }

        try:
            response = httpx.post(f"{API_URL}/query", json=payload, timeout=180)
            if response.status_code == 200:
                data = response.json()
                answer_text = data.get("answer", "")
                task_type = data.get("task_type", "")
                confidence = data.get("confidence", 0)
                artifacts = data.get("artifacts")

                # Add confidence badge
                if confidence > 0:
                    conf_pct = int(confidence * 100)
                    if conf_pct >= 70:
                        badge = f"🟢 Confidence: {conf_pct}%"
                    elif conf_pct >= 40:
                        badge = f"🟡 Confidence: {conf_pct}%"
                    else:
                        badge = f"🔴 Confidence: {conf_pct}%"
                    answer_text = f"{badge}\n\n{answer_text}"

                message_placeholder.markdown(answer_text)

                chart = None
                df = None

                if task_type == "forecast" and artifacts:
                    chart, df = _render_forecast(artifacts)
                elif task_type == "scenario" and artifacts:
                    chart, df = _render_scenario(artifacts)
                elif task_type == "anomaly" and artifacts:
                    chart, df = _render_anomaly(artifacts)
                elif task_type == "sql" and artifacts:
                    df = pd.DataFrame(artifacts.get("preview_rows", []))

                if chart is not None:
                    st.altair_chart(chart)
                if df is not None and not df.empty:
                    st.dataframe(df)

                # Backtest expander for forecast artifacts
                if task_type == "forecast" and artifacts and artifacts.get("backtest_metrics"):
                    bt = artifacts["backtest_metrics"]
                    with st.expander("📊 Backtest & Model Transparency"):
                        cols = st.columns(4)
                        cols[0].metric("MAE", f"{bt.get('mae', 'N/A'):.2f}" if bt.get("mae") else "N/A")
                        cols[1].metric("MAPE", f"{bt.get('mape', 0):.1%}" if bt.get("mape") else "N/A")
                        cols[2].metric("80% Coverage", f"{bt.get('coverage_80', 0):.0%}" if bt.get("coverage_80") else "N/A")
                        cols[3].metric("Beats Baseline", "✅ Yes" if bt.get("beats_baseline") else "⚠️ No")

                # CSV download
                if df is not None and not df.empty:
                    csv_data = df.to_csv(index=False)
                    st.download_button("📥 Download as CSV", csv_data, "forecast_results.csv", "text/csv")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "chart": chart,
                    "dataframe": df
                })
            else:
                error_text = f"Error: Server responded with {response.status_code}"
                message_placeholder.error(error_text)
        except Exception as e:
            message_placeholder.error(f"Failed to connect to API: {str(e)}")
