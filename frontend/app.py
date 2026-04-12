from __future__ import annotations

import os
from typing import Any

import altair as alt
import httpx
import pandas as pd
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def api_get(path: str) -> dict[str, Any]:
    response = httpx.get(f"{API_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = httpx.post(f"{API_URL}{path}", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def render_sql_table(artifacts: dict[str, Any]) -> pd.DataFrame | None:
    frame = pd.DataFrame(artifacts.get("preview_rows", []))
    return frame if not frame.empty else None


def render_forecast_chart(artifacts: dict[str, Any]) -> alt.Chart | None:
    points = pd.DataFrame(artifacts.get("point_forecast", []))
    intervals = pd.DataFrame(artifacts.get("prediction_intervals", []))
    if points.empty and intervals.empty:
        return None

    layers = []
    if not intervals.empty and {"period", "low_80", "high_80"}.issubset(intervals.columns):
        intervals["period"] = pd.to_datetime(intervals["period"])
        layers.append(
            alt.Chart(intervals)
            .mark_area(opacity=0.2, color="#5E9")
            .encode(x="period:T", y="low_80:Q", y2="high_80:Q")
        )

    if not points.empty and {"period", "value"}.issubset(points.columns):
        points["period"] = pd.to_datetime(points["period"])
        layers.append(
            alt.Chart(points)
            .mark_line(point=True, color="#2B6CB0")
            .encode(
                x="period:T",
                y="value:Q",
                tooltip=[alt.Tooltip("period:T", title="Period"), alt.Tooltip("value:Q", title="Value")],
            )
        )

    if not layers:
        return None
    return alt.layer(*layers).properties(height=320, title="Forecast")


def render_scenario_chart(artifacts: dict[str, Any]) -> alt.Chart | None:
    baseline = pd.DataFrame(artifacts.get("baseline_forecast", []))
    scenario = pd.DataFrame(artifacts.get("scenario_forecast", []))
    if baseline.empty and scenario.empty:
        return None

    layers = []
    if not baseline.empty:
        baseline["period"] = pd.to_datetime(baseline["period"])
        layers.append(
            alt.Chart(baseline)
            .mark_line(point=True, color="#2B6CB0")
            .encode(x="period:T", y="value:Q", tooltip=["period:T", "value:Q"])
        )
    if not scenario.empty:
        scenario["period"] = pd.to_datetime(scenario["period"])
        layers.append(
            alt.Chart(scenario)
            .mark_line(point=True, color="#D97706", strokeDash=[6, 3])
            .encode(x="period:T", y="value:Q", tooltip=["period:T", "value:Q"])
        )
    return alt.layer(*layers).properties(height=320, title="Scenario Comparison")


def render_anomaly_chart(artifacts: dict[str, Any]) -> alt.Chart | None:
    baseline = pd.DataFrame(artifacts.get("baseline", []))
    anomalies = pd.DataFrame(artifacts.get("anomalies", []))
    if baseline.empty:
        return None

    baseline["period"] = pd.to_datetime(baseline["period"])
    line = alt.Chart(baseline).mark_line(color="#64748B").encode(x="period:T", y="value:Q")
    if anomalies.empty:
        return line.properties(height=320, title="Anomaly Scan")

    anomalies["period"] = pd.to_datetime(anomalies["period"])
    dots = alt.Chart(anomalies).mark_circle(size=110, color="#DC2626").encode(
        x="period:T",
        y="actual:Q",
        tooltip=["period:T", "actual:Q", "severity:N", "direction:N"],
    )
    return alt.layer(line, dots).properties(height=320, title="Anomaly Scan")


def render_component_status(components: dict[str, str]) -> None:
    for name, state in components.items():
        if state == "up":
            st.success(f"{name}: up")
        else:
            st.warning(f"{name}: {state}")


def build_config(connection_type: str) -> dict[str, Any]:
    if connection_type in {"duckdb", "sqlite"}:
        path = st.text_input("Database Path", value="data/demo.duckdb" if connection_type == "duckdb" else "data/local.sqlite")
        return {"path": path}

    mode = st.radio("Connection Input", options=["Fields", "DSN"], horizontal=True, key=f"{connection_type}_mode")
    if mode == "DSN":
        dsn = st.text_input("DSN", value="", placeholder="postgresql+psycopg2://user:pass@host:5432/db")
        return {"dsn": dsn}

    host = st.text_input("Host", value="postgres")
    port_default = 5432 if connection_type == "postgres" else 3306
    port = st.number_input("Port", value=port_default, step=1)
    database = st.text_input("Database", value="natwest_db")
    username = st.text_input("Username", value="admin")
    password = st.text_input("Password", value="", type="password")
    return {
        "host": host,
        "port": int(port),
        "database": database,
        "username": username,
        "password": password,
    }


st.set_page_config(page_title="AI Forecast Platform", layout="wide")
st.title("AI Predictive Forecast Platform")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Connection Wizard")
    try:
        health = api_get("/health")
        st.caption(
            f"Active: {health.get('active_connection') or 'none'} "
            f"({health.get('active_connection_type') or 'n/a'})"
        )
        render_component_status(health.get("components", {}))
    except Exception as exc:
        st.error(f"Health check failed: {exc}")

    try:
        connections = api_get("/api/connections")
    except Exception as exc:
        st.error(f"Could not load connections: {exc}")
        connections = {"profiles": [], "active_profile_id": None}

    profiles = connections.get("profiles", [])
    if profiles:
        profile_labels = [f"{item['name']} ({item['connection_type']})" for item in profiles]
        selected = st.selectbox("Saved Profiles", options=profile_labels)
        selected_index = profile_labels.index(selected)
        selected_profile = profiles[selected_index]
        if st.button("Activate Selected Profile"):
            try:
                api_post(f"/api/connections/{selected_profile['id']}/activate", {})
                st.success("Profile activated.")
                st.rerun()
            except Exception as exc:
                st.error(f"Activation failed: {exc}")

    st.subheader("New Connection")
    profile_name = st.text_input("Profile Name", value="my-connection")
    connection_type = st.selectbox("Connection Type", options=["postgres", "mysql", "sqlite", "duckdb"])
    config = build_config(connection_type)

    col_test, col_save = st.columns(2)
    with col_test:
        if st.button("Test Connection"):
            try:
                result = api_post("/api/connections/test", {"connection_type": connection_type, "config": config})
                if result.get("status") == "ok":
                    st.success(
                        f"Connected ({result.get('dialect')}) with {result.get('table_count', 0)} table(s)."
                    )
                else:
                    st.error(result.get("message", "Connection failed."))
            except Exception as exc:
                st.error(f"Test failed: {exc}")
    with col_save:
        if st.button("Save and Activate"):
            try:
                api_post(
                    "/api/connections",
                    {
                        "name": profile_name,
                        "connection_type": connection_type,
                        "config": config,
                        "activate": True,
                    },
                )
                st.success("Connection profile saved and activated.")
                st.rerun()
            except Exception as exc:
                st.error(f"Save failed: {exc}")

    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("CSV / Excel", type=["csv", "xlsx", "xls"])
    if st.button("Upload into Active DB") and uploaded_file is not None:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        try:
            response = httpx.post(f"{API_URL}/api/data/upload", files=files, timeout=120)
            response.raise_for_status()
            data = response.json()
            st.success(data.get("message", "Upload complete."))
            for item in data.get("warnings", []):
                st.warning(item)
        except Exception as exc:
            st.error(f"Upload failed: {exc}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("chart") is not None:
            st.altair_chart(message["chart"], use_container_width=True)
        if message.get("table") is not None:
            st.dataframe(message["table"], use_container_width=True)

prompt = st.chat_input("Ask a question about your connected data")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Running query...")
        try:
            payload = {"question": prompt, "mode": "auto"}
            response = api_post("/query", payload)
            answer = response.get("answer", "No answer returned.")
            confidence = response.get("confidence")
            if confidence is not None:
                answer = f"Confidence: {int(float(confidence) * 100)}%\n\n{answer}"
            placeholder.markdown(answer)

            chart = None
            table = None
            artifacts = response.get("artifacts")
            task_type = response.get("task_type")
            if artifacts:
                if task_type == "sql":
                    table = render_sql_table(artifacts)
                elif task_type == "forecast":
                    chart = render_forecast_chart(artifacts)
                    table = pd.DataFrame(artifacts.get("point_forecast", []))
                elif task_type == "scenario":
                    chart = render_scenario_chart(artifacts)
                    table = pd.DataFrame(artifacts.get("scenario_forecast", []))
                elif task_type == "anomaly":
                    chart = render_anomaly_chart(artifacts)
                    table = pd.DataFrame(artifacts.get("anomalies", []))

            for item in response.get("warnings", []):
                if isinstance(item, dict):
                    st.warning(f"{item.get('kind')}: {item.get('message')}")

            if chart is not None:
                st.altair_chart(chart, use_container_width=True)
            if table is not None and not table.empty:
                st.dataframe(table, use_container_width=True)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "chart": chart,
                    "table": table,
                }
            )
        except Exception as exc:
            placeholder.error(f"Query failed: {exc}")
