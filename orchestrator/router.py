from __future__ import annotations

import json
from infra.metrics_registry import MetricRegistry
from infra.nim_gateway import nim_gateway


async def route_question(
    *,
    question: str,
    mode: str,
    metric_registry: MetricRegistry,
    table_names: set[str],
) -> str:
    """LLM-driven routing using NVIDIA NIM. Zero hardcoding."""
    if mode in {"sql", "forecast", "anomaly"}:
        return mode

    if not nim_gateway.enabled:
        return "forecast"

    # Build dynamic context from whatever data is currently loaded
    metrics_context = "\n".join([
        f"- {m.key}: {m.label} (aliases: {', '.join(m.aliases)})"
        for m in metric_registry.metrics.values()
    ]) or "(no metrics registered yet)"

    tables_context = ", ".join(table_names) or "(no tables loaded)"

    system_prompt = f"""You are a routing engine for a predictive analytics application.
Classify the user's intent into exactly ONE of these categories:
- 'forecast': Predicting future values, trends, projections.
- 'scenario': "What-if" simulations (e.g., "What if sales increase by 10%?").
- 'anomaly': Finding unusual spikes, dips, or outliers in historical data.
- 'sql': General data queries, aggregations, lookups, or comparisons.
- 'unclear': Cannot determine intent from the question.

Available Metrics:
{metrics_context}

Available Tables:
{tables_context}

Respond with ONLY the category name. No explanation, no quotes."""

    try:
        response = await nim_gateway.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            max_tokens=16,
            temperature=0.0,
        )
        intent = response.strip().lower().strip("'\"")
        if intent in {"forecast", "scenario", "anomaly", "sql", "unclear"}:
            return intent
        return "forecast"
    except Exception:
        return "forecast"
