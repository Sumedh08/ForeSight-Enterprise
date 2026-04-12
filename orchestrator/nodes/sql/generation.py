from __future__ import annotations

from typing import Any

from infra.nim_gateway import NIMGateway


async def generate_sql_candidates(
    question: str,
    *,
    schema_snapshot: str,
    context: dict[str, Any],
    nim_gateway: NIMGateway,
) -> list[str]:
    """
    Agentar-Scale-SQL inspired generation pipeline.
    
    Stage 1: Task Understanding — LLM analyzes the question against the schema
    Stage 2: Diverse Synthesis — multiple candidates at different temperatures
    
    No hardcoded table names, column names, or dataset-specific logic.
    """
    if not nim_gateway.enabled:
        return []

    # ── Stage 1: Task Understanding ──────────────────────────────────────
    # The LLM first analyzes which tables/columns are relevant before writing SQL
    understanding_prompt = [
        {
            "role": "system",
            "content": (
                "You are a SQL expert. Analyze the user's question against the database schema below.\n"
                "Identify:\n"
                "1. Which tables are relevant\n"
                "2. Which columns to SELECT\n"
                "3. Any WHERE filters needed\n"
                "4. Any aggregations (GROUP BY, SUM, AVG, COUNT)\n"
                "5. The sort order and any LIMIT\n\n"
                f"Schema:\n{schema_snapshot}\n\n"
                "Respond with a brief analysis, then write the SQL query.\n"
                "Return ONLY the final SQL query on the last line, with no markdown fences."
            ),
        },
        {"role": "user", "content": question},
    ]

    # ── Stage 2: Diverse Synthesis ───────────────────────────────────────
    # Generate candidates at different temperatures for diversity
    candidates: list[str] = []

    # Primary candidate — low temperature for accuracy (with task understanding)
    try:
        response = await nim_gateway.chat(
            understanding_prompt,
            temperature=0.1,
            max_tokens=400,
        )
        sql = _extract_sql(response)
        if sql:
            candidates.append(sql)
    except Exception:
        pass

    # Secondary candidate — slightly higher temperature for diversity
    direct_prompt = _direct_generation_prompt(question, schema_snapshot, context.get("retrieved_examples", []))
    try:
        response = await nim_gateway.chat(
            direct_prompt,
            temperature=0.4,
            max_tokens=300,
        )
        sql = _extract_sql(response)
        if sql:
            candidates.append(sql)
    except Exception:
        pass

    # Third candidate — ICL (In-Context Learning) with examples if available
    examples = context.get("retrieved_examples", [])
    if examples:
        icl_prompt = _icl_generation_prompt(question, schema_snapshot, examples)
        try:
            response = await nim_gateway.chat(
                icl_prompt,
                temperature=0.2,
                max_tokens=300,
            )
            sql = _extract_sql(response)
            if sql:
                candidates.append(sql)
        except Exception:
            pass

    # Deduplicate
    seen: set[str] = set()
    deduped: list[str] = []
    for c in candidates:
        normalized = c.strip()
        if normalized and normalized not in seen:
            deduped.append(normalized)
            seen.add(normalized)
    return deduped[:6]


def _extract_sql(response: str) -> str | None:
    """Extract the SQL query from an LLM response, handling markdown fences."""
    text = response.strip()
    # Remove markdown code fences if present
    if "```sql" in text:
        text = text.split("```sql", 1)[1]
        text = text.split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1]
        text = text.split("```", 1)[0]
    
    # Take the last SQL-like line if there's analysis before it
    lines = text.strip().split("\n")
    sql_lines = []
    collecting = False
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.upper().startswith(("SELECT", "WITH")):
            sql_lines.insert(0, stripped)
            collecting = True
        elif collecting and stripped:
            sql_lines.insert(0, stripped)
        elif collecting:
            break

    sql = " ".join(sql_lines).strip().rstrip(";").strip("`")
    return sql if sql and sql.upper().startswith(("SELECT", "WITH")) else None


def _direct_generation_prompt(question: str, schema_snapshot: str, examples: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Direct text-to-SQL prompt without task understanding step."""
    rendered_examples = "\n\n".join(
        f"Question: {example['question']}\nSQL: {example['sql']}" for example in examples[:3]
    )
    examples_section = f"\nExamples:\n{rendered_examples}\n" if rendered_examples else ""
    content = (
        "You are a precise text-to-SQL system.\n"
        "Return ONLY the SQL query with no markdown fences, no explanation.\n"
        "Use only tables and columns that exist in the schema.\n"
        f"Schema:\n{schema_snapshot}\n"
        f"{examples_section}\n"
        f"Question: {question}"
    )
    return [{"role": "user", "content": content}]


def _icl_generation_prompt(question: str, schema_snapshot: str, examples: list[dict[str, Any]]) -> list[dict[str, str]]:
    """In-Context Learning prompt that emphasizes following example patterns."""
    rendered = "\n\n".join(
        f"Q: {ex['question']}\nA: {ex['sql']}" for ex in examples[:3]
    )
    content = (
        "You are a text-to-SQL system. Follow the pattern shown in the examples below.\n"
        "Return ONLY the SQL query. No explanation, no markdown.\n\n"
        f"Database Schema:\n{schema_snapshot}\n\n"
        f"Examples:\n{rendered}\n\n"
        f"Q: {question}\nA:"
    )
    return [{"role": "user", "content": content}]
