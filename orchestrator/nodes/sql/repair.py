from __future__ import annotations

from infra.nim_gateway import NIMGateway


async def repair_candidate(
    candidate_sql: str,
    *,
    error_message: str,
    question: str,
    schema_snapshot: str,
    nim_gateway: NIMGateway,
) -> str | None:
    cleaned = candidate_sql.strip().strip("`")
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1]
    if cleaned != candidate_sql:
        return cleaned

    if not nim_gateway.enabled:
        return None

    prompt = [
        {
            "role": "user",
            "content": (
                "Fix the SQL query so it becomes a valid read-only query for the provided schema.\n"
                "Return only SQL.\n"
                f"Schema:\n{schema_snapshot}\n\n"
                f"Question: {question}\n"
                f"SQL: {candidate_sql}\n"
                f"Error: {error_message}"
            ),
        }
    ]
    try:
        repaired = await nim_gateway.chat(prompt, temperature=0.1, max_tokens=220)
        return repaired.strip().strip("`")
    except Exception:
        return None
