from __future__ import annotations

from typing import Any

from infra.nim_gateway import nim_gateway


def _structural_score(candidate: dict[str, Any], question: str) -> float:
    """Quick structural scoring — no dataset-specific logic, just SQL quality signals."""
    sql = candidate["sql"].lower()
    score = 0.3  # Base score for passing validation

    # Reward queries that use aggregation when the question implies it
    agg_words = {"total", "sum", "average", "avg", "count", "how many", "max", "min"}
    if any(word in question.lower() for word in agg_words):
        if any(fn in sql for fn in ("sum(", "avg(", "count(", "max(", "min(")):
            score += 0.2

    # Reward ORDER BY for ranking questions
    rank_words = {"top", "highest", "lowest", "best", "worst", "most", "least", "rank"}
    if any(word in question.lower() for word in rank_words):
        if "order by" in sql:
            score += 0.15

    # Reward WHERE clauses for filter questions
    filter_words = {"where", "filter", "only", "specific", "particular", "in the"}
    if any(word in question.lower() for word in filter_words):
        if "where" in sql:
            score += 0.15

    # Penalize overly simple queries
    if sql.count("select") == 1 and "where" not in sql and "group by" not in sql:
        score -= 0.05

    # Bonus for using multiple tables (more relevant = more complete answer)
    score += min(len(candidate.get("tables", [])), 3) * 0.05

    return score


async def select_best_candidate(
    valid_candidates: list[dict[str, Any]], question: str
) -> tuple[dict[str, Any], float]:
    """
    Agentar-Scale-SQL Stage 3: Tournament Selection.
    
    Uses LLM pairwise comparison when NIM is available,
    falls back to structural scoring otherwise.
    """
    if len(valid_candidates) == 1:
        return valid_candidates[0], 1.0

    # Try LLM-driven tournament selection
    if nim_gateway.enabled and len(valid_candidates) >= 2:
        try:
            winner = await _llm_tournament(valid_candidates, question)
            if winner is not None:
                scores = [_structural_score(c, question) for c in valid_candidates]
                sorted_scores = sorted(scores, reverse=True)
                margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0
                return winner, max(0.0, min(1.0, margin + 0.5))
        except Exception:
            pass

    # Fallback: structural scoring
    ranked = sorted(
        valid_candidates,
        key=lambda item: _structural_score(item, question),
        reverse=True,
    )
    best = ranked[0]
    margin = _structural_score(best, question) - _structural_score(ranked[1], question)
    return best, max(0.0, min(1.0, margin + 0.5))


async def _llm_tournament(
    candidates: list[dict[str, Any]], question: str
) -> dict[str, Any] | None:
    """LLM compares SQL candidates head-to-head and picks the best."""
    candidate_list = "\n\n".join(
        f"Candidate {i+1}:\n{c['sql']}" for i, c in enumerate(candidates[:4])
    )
    prompt = [
        {
            "role": "user",
            "content": (
                f"Given this question: {question}\n\n"
                f"Which SQL query best answers the question? Choose the number.\n\n"
                f"{candidate_list}\n\n"
                "Reply with ONLY the candidate number (1, 2, 3, or 4). No explanation."
            ),
        }
    ]
    response = await nim_gateway.chat(prompt, temperature=0.0, max_tokens=8)
    try:
        choice = int(response.strip().split()[0]) - 1
        if 0 <= choice < len(candidates):
            return candidates[choice]
    except (ValueError, IndexError):
        pass
    return None
