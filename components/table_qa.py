from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from components.errors import ComponentBlockedError
from components.llm import GenerationConfig, LLMClient, Message


def _render_rows(rows: list[dict[str, Any]], limit: int = 40) -> str:
    if not rows:
        return "No rows returned."
    columns = list(rows[0].keys())
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows[:limit]:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, divider, *body])


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    answer: str
    source_language: str
    answer_language: str
    evidence_row_indices: list[int]
    evidence_columns: list[str]
    grounded: bool


class TableQAPipeline:
    def __init__(self, *, llm: LLMClient) -> None:
        self.llm = llm

    async def answer_question(
        self,
        *,
        question: str,
        rows: list[dict[str, Any]],
        answer_language: str | None = None,
    ) -> GroundedAnswer:
        if not rows:
            return GroundedAnswer(
                answer="No rows were available to answer the question.",
                source_language="en",
                answer_language=answer_language or "en",
                evidence_row_indices=[],
                evidence_columns=[],
                grounded=True,
            )

        prompt = (
            "You are a grounded table question answering system.\n"
            "Return JSON with keys: source_language, answer_language, answer, evidence_row_indices, evidence_columns, grounded.\n"
            "You must answer using only the rows shown below. If the rows are insufficient, set answer to NOT_ANSWERABLE and grounded to false.\n"
            "Preserve the user's language when possible.\n"
            f"Requested answer language: {answer_language or 'same as question'}\n\n"
            f"Question: {question}\n\n"
            f"Rows:\n{_render_rows(rows)}"
        )
        payload = await self.llm.generate_json([Message(role="user", content=prompt)], GenerationConfig(temperature=0.0))
        answer = str(payload.get("answer", "")).strip()
        if not answer:
            raise ComponentBlockedError("The table QA model returned an empty answer.")
        evidence_row_indices = [int(item) for item in payload.get("evidence_row_indices", []) if isinstance(item, int)]
        evidence_columns = [str(item) for item in payload.get("evidence_columns", []) if isinstance(item, str)]
        grounded = bool(payload.get("grounded", False))
        if grounded and not self._validate_evidence(rows, evidence_row_indices, evidence_columns):
            grounded = False
        return GroundedAnswer(
            answer=answer,
            source_language=str(payload.get("source_language", "en")),
            answer_language=str(payload.get("answer_language", answer_language or payload.get("source_language", "en"))),
            evidence_row_indices=evidence_row_indices,
            evidence_columns=evidence_columns,
            grounded=grounded,
        )

    @staticmethod
    def _validate_evidence(rows: list[dict[str, Any]], indices: list[int], columns: list[str]) -> bool:
        if not indices or not columns:
            return False
        if any(index < 0 or index >= len(rows) for index in indices):
            return False
        existing = set(rows[0].keys())
        return all(column in existing for column in columns)
