from __future__ import annotations

import re
from collections import Counter

from infra.vector_store import RetrievalStore, tokenize


VALUE_RE = re.compile(r"'([^']+)'|\b(\d+(?:\.\d+)?)\b")


def extract_literals(question: str) -> list[str]:
    literals = []
    for match in VALUE_RE.finditer(question):
        literal = match.group(1) or match.group(2)
        if literal:
            literals.append(literal)
    return literals


def build_question_skeleton(question: str) -> str:
    skeleton = VALUE_RE.sub("<VALUE>", question)
    return re.sub(r"\s+", " ", skeleton).strip()


def build_sql_context(question: str, store: RetrievalStore) -> dict:
    examples = store.retrieve_examples(question)
    cells = store.retrieve_cell_values(question)
    token_counts = Counter(tokenize(question))
    keywords = [token for token, _ in token_counts.most_common(8)]
    retrieval_score = 0.0
    if examples:
        retrieval_score += 0.6
    if cells:
        retrieval_score += 0.4
    return {
        "keywords": keywords,
        "database_literals": extract_literals(question),
        "question_skeleton": build_question_skeleton(question),
        "retrieved_examples": examples,
        "retrieved_cells": cells,
        "retrieval_score": min(retrieval_score, 1.0),
    }
