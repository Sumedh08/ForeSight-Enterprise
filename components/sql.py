from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from rank_bm25 import BM25Okapi

from components.connectors import DatabaseConnector, DatabaseSchema, QueryExecutionResult, serialize_schema
from components.errors import ComponentBlockedError
from components.llm import GenerationConfig, LLMClient, Message
from safety.sql_guard import guard_sql


@dataclass(frozen=True, slots=True)
class SQLCandidate:
    sql: str
    guarded_sql: str | None = None
    tables: list[str] = field(default_factory=list)
    status: str = "candidate"
    validation_error: str | None = None


@dataclass(frozen=True, slots=True)
class TextToSQLResult:
    status: str
    sql: str | None
    executed_sql: str | None
    selected_tables: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    candidates: list[SQLCandidate]
    warnings: list[str]


class SchemaRetriever:
    def __init__(self, schema: DatabaseSchema) -> None:
        self.schema = schema
        self.documents = [
            " ".join(
                [table.name]
                + [column.name for column in table.columns]
                + [str(sample) for column in table.columns for sample in column.sample_values]
            )
            for table in schema.tables
        ]
        self.tokens = [document.lower().split() for document in self.documents]
        self.bm25 = BM25Okapi(self.tokens) if self.tokens else None

    def top_tables(self, question: str, limit: int = 6) -> list[str]:
        if self.bm25 is None:
            return [table.name for table in self.schema.tables[:limit]]
        scores = self.bm25.get_scores(question.lower().split())
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        selected = [self.schema.tables[index].name for index, score in ranked[:limit] if score > 0]
        return selected or [table.name for table in self.schema.tables[:limit]]


class TextToSQLPipeline:
    def __init__(self, *, llm: LLMClient, row_cap: int = 500, candidate_count: int = 4) -> None:
        self.llm = llm
        self.row_cap = row_cap
        self.candidate_count = candidate_count

    async def run(
        self,
        *,
        question: str,
        connector: DatabaseConnector,
        external_knowledge: str | None = None,
        conversation_context: str | None = None,
    ) -> TextToSQLResult:
        schema = connector.introspect_schema()
        retriever = SchemaRetriever(schema)
        retrieved_tables = retriever.top_tables(question)
        schema_snapshot = serialize_schema(schema, table_names=retrieved_tables)
        candidates = await self._generate_candidates(
            question=question,
            schema_snapshot=schema_snapshot,
            dialect=schema.dialect,
            candidate_count=self.candidate_count,
            external_knowledge=external_knowledge,
            conversation_context=conversation_context,
        )
        validated: list[SQLCandidate] = []
        allowed_tables = {table.name for table in schema.tables}
        for item in candidates:
            try:
                guarded = guard_sql(item.sql, dialect=schema.dialect, allowed_tables=allowed_tables, row_cap=self.row_cap)
                connector.dry_run(guarded.sql)
                validated.append(SQLCandidate(sql=item.sql, guarded_sql=guarded.sql, tables=guarded.tables, status="valid"))
            except Exception as exc:
                validated.append(SQLCandidate(sql=item.sql, tables=item.tables, status="invalid", validation_error=str(exc)))

        valid_only = [candidate for candidate in validated if candidate.status == "valid"]
        if not valid_only:
            raise ComponentBlockedError("No SQL candidate survived AST validation and dry-run execution.")

        selected = await self._select_candidate(
            question=question,
            candidates=valid_only,
            schema_snapshot=schema_snapshot,
            external_knowledge=external_knowledge,
            conversation_context=conversation_context,
        )
        execution_sql = selected.guarded_sql or selected.sql
        execution = connector.execute(execution_sql)
        return TextToSQLResult(
            status="working",
            sql=selected.sql,
            executed_sql=execution_sql,
            selected_tables=selected.tables,
            rows=execution.rows,
            row_count=execution.row_count,
            candidates=validated,
            warnings=[],
        )

    async def _generate_candidates(
        self,
        *,
        question: str,
        schema_snapshot: str,
        dialect: str,
        candidate_count: int,
        external_knowledge: str | None,
        conversation_context: str | None,
    ) -> list[SQLCandidate]:
        knowledge_block = f"\nExternal knowledge:\n{external_knowledge}" if external_knowledge else ""
        conversation_block = f"\nConversation context:\n{conversation_context}" if conversation_context else ""
        prompt = (
            "You are a benchmark-grade text-to-SQL system.\n"
            f"Target SQL dialect: {dialect}.\n"
            "Return JSON with this shape: {\"candidates\": [{\"sql\": \"...\"}, ...]}.\n"
            "Generate diverse executable SQL candidates for the question.\n"
            "Do not include markdown fences or explanations.\n"
            f"Schema:\n{schema_snapshot}{knowledge_block}{conversation_block}\n\n"
            f"Question: {question}\n"
            f"Number of candidates: {candidate_count}"
        )
        payload = await self.llm.generate_json([Message(role="user", content=prompt)], GenerationConfig(temperature=0.2))
        raw_candidates = payload.get("candidates", [])
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ComponentBlockedError("The model did not return any SQL candidates.")
        cleaned: list[SQLCandidate] = []
        seen = set()
        for item in raw_candidates:
            if not isinstance(item, dict):
                continue
            sql = str(item.get("sql", "")).strip().strip("`")
            if sql and sql not in seen:
                cleaned.append(SQLCandidate(sql=sql))
                seen.add(sql)
        if not cleaned:
            raise ComponentBlockedError("The model output could not be parsed into SQL candidates.")
        return cleaned[:candidate_count]

    async def _select_candidate(
        self,
        *,
        question: str,
        candidates: list[SQLCandidate],
        schema_snapshot: str,
        external_knowledge: str | None,
        conversation_context: str | None,
    ) -> SQLCandidate:
        if len(candidates) == 1:
            return candidates[0]
        knowledge_block = f"\nExternal knowledge:\n{external_knowledge}" if external_knowledge else ""
        conversation_block = f"\nConversation context:\n{conversation_context}" if conversation_context else ""
        candidate_listing = json.dumps(
            [{"index": index, "sql": candidate.sql, "tables": candidate.tables} for index, candidate in enumerate(candidates)],
            indent=2,
        )
        prompt = (
            "You are a careful SQL selector.\n"
            "Return JSON with {\"selected_index\": <int>}.\n"
            "Choose the candidate that best answers the question while staying faithful to the schema.\n"
            f"Schema:\n{schema_snapshot}{knowledge_block}{conversation_block}\n\n"
            f"Question: {question}\nCandidates:\n{candidate_listing}"
        )
        payload = await self.llm.generate_json([Message(role="user", content=prompt)], GenerationConfig(temperature=0.0))
        index = payload.get("selected_index")
        if not isinstance(index, int) or index < 0 or index >= len(candidates):
            raise ComponentBlockedError("The model did not return a valid candidate selection.")
        return candidates[index]


def normalize_sql(sql: str, dialect: str) -> str:
    try:
        import sqlglot

        parsed = sqlglot.parse_one(sql, read=dialect)
        return parsed.sql(dialect=dialect)
    except Exception:
        return " ".join(sql.lower().split())


def result_signature(result: QueryExecutionResult) -> list[tuple]:
    tuples = [tuple(row.get(column) for column in result.columns) for row in result.rows]
    return sorted(tuples)
