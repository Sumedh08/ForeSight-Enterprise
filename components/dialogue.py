from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from components.connectors import DatabaseConnector
from components.errors import ComponentBlockedError
from components.llm import GenerationConfig, LLMClient, Message
from components.sql import TextToSQLPipeline, TextToSQLResult
from components.table_qa import GroundedAnswer, TableQAPipeline


@dataclass(slots=True)
class ConversationTurn:
    user_message: str
    assistant_message: str
    action: str
    sql: str | None = None


@dataclass(slots=True)
class ConversationState:
    session_id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    pending_question: str | None = None
    pending_clarification: str | None = None


@dataclass(frozen=True, slots=True)
class DialogueResult:
    action: str
    assistant_message: str
    sql_result: TextToSQLResult | None
    grounded_answer: GroundedAnswer | None
    state: ConversationState


class InMemoryConversationStore:
    def __init__(self) -> None:
        self._states: dict[str, ConversationState] = {}

    def get(self, session_id: str) -> ConversationState:
        return self._states.setdefault(session_id, ConversationState(session_id=session_id))

    def save(self, state: ConversationState) -> None:
        self._states[state.session_id] = state


class DialogueSystem:
    def __init__(
        self,
        *,
        llm: LLMClient,
        sql_pipeline: TextToSQLPipeline,
        table_qa: TableQAPipeline,
        store: InMemoryConversationStore | None = None,
    ) -> None:
        self.llm = llm
        self.sql_pipeline = sql_pipeline
        self.table_qa = table_qa
        self.store = store or InMemoryConversationStore()

    async def reply(self, *, session_id: str, user_message: str, connector: DatabaseConnector) -> DialogueResult:
        state = self.store.get(session_id)
        decision = await self._decide_action(state=state, user_message=user_message)
        action = str(decision.get("action", "clarify"))
        if action == "clarify":
            clarification = str(decision.get("clarification_question", "")).strip()
            if not clarification:
                raise ComponentBlockedError("Dialogue policy requested clarification but did not provide a question.")
            state.pending_question = str(decision.get("resolved_question", user_message))
            state.pending_clarification = clarification
            state.turns.append(ConversationTurn(user_message=user_message, assistant_message=clarification, action="clarify"))
            self.store.save(state)
            return DialogueResult(action="clarify", assistant_message=clarification, sql_result=None, grounded_answer=None, state=state)

        resolved_question = str(decision.get("resolved_question", user_message)).strip() or user_message
        if state.pending_question and state.pending_clarification:
            resolved_question = str(decision.get("resolved_question", f"{state.pending_question}\nClarification: {user_message}")).strip()

        sql_result = await self.sql_pipeline.run(
            question=resolved_question,
            connector=connector,
            conversation_context=self._render_history(state),
        )
        grounded = await self.table_qa.answer_question(question=resolved_question, rows=sql_result.rows)
        assistant_message = grounded.answer
        state.pending_question = None
        state.pending_clarification = None
        state.turns.append(
            ConversationTurn(
                user_message=user_message,
                assistant_message=assistant_message,
                action="answer",
                sql=sql_result.sql,
            )
        )
        self.store.save(state)
        return DialogueResult(
            action="answer",
            assistant_message=assistant_message,
            sql_result=sql_result,
            grounded_answer=grounded,
            state=state,
        )

    async def _decide_action(self, *, state: ConversationState, user_message: str) -> dict[str, Any]:
        prompt = (
            "You are a dialogue manager for conversational database QA.\n"
            "Return JSON with keys: action, clarification_question, resolved_question.\n"
            "action must be either 'clarify' or 'query'.\n"
            "Ask for clarification only when the user request is missing a critical detail.\n"
            f"Conversation history:\n{self._render_history(state)}\n\n"
            f"Pending question: {state.pending_question or 'none'}\n"
            f"Pending clarification: {state.pending_clarification or 'none'}\n"
            f"User message: {user_message}"
        )
        return await self.llm.generate_json([Message(role="user", content=prompt)], GenerationConfig(temperature=0.0))

    @staticmethod
    def _render_history(state: ConversationState) -> str:
        if not state.turns:
            return "No prior turns."
        rendered = []
        for turn in state.turns[-6:]:
            rendered.append(f"User: {turn.user_message}")
            rendered.append(f"Assistant: {turn.assistant_message}")
            if turn.sql:
                rendered.append(f"SQL: {turn.sql}")
        return "\n".join(rendered)
