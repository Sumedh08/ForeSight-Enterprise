from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from components.errors import ComponentBlockedError
from components.llm import GenerationConfig, LLMClient, Message


def _infer_field_type(value: Any) -> str:
    if isinstance(value, (int, float)):
        return "quantitative"
    if hasattr(value, "isoformat"):
        return "temporal"
    text = str(value)
    if text.count("-") >= 2 or text.count("/") >= 2:
        return "temporal"
    return "nominal"


class ChartPlan(BaseModel):
    chart_type: Literal["bar", "line", "scatter"]
    x: str
    y: str
    color: str | None = None
    title: str = Field(default="Generated Chart")


@dataclass(frozen=True, slots=True)
class VisualizationResult:
    plan: ChartPlan
    vega_lite_spec: dict[str, Any]
    valid: bool


class VisualizationPipeline:
    def __init__(self, *, llm: LLMClient) -> None:
        self.llm = llm

    async def generate(self, *, question: str, rows: list[dict[str, Any]]) -> VisualizationResult:
        if not rows:
            raise ComponentBlockedError("Visualization generation requires query result rows.")
        columns = list(rows[0].keys())
        field_types = {column: _infer_field_type(rows[0][column]) for column in columns}
        prompt = (
            "You are a text-to-visualization planner.\n"
            "Return JSON with keys: chart_type, x, y, color, title.\n"
            "Allowed chart_type values are bar, line, scatter.\n"
            "Only reference columns that exist in the dataset.\n"
            f"Columns and inferred types: {field_types}\n"
            f"Question: {question}"
        )
        payload = await self.llm.generate_json([Message(role="user", content=prompt)], GenerationConfig(temperature=0.0))
        plan = ChartPlan.model_validate(payload)
        if plan.x not in columns or plan.y not in columns:
            raise ComponentBlockedError("Visualization plan referenced columns that do not exist.")
        if plan.color and plan.color not in columns:
            raise ComponentBlockedError("Visualization plan referenced a color field that does not exist.")
        return VisualizationResult(plan=plan, vega_lite_spec=self._build_spec(plan, field_types), valid=True)

    @staticmethod
    def _build_spec(plan: ChartPlan, field_types: dict[str, str]) -> dict[str, Any]:
        mark = "point" if plan.chart_type == "scatter" else plan.chart_type
        spec: dict[str, Any] = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "mark": mark,
            "encoding": {
                "x": {"field": plan.x, "type": field_types.get(plan.x, "nominal")},
                "y": {"field": plan.y, "type": field_types.get(plan.y, "quantitative")},
            },
            "title": plan.title,
        }
        if plan.color:
            spec["encoding"]["color"] = {"field": plan.color, "type": field_types.get(plan.color, "nominal")}
        return spec
