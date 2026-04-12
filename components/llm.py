from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx

from components.errors import BenchmarkAccessError, ModelAccessError
from infra.settings import settings


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    temperature: float = 0.1
    max_tokens: int = 1024


class ModelMode(str, Enum):
    SAFE = "safe"
    BENCHMARK = "benchmark"


class LLMClient(ABC):
    name: str

    @abstractmethod
    async def generate(self, messages: list[Message], config: GenerationConfig | None = None) -> str:
        raise NotImplementedError

    async def generate_json(self, messages: list[Message], config: GenerationConfig | None = None) -> dict[str, Any]:
        raw = await self.generate(messages, config=config)
        return parse_json_payload(raw)


def parse_json_payload(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Model returned JSON, but it was not an object.")
    return payload


class ChatCompletionsClient(LLMClient):
    def __init__(self, *, name: str, base_url: str, model: str, api_key: str | None = None) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or ""

    async def generate(self, messages: list[Message], config: GenerationConfig | None = None) -> str:
        cfg = config or GenerationConfig()
        payload_messages = [{"role": message.role, "content": message.content} for message in messages]
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "messages": payload_messages,
                    "temperature": cfg.temperature,
                    "max_tokens": cfg.max_tokens,
                },
            )
            response.raise_for_status()
            payload = response.json()
        return payload["choices"][0]["message"]["content"]


class ModelRouter:
    def __init__(self) -> None:
        self.safe_model_name = os.getenv("GEMMA_MODEL", settings.gemma_model)
        self.benchmark_model_name = os.getenv("NIM_MODEL", settings.nim_model)

    def get_client(self, mode: ModelMode) -> LLMClient:
        if mode == ModelMode.SAFE:
            gemma_base_url = os.getenv("GEMMA_BASE_URL", settings.gemma_base_url)
            gemma_model = os.getenv("GEMMA_MODEL", settings.gemma_model)
            gemma_api_key = os.getenv("GEMMA_API_KEY", settings.gemma_api_key)
            if not gemma_base_url or not gemma_model:
                raise ModelAccessError(
                    "Safe mode requires a Gemma-compatible chat endpoint. "
                    "Set GEMMA_BASE_URL and GEMMA_MODEL before running safe-mode validation."
                )
            return ChatCompletionsClient(
                name=f"Gemma:{gemma_model}",
                base_url=gemma_base_url,
                model=gemma_model,
                api_key=gemma_api_key,
            )
        nvidia_api_key = os.getenv("NVIDIA_API_KEY", settings.nvidia_api_key)
        nim_base_url = os.getenv("NIM_BASE_URL", settings.nim_base_url)
        nim_model = os.getenv("NIM_MODEL", settings.nim_model)
        if not nvidia_api_key:
            raise BenchmarkAccessError(
                "Benchmark mode requires NVIDIA_API_KEY. "
                "The system must not run NVIDIA NIM benchmarks without it."
            )
        return ChatCompletionsClient(
            name=f"NIM:{nim_model}",
            base_url=nim_base_url,
            model=nim_model,
            api_key=nvidia_api_key,
        )

    def describe(self, mode: ModelMode) -> dict[str, str]:
        if mode == ModelMode.SAFE:
            gemma_base_url = os.getenv("GEMMA_BASE_URL", settings.gemma_base_url)
            gemma_model = os.getenv("GEMMA_MODEL", settings.gemma_model)
            status = "configured" if gemma_base_url and gemma_model else "missing_config"
            return {"mode": mode.value, "provider": "gemma", "status": status, "model": gemma_model}
        nvidia_api_key = os.getenv("NVIDIA_API_KEY", settings.nvidia_api_key)
        nim_model = os.getenv("NIM_MODEL", settings.nim_model)
        status = "configured" if nvidia_api_key else "missing_api_key"
        return {"mode": mode.value, "provider": "nvidia_nim", "status": status, "model": nim_model}
