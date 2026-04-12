from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from infra.settings import settings


class NIMGateway:
    def __init__(self) -> None:
        self.base_url = settings.nim_base_url.rstrip("/")
        self.model = settings.nim_model
        self.api_key = settings.nvidia_api_key

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    async def health(self) -> str:
        if not self.enabled:
            return "degraded"
        try:
            await self.chat(
                [{"role": "user", "content": "Reply with the single word ok."}],
                max_tokens=8,
                temperature=0,
            )
            return "up"
        except Exception:
            return "degraded"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        if not self.enabled:
            raise RuntimeError("NVIDIA_API_KEY is not configured")
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": model or self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            response.raise_for_status()
            payload = response.json()
            return payload["choices"][0]["message"]["content"]


nim_gateway = NIMGateway()
