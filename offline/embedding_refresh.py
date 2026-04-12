from __future__ import annotations

from infra.vector_store import RetrievalStore


def refresh_retrieval_artifacts() -> dict:
    store = RetrievalStore()
    return {"example_count": len(store.examples), "status": store.health()}


if __name__ == "__main__":
    print(refresh_retrieval_artifacts())
