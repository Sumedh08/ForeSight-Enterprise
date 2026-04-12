from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCAN_PATHS = [
    ROOT / "api",
    ROOT / "infra",
    ROOT / "orchestrator" / "enterprise_orchestrator.py",
    ROOT / "frontend" / "app.py",
]

# Runtime code should not hardcode environment-specific infra credentials or demo placeholders.
BANNED_LITERALS = [
    "postgresql://admin:adminpassword@localhost:5432/natwest_db",
    "default_predictor",
    "localhost:47334",
    "localhost:8081",
    "localhost:5555",
]


def iter_python_files() -> list[Path]:
    files: list[Path] = []
    for item in SCAN_PATHS:
        if item.is_file():
            files.append(item)
            continue
        if item.exists():
            files.extend(path for path in item.rglob("*.py") if "__pycache__" not in path.parts)
    return files


def test_no_banned_runtime_literals():
    violations: list[str] = []
    for path in iter_python_files():
        content = path.read_text(encoding="utf-8")
        for literal in BANNED_LITERALS:
            if literal in content:
                violations.append(f"{path.relative_to(ROOT)} contains '{literal}'")
    assert not violations, "Hardcoded runtime literals detected:\n" + "\n".join(violations)
