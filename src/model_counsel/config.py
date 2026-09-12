"""Configuration helpers shared by the workflow and no-cost preflight checks."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

# CrewAI 1.15.x snapshots can serialize LLM credentials and complete prompts.
# This public project deliberately has no environment switch that enables them.
CHECKPOINTING_ENABLED = False


def env_bool(
    name: str,
    default: bool = False,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Read a strict boolean environment variable."""
    source = os.environ if environ is None else environ
    raw = source.get(name)
    if raw is None:
        return default

    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of true/false, yes/no, on/off, or 1/0.")


def env_int(
    name: str,
    default: int,
    minimum: int,
    maximum: int,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Read and clamp an integer environment variable."""
    source = os.environ if environ is None else environ
    try:
        value = int(source.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    return min(max(value, minimum), maximum)


def env_float(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
    environ: Mapping[str, str] | None = None,
) -> float:
    """Read and clamp a floating-point environment variable."""
    source = os.environ if environ is None else environ
    try:
        value = float(source.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc
    return min(max(value, minimum), maximum)


def project_root(environ: Mapping[str, str] | None = None) -> Path:
    """Resolve the project root without relying on a user-specific path."""
    source = os.environ if environ is None else environ
    configured = source.get("MODEL_COUNSEL_PROJECT_ROOT")
    return Path(configured).expanduser().resolve() if configured else Path.cwd().resolve()


def resolve_from_root(root: Path, raw_path: str) -> Path:
    """Resolve an absolute path or a project-root-relative path."""
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()
