"""Command-line entry points with a no-cost preflight path."""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from .config import (
    CHECKPOINTING_ENABLED,
    env_bool,
    env_float,
    env_int,
    project_root,
    resolve_from_root,
)


def run_preflight(*, require_credentials: bool = False) -> list[str]:
    """Validate local paths and configuration without importing CrewAI."""
    load_dotenv()
    errors: list[str] = []
    root = project_root()

    if not (3, 10) <= sys.version_info[:2] < (3, 14):
        errors.append("Python 3.10 through 3.13 is required.")

    try:
        env_bool("VERBOSE", True)
        env_bool("ENABLE_CACHE", False)
        env_int("MAX_RPM", 6, 1, 20)
        env_int("MAX_EXECUTION_SECONDS", 600, 60, 1800)
        env_float("TARGET_RUN_BUDGET_USD", 5.00, 0.50, 100.00)
    except ValueError as exc:
        errors.append(str(exc))

    context_path = resolve_from_root(
        root, os.getenv("MASTER_CONTEXT_PATH", "examples/synthetic_context.md")
    )
    if not context_path.is_file():
        errors.append(f"Context file not found: {context_path}")
    elif not context_path.read_text(encoding="utf-8").strip():
        errors.append(f"Context file is empty: {context_path}")

    evidence_dir = resolve_from_root(root, os.getenv("EVIDENCE_INPUT_DIR", "examples/data"))
    if not evidence_dir.is_dir():
        errors.append(f"Evidence directory not found: {evidence_dir}")

    output_dir = resolve_from_root(root, os.getenv("OUTPUT_DIR", "outputs"))
    try:
        output_dir.relative_to(root)
    except ValueError:
        errors.append("OUTPUT_DIR must remain inside MODEL_COUNSEL_PROJECT_ROOT.")

    if require_credentials:
        missing = [key for key in ("OPENAI_API_KEY", "SERPER_API_KEY") if not os.getenv(key)]
        if missing:
            errors.append("Missing required variables: " + ", ".join(missing))

    if CHECKPOINTING_ENABLED:
        errors.append("Security invariant violated: checkpointing must be disabled.")
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-counsel",
        description="Evidence-first multi-agent contract-supply research workflow.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="Run no-cost configuration checks.")
    check.add_argument(
        "--require-credentials",
        action="store_true",
        help="Also require OpenAI and Serper API keys.",
    )
    subparsers.add_parser("run", help="Run the live CrewAI workflow.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "check":
        errors = run_preflight(require_credentials=args.require_credentials)
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        print("Preflight passed.")
        print(f"Project root: {project_root()}")
        print(f"Checkpointing enabled: {CHECKPOINTING_ENABLED}")
        return 0

    from .workflow import main as run_workflow

    return run_workflow()
