"""Deterministic validation for CrewAI task and generated report outputs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol


class TaskResult(Protocol):
    """Minimal task-output interface used by guardrails and unit tests."""

    raw: str | None


def validate_report_text(
    text: str,
    *,
    marker: str,
    minimum_characters: int,
    minimum_source_urls: int = 1,
    required_fragments: tuple[str, ...] = (),
) -> list[str]:
    """Return deterministic structural errors for one report body."""
    errors: list[str] = []
    cleaned = text.strip()
    if not cleaned:
        return ["file is empty"]
    if len(cleaned) < minimum_characters:
        errors.append(f"only {len(cleaned):,} characters; expected at least {minimum_characters:,}")
    if marker not in cleaned:
        errors.append(f"missing completion marker {marker}")
    source_url_count = len(re.findall(r"https?://[^\s)>]+", cleaned))
    if source_url_count < minimum_source_urls:
        errors.append(
            f"found {source_url_count} source URLs; expected at least {minimum_source_urls}"
        )
    lowered = cleaned.lower()
    for fragment in required_fragments:
        if fragment.lower() not in lowered:
            errors.append(f"missing required section or term: {fragment}")
    return errors


def require_complete_output(
    marker: str,
    minimum_characters: int,
    minimum_source_urls: int = 1,
):
    """Build a CrewAI guardrail that rejects incomplete task output."""

    def validate(result: TaskResult):
        raw = (result.raw or "").strip()
        errors = validate_report_text(
            raw,
            marker=marker,
            minimum_characters=minimum_characters,
            minimum_source_urls=minimum_source_urls,
        )
        return (False, errors[0]) if errors else (True, raw)

    return validate


def parse_shortlist_candidates(text: str) -> dict[str, str]:
    """Extract exact shortlist names from canonical or numbered headings."""
    canonical_matches = re.findall(
        r"^### Candidate ([A-C])[ \t]+[—-][ \t]+\*\*(.+?)\*\*"
        r"(?:[ \t]+[^\r\n]*)?$",
        text,
        flags=re.MULTILINE,
    )
    canonical = {letter: name.strip() for letter, name in canonical_matches}
    if set(canonical) == {"A", "B", "C"}:
        return canonical

    shortlist_section_match = re.search(
        r"^#{1,3}[ \t]+(?:Shortlist:[ \t]*)?Exactly Three Research "
        r"Candidates[^\r\n]*$(.*?)(?=^#[^#]|\Z)",
        text,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if not shortlist_section_match:
        return canonical

    numbered_matches = re.findall(
        r"^#{2,3}[ \t]+([123])\)[ \t]+\*\*(.+?)\*\*"
        r"(?:[ \t]+[^\r\n]*)?$",
        shortlist_section_match.group(1),
        flags=re.MULTILINE,
    )
    return {chr(64 + int(position)): name.strip() for position, name in numbered_matches}


def require_opportunity_map_output(
    marker: str,
    minimum_characters: int,
    minimum_source_urls: int = 6,
):
    """Require a complete map with three distinct, readable candidates."""
    complete_guardrail = require_complete_output(marker, minimum_characters, minimum_source_urls)

    def validate(result: TaskResult):
        complete, value = complete_guardrail(result)
        if not complete:
            return complete, value
        shortlist = parse_shortlist_candidates(value)
        if set(shortlist) != {"A", "B", "C"}:
            return (
                False,
                "The shortlist must contain exactly three headings formatted as "
                "'### Candidate A — **Exact candidate name**', then B and C.",
            )
        if len({name.casefold().strip() for name in shortlist.values()}) != 3:
            return False, "Candidates A, B, and C must be distinct opportunities."
        return True, value

    return validate


def require_candidate_output(
    rank: int,
    marker: str,
    minimum_characters: int,
    opportunity_map_task: Any,
    minimum_source_urls: int = 4,
):
    """Require a deep dive to identify its assigned shortlist candidate."""
    complete_guardrail = require_complete_output(marker, minimum_characters, minimum_source_urls)
    candidate_letter = chr(64 + rank)

    def validate(result: TaskResult):
        complete, value = complete_guardrail(result)
        if not complete:
            return complete, value
        map_output = getattr(opportunity_map_task, "output", None)
        map_raw = (getattr(map_output, "raw", None) or "").strip()
        expected_name = parse_shortlist_candidates(map_raw).get(candidate_letter)
        if not expected_name:
            return False, f"Could not resolve Candidate {candidate_letter}."
        if expected_name.casefold() not in value[:1500].casefold():
            return (
                False,
                f"This is Candidate {candidate_letter}. Rewrite the report for "
                f"'{expected_name}' and identify it in the opening heading.",
            )
        return True, value

    return validate


def validate_output_files(
    output_dir: Path,
    requirements: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Return completeness and candidate-consistency errors for output files."""
    errors: list[str] = []
    for filename, rules in requirements.items():
        path = output_dir / filename
        if not path.exists():
            errors.append(f"{filename}: file is missing")
            continue
        report_errors = validate_report_text(
            path.read_text(encoding="utf-8"),
            marker=str(rules["marker"]),
            minimum_characters=int(rules["minimum_characters"]),
            minimum_source_urls=int(rules["minimum_source_urls"]),
            required_fragments=tuple(rules.get("required_fragments", ())),
        )
        errors.extend(f"{filename}: {error}" for error in report_errors)

    map_path = output_dir / "01_opportunity_map.md"
    if not map_path.exists():
        return errors
    shortlist = parse_shortlist_candidates(map_path.read_text(encoding="utf-8"))
    if set(shortlist) != {"A", "B", "C"}:
        errors.append("01_opportunity_map.md: expected Candidate A/B/C headings")
        return errors
    if len({name.casefold().strip() for name in shortlist.values()}) != 3:
        errors.append("01_opportunity_map.md: shortlist names are not distinct")
        return errors
    for rank in range(1, 4):
        letter = chr(64 + rank)
        filename = f"02{chr(96 + rank)}_validation_candidate_{rank}.md"
        path = output_dir / filename
        if (
            path.exists()
            and shortlist[letter].casefold()
            not in path.read_text(encoding="utf-8")[:1500].casefold()
        ):
            errors.append(
                f"{filename}: opening does not identify Candidate {letter}: {shortlist[letter]}"
            )
    return errors
