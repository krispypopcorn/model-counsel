from __future__ import annotations

from types import SimpleNamespace

from model_counsel.validation import (
    parse_shortlist_candidates,
    require_candidate_output,
    require_complete_output,
    validate_output_files,
    validate_report_text,
)

CANONICAL_SHORTLIST = """# Report
## Shortlist of exactly three research candidates
### Candidate A — **Reusable transit containers**
### Candidate B — **Institutional safety labels**
### Candidate C — **Inspection sample kits**
"""


def test_parse_canonical_shortlist() -> None:
    assert parse_shortlist_candidates(CANONICAL_SHORTLIST) == {
        "A": "Reusable transit containers",
        "B": "Institutional safety labels",
        "C": "Inspection sample kits",
    }


def test_parse_numbered_shortlist_fallback() -> None:
    text = """## Exactly Three Research Candidates
### 1) **Candidate one**
### 2) **Candidate two**
### 3) **Candidate three**
"""
    assert parse_shortlist_candidates(text) == {
        "A": "Candidate one",
        "B": "Candidate two",
        "C": "Candidate three",
    }


def test_completion_and_candidate_identity_guardrails() -> None:
    body = "Reusable transit containers\n" + "x" * 40 + "\nhttps://example.com\nDONE"
    complete = require_complete_output("DONE", 20, 1)
    assert complete(SimpleNamespace(raw=body))[0] is True

    map_task = SimpleNamespace(output=SimpleNamespace(raw=CANONICAL_SHORTLIST))
    candidate = require_candidate_output(1, "DONE", 20, map_task, 1)
    assert candidate(SimpleNamespace(raw=body))[0] is True
    wrong = body.replace("Reusable transit containers", "Different candidate")
    assert candidate(SimpleNamespace(raw=wrong))[0] is False


def test_report_validation_lists_all_structural_errors() -> None:
    errors = validate_report_text(
        "short",
        marker="DONE",
        minimum_characters=20,
        minimum_source_urls=1,
        required_fragments=("Decision Gates",),
    )
    assert len(errors) == 4


def test_directory_validation_reports_missing_files(tmp_path) -> None:
    requirements = {
        "01_opportunity_map.md": {
            "marker": "DONE",
            "minimum_characters": 1,
            "minimum_source_urls": 0,
            "required_fragments": [],
        }
    }
    assert validate_output_files(tmp_path, requirements) == [
        "01_opportunity_map.md: file is missing"
    ]
