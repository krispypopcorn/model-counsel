"""
Budget-controlled sequential CrewAI business research workflow.

This version is designed to prevent the failure mode where CrewAI reports a
successful run even though one or more Markdown reports were truncated.

Key design choices:
- Uses a deterministic sequential process with one directly assigned agent per task.
- Uses separate completion-token budgets for the scout, researcher, critic, and writer.
- Gives agents explicit search budgets and bounds them with max_iter.
- Keeps planning, delegation, memory, and automatic full reruns disabled.
- Requires exact completion markers and required sections in every output file.
- Archives stale outputs before each run so old files cannot create a false success.
- Records run configuration, package versions, usage metrics, and completion status.

Recommended contract-first rerun:
- Set RUN_MODE=discovery.
- Set OUTPUT_DIR=Output_revised or another fresh directory.
- Keep the original incomplete run for comparison.

Revision mode:
- Set RUN_MODE=revision.
- Set PREVIOUS_REPORT_PATH to a previously completed final report.
- The previous report must contain <!-- COMPLETE:FINAL_REPORT -->.

Important:
No local script can guarantee an exact provider charge. The target budget is a
planning guardrail. Actual cost depends on model pricing, prompt size, tool calls,
retries, and provider behaviour.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import traceback
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .config import (
    CHECKPOINTING_ENABLED,
    env_bool,
    env_float,
    env_int,
    project_root,
    resolve_from_root,
)
from .validation import (
    require_candidate_output,
    require_complete_output,
    require_opportunity_map_output,
    validate_output_files,
)

# CrewAI initializes tracing and checkpoint storage while it is imported. Load
# the local environment and pin that storage before importing CrewAI so even a
# no-cost preflight remains project-local.
load_dotenv()
_EARLY_ROOT_DIR = project_root()
_EARLY_STORAGE_RAW = Path(os.getenv("CREWAI_STORAGE_DIR", ".model_counsel")).expanduser()
_EARLY_STORAGE_DIR = (
    _EARLY_STORAGE_RAW if _EARLY_STORAGE_RAW.is_absolute() else _EARLY_ROOT_DIR / _EARLY_STORAGE_RAW
).resolve()
_EARLY_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["CREWAI_STORAGE_DIR"] = str(_EARLY_STORAGE_DIR)

from crewai import LLM, Agent, Crew, Process, Task  # noqa: E402
from crewai_tools import ScrapeWebsiteTool, SerperDevTool  # noqa: E402


def package_version(package_name: str) -> str | None:
    """Return an installed package version without failing the run."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


# ============================================================
# 2. Paths and run mode
# ============================================================

ROOT_DIR = _EARLY_ROOT_DIR

CREWAI_STORAGE_DIR = resolve_from_root(
    ROOT_DIR,
    os.getenv("CREWAI_STORAGE_DIR", ".model_counsel"),
).resolve()
CREWAI_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
# CrewAI's path helper accepts an absolute path through this variable. Keeping
# checkpoint SQLite files inside the project makes the run portable and avoids
# relying on a writable user-level application-data directory.
os.environ["CREWAI_STORAGE_DIR"] = str(CREWAI_STORAGE_DIR)

CONTEXT_PATH = resolve_from_root(
    ROOT_DIR,
    os.getenv("MASTER_CONTEXT_PATH", "examples/synthetic_context.md"),
)

OUTPUT_DIR = resolve_from_root(
    ROOT_DIR,
    os.getenv("OUTPUT_DIR", "outputs"),
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def crewai_output_path(filename: str) -> str:
    """Return a project-relative output path safe for CrewAI Task.output_file."""
    absolute_path = (OUTPUT_DIR / filename).resolve()

    try:
        relative_path = absolute_path.relative_to(ROOT_DIR)
    except ValueError as exc:
        raise ValueError(
            "OUTPUT_DIR must be inside the project directory because CrewAI "
            "Task.output_file requires project-relative paths. "
            f"Project directory: {ROOT_DIR}; output directory: {OUTPUT_DIR}"
        ) from exc

    return relative_path.as_posix()


RUN_MODE = os.getenv("RUN_MODE", "discovery").strip().lower()
if RUN_MODE not in {"discovery", "revision"}:
    raise ValueError("RUN_MODE must be either 'discovery' or 'revision'.")

PREVIOUS_REPORT_PATH = resolve_from_root(
    ROOT_DIR,
    os.getenv(
        "PREVIOUS_REPORT_PATH",
        str(OUTPUT_DIR / "business_research_report.md"),
    ),
)

if not CONTEXT_PATH.exists():
    raise FileNotFoundError(f"Master context was not found at: {CONTEXT_PATH}")

founder_context = CONTEXT_PATH.read_text(encoding="utf-8").strip()
if not founder_context:
    raise ValueError("context file is empty.")

EVIDENCE_INPUT_DIR = resolve_from_root(
    ROOT_DIR,
    os.getenv("EVIDENCE_INPUT_DIR", "examples/data"),
)
EVIDENCE_PACK_MAX_CHARACTERS = env_int(
    "EVIDENCE_PACK_MAX_CHARACTERS",
    80000,
    5000,
    250000,
)
EVIDENCE_INPUT_FILENAMES = (
    "category_summary.csv",
    "buyer_interviews.csv",
    "supplier_quotes.csv",
    "evidence_notes.md",
    "buyer_universe_summary.csv",
    "external_dataset_catalog.csv",
    "contract_evidence.csv",
)
EVIDENCE_FILE_MAX_CHARACTERS = {
    "category_summary.csv": 14000,
    "buyer_interviews.csv": 8000,
    "supplier_quotes.csv": 8000,
    "evidence_notes.md": 8000,
    "buyer_universe_summary.csv": 10000,
    "external_dataset_catalog.csv": 10000,
    "contract_evidence.csv": 45000,
}


def load_local_evidence_pack() -> tuple[str, list[dict[str, Any]]]:
    """Load bounded, explicitly named real-world evidence files."""
    sections: list[str] = []
    manifest: list[dict[str, Any]] = []
    remaining = EVIDENCE_PACK_MAX_CHARACTERS

    for filename in EVIDENCE_INPUT_FILENAMES:
        path = EVIDENCE_INPUT_DIR / filename
        if not path.exists():
            manifest.append({"filename": filename, "exists": False, "included_characters": 0})
            continue

        raw = path.read_text(encoding="utf-8").strip()
        file_limit = EVIDENCE_FILE_MAX_CHARACTERS.get(filename, remaining)
        included = raw[: min(remaining, file_limit)] if remaining > 0 else ""
        truncated = len(included) < len(raw)
        data_rows = max(0, len(raw.splitlines()) - 1) if path.suffix.casefold() == ".csv" else None
        manifest.append(
            {
                "filename": filename,
                "exists": True,
                "size_characters": len(raw),
                "included_characters": len(included),
                "truncated": truncated,
                "data_rows": data_rows,
            }
        )

        if included:
            sections.append(
                f"### Local evidence file: {filename}\n\n{included}"
                + ("\n\n[TRUNCATED AT THE CONFIGURED EVIDENCE-PACK LIMIT]" if truncated else "")
            )
            remaining -= len(included)

    return "\n\n".join(sections), manifest


local_evidence_pack, evidence_pack_manifest = load_local_evidence_pack()


# ============================================================
# 3. Credentials and model settings
# ============================================================

required_keys = ["OPENAI_API_KEY", "SERPER_API_KEY"]
missing_keys = sorted(key for key in required_keys if not os.getenv(key))

if missing_keys:
    raise EnvironmentError(
        "Missing required environment variables: "
        + ", ".join(missing_keys)
        + ". Add them to a local .env file."
    )

WORKER_MODEL = os.getenv(
    "WORKER_MODEL",
    "openai/gpt-5.4-mini",
).strip()

# Backward-compatible fallback: an existing MANAGER_MODEL value can become
# the final synthesis model after switching away from hierarchical execution.
FINAL_MODEL = os.getenv(
    "FINAL_MODEL",
    os.getenv("MANAGER_MODEL", "openai/gpt-5.6-terra"),
).strip()

for variable_name, model_name in {
    "WORKER_MODEL": WORKER_MODEL,
    "FINAL_MODEL": FINAL_MODEL,
}.items():
    if not model_name.startswith("openai/"):
        raise ValueError(f"{variable_name} must use an OpenAI model in this OpenAI-only build.")

VERBOSE = env_bool("VERBOSE", True)
OPENAI_MAX_RETRIES = env_int("OPENAI_MAX_RETRIES", 1, 0, 3)
MAX_RPM = env_int("MAX_RPM", 6, 1, 20)
MAX_EXECUTION_SECONDS = env_int(
    "MAX_EXECUTION_SECONDS",
    600,
    60,
    1800,
)

TARGET_RUN_BUDGET_USD = env_float("TARGET_RUN_BUDGET_USD", 5.00, 0.50, 100.00)
ENABLE_CACHE = env_bool("ENABLE_CACHE", False)

SCOUT_MAX_ITER = env_int("SCOUT_MAX_ITER", 6, 1, 6)
RESEARCHER_MAX_ITER = env_int("RESEARCHER_MAX_ITER", 14, 4, 14)
CRITIC_MAX_ITER = env_int("CRITIC_MAX_ITER", 4, 1, 4)
STRATEGIST_MAX_ITER = env_int("STRATEGIST_MAX_ITER", 1, 1, 3)

SCOUT_MAX_COMPLETION_TOKENS = env_int(
    "SCOUT_MAX_COMPLETION_TOKENS",
    8000,
    2000,
    12000,
)
RESEARCHER_MAX_COMPLETION_TOKENS = env_int(
    "RESEARCHER_MAX_COMPLETION_TOKENS",
    9000,
    3000,
    18000,
)
CRITIC_MAX_COMPLETION_TOKENS = env_int(
    "CRITIC_MAX_COMPLETION_TOKENS",
    8000,
    2500,
    14000,
)
FINAL_MAX_COMPLETION_TOKENS = env_int(
    "FINAL_MAX_COMPLETION_TOKENS",
    14000,
    4000,
    20000,
)
TOOL_MAX_COMPLETION_TOKENS = env_int(
    "TOOL_MAX_COMPLETION_TOKENS",
    1200,
    400,
    3000,
)

SEARCH_RESULTS_PER_QUERY = env_int(
    "SEARCH_RESULTS_PER_QUERY",
    6,
    2,
    6,
)
SCOUT_SEARCH_LIMIT = env_int("SCOUT_SEARCH_LIMIT", 6, 0, 6)
RESEARCHER_SEARCH_LIMIT = env_int("RESEARCHER_SEARCH_LIMIT", 5, 0, 6)
CRITIC_SEARCH_LIMIT = env_int("CRITIC_SEARCH_LIMIT", 4, 0, 4)
RESEARCHER_SCRAPE_LIMIT = env_int("RESEARCHER_SCRAPE_LIMIT", 6, 0, 8)


# ============================================================
# 4. Revision context
# ============================================================

previous_report = ""

if RUN_MODE == "revision":
    if not PREVIOUS_REPORT_PATH.exists():
        raise FileNotFoundError(
            "RUN_MODE is 'revision', but PREVIOUS_REPORT_PATH does not exist: "
            f"{PREVIOUS_REPORT_PATH}"
        )

    previous_report = PREVIOUS_REPORT_PATH.read_text(encoding="utf-8").strip()

    if "<!-- COMPLETE:FINAL_REPORT -->" not in previous_report:
        raise ValueError(
            "Revision mode requires a completed prior report containing "
            "<!-- COMPLETE:FINAL_REPORT -->. Use discovery mode when the prior "
            "report was truncated or incomplete."
        )

revision_instructions = (
    f"""
REVISION MODE
-------------
A completed previous report is included below.

Do not repeat acceptable work. Concentrate on:
- Unsupported or weak claims.
- Opportunities rejected or heavily penalized by the critic.
- Missing buyer, competitor, supplier, regulatory, landed-cost, cash-flow,
  customer-acquisition, or working-capital evidence.
- Replacement opportunities from different sectors or business models.
- More precise and cheaper commercial validation steps.

Previous completed report:
--------------------------
{previous_report}
"""
    if previous_report
    else """
DISCOVERY MODE
--------------
This is a fresh contract-first discovery pass. Begin with tenders, awards,
contract history, standing offers, buyer specifications, and comparable private
projects. Derive verticals from purchasing evidence rather than brainstorming
products first. Do not rely on any incomplete prior report.
"""
)

local_evidence_context = (
    f"""
LOCAL REAL-WORLD EVIDENCE PACK
------------------------------
The following files were supplied locally. Evaluate their provenance, status,
dates, definitions, duplicates, and missing fields before relying on them.
Local evidence does not override a contradictory primary source.

{local_evidence_pack}
"""
    if local_evidence_pack
    else """
LOCAL REAL-WORLD EVIDENCE PACK
------------------------------
No populated local evidence files were supplied for this run. Public web
research may identify opportunities, but missing bulk tender exports, award
records, buyer interviews, and supplier RFQs must be listed as evidence gaps.
"""
)

shared_research_context = f"""
FOUNDER MANDATE
---------------
{founder_context}

{revision_instructions}

{local_evidence_context}

EXECUTION AND EVIDENCE RULES
----------------------------
1. Be concise, complete, and evidence-dense.
2. Do not repeat the full founder context in the output.
3. Separate verified facts, evidence-backed estimates, working assumptions,
   and unknowns requiring validation.
4. Use web searches only for material questions and prefer primary or
   authoritative sources.
5. Record source title, publisher, URL, publication or data date when available,
   access date, the exact proposition supported, and confidence.
6. Do not search the same claim repeatedly.
7. Never force a winner when evidence is weak.
8. Distinguish category existence from founder-specific customer demand.
9. Research incumbent competitors and explain why a customer might switch.
10. Recommend interviews, supplier quotes, samples, or paid data when another
    web search would not reliably resolve the uncertainty.
11. Do not treat an LOI as revenue, financing collateral, or production approval.
12. Finish every required section before writing the completion marker.
13. Start with contract and award evidence; do not invent a product and then
    search only for facts that support it.
14. Distinguish award value, amendment value, maximum ceiling, estimated
    quantity, committed minimum, and actual spend.
15. Do not count an open tender, supplier catalogue, marketplace page, or generic
    project announcement as a completed award.
16. When a procurement portal is inaccessible or omits values, state the exact
    missing field and the manual download, paid access, information request, or
    buyer interview needed.
17. Every shortlisted vertical must cite at least two identifiable purchasing
    events from the last 36 months or be clearly rejected for insufficient
    contract evidence.
18. Across the final shortlist, preserve enough evidence to reverse-engineer at
    least five real contracts or explicitly issue a no-go conclusion.
"""


# ============================================================
# 5. LLMs and tools
# ============================================================

common_llm_settings = {
    "timeout": 180,
    "max_retries": OPENAI_MAX_RETRIES,
}

scout_llm = LLM(
    model=WORKER_MODEL,
    max_completion_tokens=SCOUT_MAX_COMPLETION_TOKENS,
    **common_llm_settings,
)

researcher_llm = LLM(
    model=WORKER_MODEL,
    max_completion_tokens=RESEARCHER_MAX_COMPLETION_TOKENS,
    **common_llm_settings,
)

critic_llm = LLM(
    model=WORKER_MODEL,
    max_completion_tokens=CRITIC_MAX_COMPLETION_TOKENS,
    **common_llm_settings,
)

final_llm = LLM(
    model=FINAL_MODEL,
    max_completion_tokens=FINAL_MAX_COMPLETION_TOKENS,
    **common_llm_settings,
)

# Tool selection does not need the full report-writing output allowance.
# CrewAI supports a separate function_calling_llm for tool calls.
tool_calling_llm = LLM(
    model=WORKER_MODEL,
    max_completion_tokens=TOOL_MAX_COMPLETION_TOKENS,
    **common_llm_settings,
)

# These are official CrewAI tools. Search and page-reading limits below are
# prompt-level limits backed by each agent's hard max_iter. CrewAI 1.15.7 does
# not document max_usage_count for SerperDevTool, so this script does not rely
# on that unsupported argument.
scout_search_tool = SerperDevTool(n_results=SEARCH_RESULTS_PER_QUERY)
researcher_search_tool = SerperDevTool(n_results=SEARCH_RESULTS_PER_QUERY)
researcher_scrape_tool = ScrapeWebsiteTool()
critic_search_tool = SerperDevTool(n_results=SEARCH_RESULTS_PER_QUERY)


# ============================================================
# 6. Agents
# ============================================================

scout = Agent(
    role="Contract, Award, and Institutional Supply Opportunity Scout",
    goal=(
        "Mine real purchasing evidence and identify genuinely distinct, high-value "
        "contract-supply verticals that fit the founder's sourcing, quality, sales, "
        "capital, and income objectives."
    ),
    backstory=(
        "You are a procurement-data analyst. You begin with tenders, award notices, "
        "contract history, standing offers, buyer specifications, project pipelines, "
        "and awarded vendors. You derive opportunities from repeated purchasing "
        "patterns and never confuse a category, tender ceiling, or open solicitation "
        "with revenue available to a new entrant."
    ),
    llm=scout_llm,
    function_calling_llm=tool_calling_llm,
    tools=[scout_search_tool] if SCOUT_SEARCH_LIMIT > 0 else [],
    allow_delegation=False,
    verbose=VERBOSE,
    max_iter=SCOUT_MAX_ITER,
    max_retry_limit=1,
    max_execution_time=MAX_EXECUTION_SECONDS,
    respect_context_window=True,
    reasoning=False,
    inject_date=True,
)

researcher = Agent(
    role="Contract Reconstruction, Buyer, Supplier, and Commercial Intelligence Lead",
    goal=(
        "Reverse-engineer real contracts and validate the shortlisted verticals using "
        "buyer, award, incumbent, specification, supplier, pricing, financing, trade, "
        "regulatory, quality, and contribution evidence."
    ),
    backstory=(
        "You are a skeptical institutional procurement and supply-chain researcher. "
        "You distinguish manufacturers from traders, award ceilings from actual "
        "spend, nominal gross margin from cash contribution, and an open tender from "
        "a winnable contract. You test qualification, bid security, payment timing, "
        "quality control, supplier capacity, and incumbent advantage."
    ),
    llm=researcher_llm,
    function_calling_llm=tool_calling_llm,
    tools=(
        [researcher_search_tool, researcher_scrape_tool]
        if RESEARCHER_SEARCH_LIMIT > 0 or RESEARCHER_SCRAPE_LIMIT > 0
        else []
    ),
    allow_delegation=False,
    verbose=VERBOSE,
    max_iter=RESEARCHER_MAX_ITER,
    max_retry_limit=1,
    max_execution_time=MAX_EXECUTION_SECONDS,
    respect_context_window=True,
    reasoning=False,
    inject_date=True,
)

critic = Agent(
    role="Contract Risk, Financing, Competition, and Evidence Critic",
    goal=(
        "Stress-test the shortlisted opportunities against bid accessibility, "
        "contract concentration, capital preservation, financing, security, "
        "competitive response, working capital, regulation, quality, logistics, "
        "customer acquisition, and evidence constraints."
    ),
    backstory=(
        "You are a conservative operating and financial risk executive. You reject "
        "fake margin, score inflation, unsupported differentiation, and cash-flow "
        "structures that can expose more capital than intended."
    ),
    llm=critic_llm,
    function_calling_llm=tool_calling_llm,
    tools=[critic_search_tool] if CRITIC_SEARCH_LIMIT > 0 else [],
    allow_delegation=False,
    verbose=VERBOSE,
    max_iter=CRITIC_MAX_ITER,
    max_retry_limit=1,
    max_execution_time=MAX_EXECUTION_SECONDS,
    respect_context_window=True,
    reasoning=False,
    inject_date=True,
)

strategist = Agent(
    role="Contract-Supply Investment Committee Writer and Validation Strategist",
    goal=(
        "Reconcile the contract evidence into a complete decision report and a "
        "commercially testable buyer, supplier, financing, and bid-validation plan "
        "without overstating certainty."
    ),
    backstory=(
        "You are an execution-focused contract-supply operator and investment-committee "
        "writer. You preserve disagreements and uncertainty, distinguish contract "
        "value from gross profit and cash exposure, design reversible tests, and "
        "impose evidence and financing gates before a bid or purchase commitment."
    ),
    llm=final_llm,
    tools=[],
    allow_delegation=False,
    verbose=VERBOSE,
    max_iter=STRATEGIST_MAX_ITER,
    max_retry_limit=1,
    max_execution_time=MAX_EXECUTION_SECONDS,
    respect_context_window=True,
    reasoning=False,
    inject_date=True,
)


# ============================================================
# 8. Tasks
# ============================================================

opportunity_map_task = Task(
    name="opportunity_map",
    description=f"""
{shared_research_context}

TASK: BUILD THE CONTRACT-EVIDENCE LANDSCAPE AND OPPORTUNITY MAP
---------------------------------------------------------------
Begin with real purchasing events from the last 36 months. Search procurement
and award evidence before naming opportunity concepts. Create exactly eight
genuinely distinct contract-supply verticals derived from repeated or high-value
purchasing evidence.

Research budget for this task:
- Use no more than {SCOUT_SEARCH_LIMIT} targeted web searches.
- Prefer dataset, award, tender, standing-offer, procurement-plan, and project
  searches that answer several questions.
- Stop searching when another query is unlikely to change the shortlist.

Coverage requirements:
- At least five sectors.
- At least three viable transaction structures across importer/distributor,
  manufacturer's representative, procurement integrator, joint bidder,
  subcontractor, or value-added supplier.
- No minor variations of the same product or supply problem.
- Do not default to small-ticket replenishment, ecommerce, or broad MRO.
- Keep each profile concise enough to complete the full task.
- Treat CAD 250,000–2 million as the scale-stage destination, not a first-order
  minimum. Require evidence of an entry-to-scale ladder: approximately
  CAD 10,000–100,000 validation orders, CAD 50,000–300,000 reference-building
  contracts, and recurring CAD 250,000–2 million scale-stage awards.
- Target one to ten anchor accounts and at least CAD 100,000 plausible gross
  profit per scale-stage anchor contract. Do not reject a smaller first order
  solely because its gross profit is below that target.
- Cover the search territories in the mandate, but do not select a territory
  merely because the founder mentioned it.

For each opportunity include:
1. Specific product family and buyer-issued requirement.
2. At least two identifiable tenders, awards, contracts, standing offers, or
   comparable private projects from the last 36 months when available.
3. For each purchasing event: buyer, title, identifier, date, status, value or
   quantity, winner when known, term, and source URL.
4. Whether value is an award, amendment, ceiling, estimate, or unknown.
5. Target buyer, decision-maker, procurement channel, buying trigger, and bid cycle.
6. Likely business model and why the founder is not trivially disintermediated.
7. Founder advantage in specification, sourcing, languages, quality assurance,
   data, financing coordination, or account development.
8. Likely sourcing regions, supplier type, inspectability, testability, and
   supplier-redundancy potential.
9. Indicative contract-to-gross-profit bridge and maximum cash-exposure questions.
10. Qualification, incumbent, certification, bonding/security, warranty,
    ethical-sourcing, logistics, and product-liability risks.
11. Repeat-contract, rollout, framework, or account-expansion potential.
12. Evidence that an entrant can progress from a smaller validation or
    reference-building contract to the scale-stage awards in this category.
13. Evidence status, contradictions, missing data, and preliminary score using
    the founder's contract-supply rubric.

Then provide:
- A procurement-evidence table that lists every cited purchasing event without
  duplicating amendments as separate contracts.
- A comparison table for all eight verticals.
- A shortlist of exactly three research candidates. A weak candidate may be
  shortlisted only for comparison and must be labelled clearly.
- Prefer these exact machine-readable shortlist headings:
  `### Candidate A — **Exact candidate name**`
  `### Candidate B — **Exact candidate name**`
  `### Candidate C — **Exact candidate name**`
  If you instead number the shortlist 1, 2, and 3, those positions will map
  deterministically to Candidate A, B, and C.
- A short explanation of why each non-shortlisted concept was deferred.

Do not shortlist a vertical supported only by supplier marketing, generic market
size, or an open tender with no comparable award history. If the search budget
cannot support eight evidence-backed verticals, include clearly labelled
insufficient-evidence concepts and do not recommend them.

End the output with exactly:
<!-- COMPLETE:OPPORTUNITY_MAP -->
""",
    expected_output=(
        "A contract-first landscape with eight distinct supply verticals, real "
        "purchasing-event evidence, transaction structures, exactly three research "
        "candidates, deferral reasons, score ranges, and the marker."
    ),
    agent=scout,
    markdown=True,
    output_file=crewai_output_path("01_opportunity_map.md"),
    guardrail=require_opportunity_map_output(
        "<!-- COMPLETE:OPPORTUNITY_MAP -->",
        5500,
    ),
    guardrail_max_retries=1,
)


def build_candidate_validation_task(rank: int) -> Task:
    """Create a dedicated deep-dive task for one shortlisted opportunity."""
    output_filename = f"02{chr(96 + rank)}_validation_candidate_{rank}.md"
    completion_marker = f"<!-- COMPLETE:VALIDATION_{rank} -->"
    candidate_letter = chr(64 + rank)

    return Task(
        name=f"validation_candidate_{rank}",
        description=f"""
{shared_research_context}

TASK: DEEP-DIVE SHORTLIST CANDIDATE {candidate_letter}
------------------------------------------------------
Read the opportunity map and validate the concept labelled exactly
"Candidate {candidate_letter}" in its "Shortlist of exactly three research
candidates" section.

The numbered concepts in the earlier "Eight opportunity concepts" section are
not shortlist positions. Do not select concept #{rank} merely because this task
is validation task #{rank}. Copy the exact bold Candidate {candidate_letter}
name from the shortlist into the first heading of this report. Do not repeat a
candidate handled by another validation task or substitute a different concept.

Research budget for this candidate:
- Use no more than {RESEARCHER_SEARCH_LIMIT} targeted web searches.
- Read no more than {RESEARCHER_SCRAPE_LIMIT} underlying webpages.
- Do not rely on search snippets when an important claim requires the full page.
- Prefer executed contracts, award notices, tender documents, bid tabulations,
  standing offers, buyer specifications, official procurement data, capital
  plans, regulatory sources, written supplier evidence, and company filings.

Required evidence coverage where relevant and publicly available:
- Reverse-engineer at least two real contracts or comparable private projects
  from the last 36 months for this exact vertical.
- At least one buyer-issued specification, evaluation, qualification, or
  procurement-process source.
- At least two named awarded vendors, incumbents, or substitute solutions.
- At least one supplier, pricing, quotation, MOQ, capacity, lead-time, payment
  term, inspection, or transaction source.
- At least one regulatory, tariff, classification, insurance, quality, or
  certification source when materially relevant.
- At least one contradictory fact, limitation, or unresolved evidence gap.

Provide:
1. Narrow contract-supply vertical, exact products, and founder role.
2. Contract reconstruction table for at least two real purchasing events:
   buyer, title, identifier, date, status, award/ceiling/estimated value, term,
   estimated quantities, winner, other bidders when known, and source URL.
3. Exact buyer-issued specifications, acceptance criteria, samples, testing,
   certifications, references, insurance, bonding/security, local inventory,
   delivery, service, ethical-sourcing, and reporting requirements.
4. Buyer organization, decision-makers, procurement route, bid calendar,
   prequalification, approval path, and reachable-account logic.
5. Named awarded vendors, incumbents, substitutes, likely bidder count, and
   specific reason a buyer would admit or switch to a new supplier.
6. Supplier types, sourcing regions, manufacturer/exporter/trader status,
   production capacity, supplier redundancy, and visit/audit feasibility.
7. Written quality plan: specification, golden sample, lab testing, factory
   audit, in-process inspection, pre-shipment inspection, AQL or other acceptance
   criteria, traceability, warranty, and corrective-action process.
8. MOQ, unit price, tooling, capacity, lead time, payment terms, Incoterms,
   returns, warranty, and quotation evidence when available.
9. HS-code family only when supportable; tariff, customs, labelling, sanctions,
   forced-labour/ethical-sourcing, regulatory, insurance, and liability issues.
10. Contract-to-gross-profit bridge distinguishing contract value, committed
    minimum, estimated utilization, founder revenue, gross profit, contribution,
    operating profit, and cash exposure.
11. Base, downside, and severe-downside cases, including 10% selling-price
    pressure, 10% cost increase, 60-day payment delay, 2% defects, partial
    utilization, and one rejected shipment.
12. Compare importer/distributor, manufacturer's representative, procurement
    integrator, joint-bid/subcontract, and direct-ship transaction structures.
13. Cash-conversion timeline showing customer deposits, supplier milestones,
    freight, duties, acceptance, holdbacks, financing need, and peak cash at risk.
14. Bid/no-bid feasibility: mandatory requirements the founder meets, could
    partner to meet, or cannot presently meet.
15. Buyer-acquisition, relationship-building, sample, quote, and bid workload.
16. Defensibility, concentration, incumbent response, and disintermediation risk.
17. Cheapest real-world validation step before bidding, travelling, or committing
    to production.
18. Source ledger with title, publisher, URL, document/data date, access date,
    exact proposition, evidence tier, contradiction, and confidence.

Explicitly model:
- contracts needed for approximately CAD 200,000 annual gross profit;
- contracts and operating profit needed for approximately CAD 100,000 founder income;
- plausible win rate and qualified pipeline required;
- sales-cycle length and revenue concentration;
- founder-time, partner, staffing, insurance, financing, and operating implications.

Do not fabricate precision. State missing variables and the exact interview, RFQ,
sample, test, tender download, paid dataset, information request, financing quote,
or professional advice required. If actual award value is unavailable, do not
replace it with a market-size estimate.

End the output with exactly:
{completion_marker}
""",
        expected_output=(
            f"A complete contract reconstruction and commercial deep dive for "
            f"shortlisted opportunity #{rank}, including two real purchasing events, "
            f"qualification, quality, supply, financing, gross-profit economics, "
            f"bid feasibility, risks, unknowns, and the marker."
        ),
        agent=researcher,
        context=[opportunity_map_task],
        tools=[researcher_search_tool, researcher_scrape_tool],
        markdown=True,
        output_file=crewai_output_path(output_filename),
        guardrail=require_candidate_output(
            rank,
            completion_marker,
            5000,
            opportunity_map_task,
        ),
        guardrail_max_retries=1,
    )


validation_candidate_1_task = build_candidate_validation_task(1)
validation_candidate_2_task = build_candidate_validation_task(2)
validation_candidate_3_task = build_candidate_validation_task(3)

validation_synthesis_task = Task(
    name="validation_synthesis",
    description=f"""
{shared_research_context}

TASK: SYNTHESIZE THE THREE VALIDATION DEEP DIVES
------------------------------------------------
Compare the opportunity map and all completed candidate deep dives. Do not use
new web research in this task. Preserve source URLs, important contradictions,
and unknowns from the deep dives.

Required structure:
1. Executive comparison.
2. Evidence-quality comparison.
3. Five-contract reconstruction table using the strongest non-duplicate
   purchasing events across the deep dives.
4. Buyer access, qualification, bid-cycle, and achievable pipeline comparison.
5. Incumbent, bidder, switching/admission-wedge, and disintermediation comparison.
6. Specification, supplier, quality-assurance, and operational comparison.
7. Contract-to-gross-profit, financing, security, cash-cycle, and maximum-cash-
   exposure comparison.
8. Path to CAD 200,000 annual gross profit and CAD 100,000 founder income.
9. Base, downside, severe-downside, rejected-shipment, and delayed-payment comparison.
10. Importer, representative, procurement-integrator, and joint-bid structure comparison.
11. Cheapest next real-world validation step for each candidate.
12. Open evidence gaps and exact data-acquisition actions.

Do not rank candidates using one-point precision. Use score ranges and explain
which differences are not economically meaningful. Reject any candidate whose
deal-size thesis rests on a framework ceiling, unawarded tender, market-size
report, or unsupported margin assumption.

End the output with exactly:
<!-- COMPLETE:VALIDATION -->
""",
    expected_output=(
        "A complete comparison of all candidate deep dives with five real contract "
        "reconstructions, evidence quality, qualification, competition, quality, "
        "financing, gross-profit math, downside cases, next tests, and the marker."
    ),
    agent=researcher,
    context=[
        opportunity_map_task,
        validation_candidate_1_task,
        validation_candidate_2_task,
        validation_candidate_3_task,
    ],
    tools=[],
    markdown=True,
    output_file=crewai_output_path("02_validation_report.md"),
    guardrail=require_complete_output(
        "<!-- COMPLETE:VALIDATION -->",
        7500,
        6,
    ),
    guardrail_max_retries=1,
)

risk_task = Task(
    name="risk_audit",
    description=f"""
{shared_research_context}

TASK: CONSERVATIVE RISK AND EVIDENCE AUDIT
------------------------------------------
Audit every shortlisted opportunity using the founder constraints and the full
100-point rubric. Use existing research first. Use no more than
{CRITIC_SEARCH_LIMIT} searches, and only when a material external fact could
change the decision.

For each opportunity:
1. Recalculate or challenge the score.
2. Provide a reasonable score range rather than false one-point precision.
3. Identify strongest evidence and unsupported or circular claims.
4. Verify that at least two real purchasing events support the vertical and that
   any contract values are correctly classified.
5. Test mandatory qualification, reference, approved-brand, local inventory,
   certification, insurance, bid-security, and performance-security requirements.
6. Test whether differentiation survives incumbent and bidder comparison.
7. Estimate validation capital, bid capital, peak working capital, maximum
   contractual exposure, and financing feasibility.
8. Stress-test late payment, partial utilization, freight, rejected shipments,
   defects, returns, tariffs, supplier failure, concentration, incumbent price
   response, and sales-cycle delay.
9. Test the paths to CAD 200,000 annual gross profit and CAD 100,000 founder income.
10. Classify it as PROCEED TO VALIDATION, REVISE, WATCHLIST, or REJECT.

Then provide:
- Ranked top three, or a clear no-go conclusion.
- A first validation candidate only if evidence justifies it.
- A bid/no-bid readiness classification for each candidate.
- Complete rejection log.
- Evidence gaps requiring interviews, supplier quotes, samples, paid data,
  tender exports, financing quotes, insurance advice, customs advice, legal
  advice, lab testing, factory audits, or certification review.
- A section headed "Decision Gates" covering exact evidence, spend, travel,
  supplier, financing, and bid gates.
- A statement explaining which ranking differences are not meaningful.

Do not approve an opportunity merely because the workflow expects a winner.

End the output with exactly:
<!-- COMPLETE:RISK_AUDIT -->
""",
    expected_output=(
        "A complete conservative audit with contract-verification checks, score "
        "ranges, bid-access and competition challenge, financing and downside stress "
        "tests, evidence gaps, rejection log, gates, and the completion marker."
    ),
    agent=critic,
    context=[opportunity_map_task, validation_synthesis_task],
    markdown=True,
    output_file=crewai_output_path("03_risk_scorecard.md"),
    guardrail=require_complete_output(
        "<!-- COMPLETE:RISK_AUDIT -->",
        6500,
        4,
    ),
    guardrail_max_retries=1,
)

final_report_task = Task(
    name="final_report",
    description=f"""
{shared_research_context}

TASK: FINAL DECISION REPORT
---------------------------
Create a complete decision report from the opportunity map, validation comparison,
and risk audit. Do not perform new web research. Preserve source links, evidence
classifications, important contradictions, and critic uncertainty. Do not merely
copy the intermediate reports.

Use these exact section headings:

## 1. Executive Decision Summary
## 2. Evidence Quality and Contract Data Coverage
## 3. Ranked Contract-Supply Verticals
## 4. Five Real Contract Reconstructions
## 5. Detailed Opportunity Profiles
## 6. Recommended Entry Thesis and Transaction Structure
## 7. Buyer Access, Pipeline, and Bid Strategy
## 8. Supplier Qualification and Quality-Control Plan
## 9. Contract Economics, Financing, and Cash Exposure
## 10. 30/60/90-Day Validation Plan
## 11. Proceed, Revise, Bid, and Stop Gates
## 12. Open Questions, Rejection Log, and Next Data Run

Required content:
- Top three opportunities or a no-go conclusion.
- Clear distinction between validation and launch approval.
- A five-contract reconstruction table with buyer, identifier, dates, status,
  value type, quantity, term, winner, source, and unresolved fields.
- Product specification, buyer problem, procurement route, qualification,
  transaction structure, supply approach, quality system, revenue logic, founder
  advantage, competition, switching/admission wedge, risks, and confidence.
- A realistic path to approximately CAD 200,000 annual gross profit and CAD
  CAD 100,000 annual founder income, without treating contract ceilings as revenue.
- A pipeline model showing assumed win rate, opportunities required, sales cycle,
  contract concentration, and relationship-building workload.
- A 30/60/90-day plan covering procurement-data analysis, buyer interviews,
  partner and incumbent mapping, supplier RFQs, samples, lab/inspection work,
  financing conversations, and proof before a bid or production commitment.
- A concrete specification-to-delivery quality plan.
- Comparison of importer/distributor, manufacturer's representative,
  procurement-integrator, joint-bid/subcontract, and direct-ship structures.
- Exact gates before spending CAD 500, CAD 1,000, CAD 5,000, and CAD 15,000,
  and before supplier travel, bid submission, contract signature, customer credit,
  production deposit, shipment, and inventory.
- Proceed, revise, bid/no-bid, and stop criteria.
- Open questions requiring human judgment or professional advice.
- Complete rejection log and real-world data-acquisition plan.
- A focused recommendation for a manually approved next data or research run,
  if justified.

Do not imply that desk research validates customer demand. Finish all twelve
sections before writing the marker.

End the output with exactly:
<!-- COMPLETE:FINAL_REPORT -->
""",
    expected_output=(
        "A complete twelve-section Markdown contract-supply decision report with "
        "five real contract reconstructions, bid access, quality, financing, "
        "gross-profit and founder-income economics, validation plan, gates, "
        "rejection log, data plan, and the completion marker."
    ),
    agent=strategist,
    context=[opportunity_map_task, validation_synthesis_task, risk_task],
    tools=[],
    markdown=True,
    output_file=crewai_output_path("business_research_report.md"),
    guardrail=require_complete_output(
        "<!-- COMPLETE:FINAL_REPORT -->",
        10000,
        8,
    ),
    guardrail_max_retries=1,
)


# ============================================================
# 9. Sequential crew
# ============================================================

EXECUTION_LOG_PATH = OUTPUT_DIR / "crewai_execution_log.json"

research_crew = Crew(
    agents=[scout, researcher, critic, strategist],
    tasks=[
        opportunity_map_task,
        validation_candidate_1_task,
        validation_candidate_2_task,
        validation_candidate_3_task,
        validation_synthesis_task,
        risk_task,
        final_report_task,
    ],
    process=Process.sequential,
    planning=False,
    memory=False,
    cache=ENABLE_CACHE,
    tracing=False,
    share_crew=False,
    verbose=VERBOSE,
    max_rpm=MAX_RPM,
    output_log_file=str(EXECUTION_LOG_PATH),
    # CrewAI 1.15.x checkpoint snapshots can serialize LLM credentials and full
    # prompts. Keep checkpointing off; output markers and run audits provide the
    # resumability/diagnostic boundary without persisting secrets.
    checkpoint=CHECKPOINTING_ENABLED,
)


# ============================================================
# 10. Deterministic output validation
# ============================================================

REQUIRED_OUTPUTS: dict[str, dict[str, Any]] = {
    "01_opportunity_map.md": {
        "marker": "<!-- COMPLETE:OPPORTUNITY_MAP -->",
        "minimum_characters": 5500,
        "minimum_source_urls": 6,
        "required_fragments": [
            "shortlist",
            "preliminary score",
            "procurement",
            "contract",
            "deferred",
        ],
    },
    "02a_validation_candidate_1.md": {
        "marker": "<!-- COMPLETE:VALIDATION_1 -->",
        "minimum_characters": 5000,
        "minimum_source_urls": 4,
        "required_fragments": [
            "contract reconstruction",
            "qualification",
            "quality",
            "CAD 200,000",
            "CAD 100,000",
            "source",
        ],
    },
    "02b_validation_candidate_2.md": {
        "marker": "<!-- COMPLETE:VALIDATION_2 -->",
        "minimum_characters": 5000,
        "minimum_source_urls": 4,
        "required_fragments": [
            "contract reconstruction",
            "qualification",
            "quality",
            "CAD 200,000",
            "CAD 100,000",
            "source",
        ],
    },
    "02c_validation_candidate_3.md": {
        "marker": "<!-- COMPLETE:VALIDATION_3 -->",
        "minimum_characters": 5000,
        "minimum_source_urls": 4,
        "required_fragments": [
            "contract reconstruction",
            "qualification",
            "quality",
            "CAD 200,000",
            "CAD 100,000",
            "source",
        ],
    },
    "02_validation_report.md": {
        "marker": "<!-- COMPLETE:VALIDATION -->",
        "minimum_characters": 7500,
        "minimum_source_urls": 6,
        "required_fragments": [
            "five-contract",
            "incumbent",
            "qualification",
            "CAD 200,000",
            "CAD 100,000",
            "downside",
            "cash",
            "supplier",
        ],
    },
    "03_risk_scorecard.md": {
        "marker": "<!-- COMPLETE:RISK_AUDIT -->",
        "minimum_characters": 6500,
        "minimum_source_urls": 4,
        "required_fragments": [
            "rejection log",
            "decision gate",
            "bid/no-bid",
            "working capital",
            "score range",
        ],
    },
    "business_research_report.md": {
        "marker": "<!-- COMPLETE:FINAL_REPORT -->",
        "minimum_characters": 10000,
        "minimum_source_urls": 8,
        "required_fragments": [
            "## 1. Executive Decision Summary",
            "## 2. Evidence Quality and Contract Data Coverage",
            "## 3. Ranked Contract-Supply Verticals",
            "## 4. Five Real Contract Reconstructions",
            "## 5. Detailed Opportunity Profiles",
            "## 6. Recommended Entry Thesis and Transaction Structure",
            "## 7. Buyer Access, Pipeline, and Bid Strategy",
            "## 8. Supplier Qualification and Quality-Control Plan",
            "## 9. Contract Economics, Financing, and Cash Exposure",
            "## 10. 30/60/90-Day Validation Plan",
            "## 11. Proceed, Revise, Bid, and Stop Gates",
            "## 12. Open Questions, Rejection Log, and Next Data Run",
            "CAD 500",
            "CAD 1,000",
            "CAD 5,000",
            "CAD 15,000",
            "CAD 200,000",
            "CAD 100,000",
        ],
    },
}


def serialize_usage_metrics(metrics: Any) -> Any:
    """Convert CrewAI usage metrics into JSON-safe data."""
    if metrics is None:
        return None
    if hasattr(metrics, "model_dump"):
        return metrics.model_dump()
    if hasattr(metrics, "dict"):
        return metrics.dict()
    return str(metrics)


def output_file_metadata() -> dict[str, Any]:
    """Return existence and size information for required output files."""
    metadata: dict[str, Any] = {}

    for filename in REQUIRED_OUTPUTS:
        path = OUTPUT_DIR / filename
        metadata[filename] = {
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }

    return metadata


def archive_existing_outputs() -> Path | None:
    """Move stale generated files aside so they cannot mask a failed new run."""
    candidate_paths = [
        *(OUTPUT_DIR / filename for filename in REQUIRED_OUTPUTS),
        OUTPUT_DIR / "run_audit.json",
        OUTPUT_DIR / "incomplete_run.json",
        OUTPUT_DIR / "last_error.txt",
        EXECUTION_LOG_PATH,
    ]

    existing = [path for path in candidate_paths if path.exists()]
    if not existing:
        return None

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive_dir = OUTPUT_DIR / "archive" / timestamp
    archive_dir.mkdir(parents=True, exist_ok=True)

    for source in existing:
        destination = archive_dir / source.name
        shutil.move(str(source), str(destination))

    return archive_dir


def build_audit(
    *,
    status: str,
    metrics: Any,
    completion_errors: list[str] | None = None,
    exception: Exception | None = None,
    archived_outputs: Path | None = None,
) -> dict[str, Any]:
    """Build a detailed, JSON-safe run audit."""
    return {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "run_mode": RUN_MODE,
        "process": "sequential",
        "context_path": str(CONTEXT_PATH),
        "output_dir": str(OUTPUT_DIR),
        "crewai_storage_dir": str(CREWAI_STORAGE_DIR),
        "evidence_input_dir": str(EVIDENCE_INPUT_DIR),
        "evidence_pack_max_characters": EVIDENCE_PACK_MAX_CHARACTERS,
        "evidence_pack_manifest": evidence_pack_manifest,
        "archived_previous_outputs": (str(archived_outputs) if archived_outputs else None),
        "worker_model": WORKER_MODEL,
        "final_model": FINAL_MODEL,
        "planning_enabled": False,
        "memory_enabled": False,
        "delegation_enabled": False,
        "automatic_full_reruns": False,
        "target_run_budget_usd": TARGET_RUN_BUDGET_USD,
        "max_rpm": MAX_RPM,
        "max_execution_seconds_per_agent": MAX_EXECUTION_SECONDS,
        "openai_max_retries": OPENAI_MAX_RETRIES,
        "agent_limits": {
            "scout": {
                "max_iter": SCOUT_MAX_ITER,
                "max_completion_tokens": SCOUT_MAX_COMPLETION_TOKENS,
                "prompt_search_limit": SCOUT_SEARCH_LIMIT,
            },
            "researcher": {
                "max_iter": RESEARCHER_MAX_ITER,
                "max_completion_tokens_per_task": RESEARCHER_MAX_COMPLETION_TOKENS,
                "prompt_search_limit_per_candidate": RESEARCHER_SEARCH_LIMIT,
                "prompt_scrape_limit_per_candidate": RESEARCHER_SCRAPE_LIMIT,
            },
            "critic": {
                "max_iter": CRITIC_MAX_ITER,
                "max_completion_tokens": CRITIC_MAX_COMPLETION_TOKENS,
                "prompt_search_limit": CRITIC_SEARCH_LIMIT,
            },
            "strategist": {
                "max_iter": STRATEGIST_MAX_ITER,
                "max_completion_tokens": FINAL_MAX_COMPLETION_TOKENS,
                "search_limit": 0,
            },
        },
        "search_results_per_query": SEARCH_RESULTS_PER_QUERY,
        "tool_max_completion_tokens": TOOL_MAX_COMPLETION_TOKENS,
        "checkpointing_enabled": CHECKPOINTING_ENABLED,
        "cache_enabled": ENABLE_CACHE,
        "usage_metrics": serialize_usage_metrics(metrics),
        "completion_errors": completion_errors or [],
        "output_files": output_file_metadata(),
        "execution_log_path": str(EXECUTION_LOG_PATH),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "crewai": package_version("crewai"),
            "crewai-tools": package_version("crewai-tools"),
            "openai": package_version("openai"),
        },
        "exception": (
            {
                "type": type(exception).__name__,
                "message": str(exception),
            }
            if exception
            else None
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write formatted JSON using UTF-8."""
    path.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )


# ============================================================
# 11. Run
# ============================================================


def main() -> int:
    print("=" * 76)
    print("BUDGET-CONTROLLED CONTRACT-SUPPLY RESEARCH")
    print("=" * 76)
    print(f"Run mode: {RUN_MODE}")
    print("Process: sequential")
    print(f"Target planning budget: USD {TARGET_RUN_BUDGET_USD:.2f}")
    print(f"Worker model: {WORKER_MODEL}")
    print(f"Final synthesis model: {FINAL_MODEL}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Evidence input directory: {EVIDENCE_INPUT_DIR}")
    populated_csv_rows = sum(int(item.get("data_rows") or 0) for item in evidence_pack_manifest)
    print(f"Local structured evidence rows: {populated_csv_rows}")
    if populated_csv_rows == 0:
        print(
            "Evidence warning: CSV templates contain no data rows; the run will "
            "depend on bounded web research and must report missing bulk data."
        )
    print("Planning: disabled")
    print("Memory: disabled")
    print("Delegation: disabled")
    print("Automatic full reruns: disabled")
    print(
        "Search limits: "
        f"Scout={SCOUT_SEARCH_LIMIT}, "
        f"Researcher={RESEARCHER_SEARCH_LIMIT} per candidate, "
        f"Scrapes={RESEARCHER_SCRAPE_LIMIT} per candidate, "
        f"Critic={CRITIC_SEARCH_LIMIT}"
    )
    print("Important: the budget value is a planning guardrail, not a provider-side spending cap.")
    print("=" * 76)

    archived_outputs = archive_existing_outputs()
    if archived_outputs:
        print(f"Archived previous generated outputs to: {archived_outputs}")

    try:
        result = research_crew.kickoff()
    except KeyboardInterrupt:
        print("\nExecution cancelled by user.")
        return 130
    except Exception as exc:
        error_path = OUTPUT_DIR / "last_error.txt"
        error_path.write_text(
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            encoding="utf-8",
        )

        metrics = getattr(research_crew, "usage_metrics", None)
        audit = build_audit(
            status="error",
            metrics=metrics,
            exception=exc,
            archived_outputs=archived_outputs,
        )
        write_json(OUTPUT_DIR / "run_audit.json", audit)

        print(f"\nExecution failed: {type(exc).__name__}: {exc}")
        print(f"Traceback recorded at: {error_path}")
        print(f"Audit recorded at: {OUTPUT_DIR / 'run_audit.json'}")
        return 1

    # Defensive fallback for CrewAI versions that do not write the final output file.
    final_report_path = OUTPUT_DIR / "business_research_report.md"
    if not final_report_path.exists() and getattr(result, "raw", None):
        final_report_path.write_text(result.raw, encoding="utf-8")

    metrics = getattr(research_crew, "usage_metrics", None)
    completion_errors = validate_output_files(OUTPUT_DIR, REQUIRED_OUTPUTS)

    if completion_errors:
        audit = build_audit(
            status="incomplete",
            metrics=metrics,
            completion_errors=completion_errors,
            archived_outputs=archived_outputs,
        )
        write_json(OUTPUT_DIR / "run_audit.json", audit)
        write_json(
            OUTPUT_DIR / "incomplete_run.json",
            {
                "status": "incomplete",
                "errors": completion_errors,
                "instruction": (
                    "Do not use revision mode with these outputs. Review the execution "
                    "log, adjust only the failed stage, and rerun discovery mode."
                ),
            },
        )

        print("\n" + "=" * 76)
        print("RESEARCH RUN INCOMPLETE")
        print("=" * 76)
        for error in completion_errors:
            print(f"- {error}")
        print(f"Audit: {OUTPUT_DIR / 'run_audit.json'}")
        print(f"Details: {OUTPUT_DIR / 'incomplete_run.json'}")
        print(f"Execution log: {EXECUTION_LOG_PATH}")
        return 2

    audit = build_audit(
        status="complete",
        metrics=metrics,
        archived_outputs=archived_outputs,
    )
    write_json(OUTPUT_DIR / "run_audit.json", audit)

    print("\n" + "=" * 76)
    print("RESEARCH COMPLETE AND VALIDATED")
    print("=" * 76)
    print(f"Opportunity map: {OUTPUT_DIR / '01_opportunity_map.md'}")
    print(f"Candidate 1 validation: {OUTPUT_DIR / '02a_validation_candidate_1.md'}")
    print(f"Candidate 2 validation: {OUTPUT_DIR / '02b_validation_candidate_2.md'}")
    print(f"Candidate 3 validation: {OUTPUT_DIR / '02c_validation_candidate_3.md'}")
    print(f"Validation comparison: {OUTPUT_DIR / '02_validation_report.md'}")
    print(f"Risk scorecard: {OUTPUT_DIR / '03_risk_scorecard.md'}")
    print(f"Final report: {final_report_path}")
    print(f"Audit: {OUTPUT_DIR / 'run_audit.json'}")
    print(f"Execution log: {EXECUTION_LOG_PATH}")

    serialized_metrics = serialize_usage_metrics(metrics)
    if serialized_metrics:
        print("\nCrewAI usage metrics:")
        print(json.dumps(serialized_metrics, indent=2, default=str))

    print("\nReview the completed report before manually approving any revision run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
