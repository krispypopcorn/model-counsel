"""Normalize recent CanadaBuys procurement files into an agent-ready evidence pack.

The script preserves a full filtered event file under research_inputs/processed
and writes a smaller, stratified contract_evidence.csv for CrewAI context.
It uses only the Python standard library so it can run in the project venv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import statistics
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "research_inputs" / "raw"
PROCESSED_DIR = ROOT / "research_inputs" / "processed"
EVIDENCE_PATH = ROOT / "research_inputs" / "contract_evidence.csv"
SUMMARY_PATH = PROCESSED_DIR / "canadabuys_category_summary.csv"
AGENT_SUMMARY_PATH = ROOT / "research_inputs" / "category_summary.csv"
FULL_EVENTS_PATH = PROCESSED_DIR / "canadabuys_contract_events_normalized.csv"
TENDER_EVENTS_PATH = PROCESSED_DIR / "canadabuys_tenders_normalized.csv"
RUN_SUMMARY_PATH = PROCESSED_DIR / "canadabuys_preparation_summary.md"

OUTPUT_FIELDS = [
    "record_id",
    "source_platform",
    "source_url",
    "procurement_identifier",
    "buyer",
    "buyer_type",
    "jurisdiction",
    "title",
    "product_family",
    "status",
    "notice_date",
    "close_date",
    "award_date",
    "start_date",
    "end_date",
    "value_type",
    "original_value",
    "original_currency",
    "normalized_value_cad",
    "estimated_quantity",
    "unit",
    "winner",
    "other_bidders",
    "contract_vehicle",
    "mandatory_requirements",
    "specification_summary",
    "delivery_requirements",
    "payment_terms",
    "bid_security",
    "performance_security",
    "incumbent_evidence",
    "last_verified_date",
    "analyst_notes",
]

TARGET_KEYWORDS = (
    "textile",
    "linen",
    "bedding",
    "sheet",
    "towel",
    "blanket",
    "garment",
    "uniform",
    "furniture",
    "fixture",
    "packaging",
    "container",
    "pallet",
    "cart",
    "shelving",
    "storage",
    "material handling",
    "laundry",
    "housekeeping",
    "mattress",
    "food service",
    "foodservice",
    "reusable",
)

EXCLUDED_FAMILY_KEYWORDS = (
    "software",
    "database",
    "subscription",
    "electronic publication",
    "information retrieval",
    "fuel",
    "gasoline",
    "diesel",
    "ammunition",
    "cartridge and propellant",
    "weapon",
    "guns or pistols",
    "defense or law enforcement",
    "pharmaceutical",
    "drugs and pharmaceutical",
    "explosive",
    "marine craft",
    "water transport vessel",
    "boat",
    "vessel",
    "motor vehicle",
    "busses",
    "automobiles or cars",
    "aircraft",
    "automatic data processing",
    "adp input-output",
    "computer",
    "information technology",
    "telecommunications",
    "printer",
    "ink cartridge",
    "storage devices",
    "edp",
    "banquet and catering services",
    "decommissioning services",
    "maintenance services",
    "repair services",
    "technical support",
    "support for",
    "netapp",
    "storagegrid",
)


def clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def first(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = clean(row.get(key))
        if value:
            return value
    return ""


def parse_date(value: str) -> date | None:
    value = clean(value)
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value[:19], fmt).date()
        except ValueError:
            continue
    return None


def parse_money(value: str) -> float | None:
    value = clean(value)
    if not value:
        return None
    normalized = re.sub(r"[^0-9.\-]", "", value.replace(",", ""))
    if normalized in {"", "-", ".", "-."}:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def stable_id(*parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return digest


def iter_csv_files(pattern: str) -> Iterable[tuple[Path, dict[str, str]]]:
    for path in sorted(RAW_DIR.glob(pattern)):
        with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield path, row


def english_category(row: dict[str, str]) -> str:
    return first(
        row,
        "procurementCategory-categorieApprovisionnement",
        "procurementCategory-categorieApprovisionnement-eng",
    )


def is_goods(category: str) -> bool:
    normalized = category.casefold()
    codes = {token.strip().upper() for token in re.split(r"[\s,;|]+", category) if token.strip()}
    return "goods" in normalized or "biens" in normalized or "*GD" in codes or "GD" in codes


def event_date(row: dict[str, str]) -> date | None:
    for key in (
        "publicationDate-datePublication",
        "amendmentDate-dateModification",
        "contractAwardDate-dateAttributionContrat",
    ):
        parsed = parse_date(row.get(key, ""))
        if parsed:
            return parsed
    return None


def tender_date(row: dict[str, str]) -> date | None:
    return parse_date(first(row, "publicationDate-datePublication"))


def tender_keys(row: dict[str, str]) -> set[str]:
    return {
        clean(value).casefold()
        for value in (
            row.get("solicitationNumber-numeroSollicitation"),
            row.get("referenceNumber-numeroReference"),
        )
        if clean(value)
    }


def load_tenders(
    cutoff: date,
    as_of: date,
) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    index: dict[str, dict[str, str]] = {}
    normalized: list[dict[str, str]] = []

    for path, row in iter_csv_files("canadabuys_tenders_*.csv"):
        observed = tender_date(row)
        if not observed or observed < cutoff or observed > as_of:
            continue
        category = english_category(row)
        if not is_goods(category):
            continue

        title = first(row, "title-titre-eng", "title-titre-fra")
        solicitation = first(row, "solicitationNumber-numeroSollicitation")
        reference = first(row, "referenceNumber-numeroReference")
        notice_url = first(row, "noticeURL-URLavis-eng", "noticeURL-URLavis-fra")
        description = first(
            row,
            "tenderDescription-descriptionAppelOffres-eng",
            "tenderDescription-descriptionAppelOffres-fra",
        )
        product_family = first(
            row,
            "unspscDescription-eng",
            "gsinDescription-nibsDescription-eng",
            "unspsc",
            "gsin-nibs",
        )
        status = first(
            row,
            "tenderStatus-appelOffresStatut-eng",
            "noticeType-avisType-eng",
        )
        buyer = first(
            row,
            "contractingEntityName-nomEntitContractante-eng",
            "endUserEntitiesName-nomEntitesUtilisateurFinal-eng",
        )
        requirements = "; ".join(
            value
            for value in (
                first(row, "procurementMethod-methodeApprovisionnement-eng"),
                first(row, "selectionCriteria-criteresSelection-eng"),
                first(row, "limitedTenderingReason-raisonAppelOffresLimite-eng"),
                first(row, "tradeAgreements-accordsCommerciaux-eng"),
            )
            if value
        )

        normalized_row = {
            "record_id": "canadabuys_tender_"
            + stable_id(reference, solicitation, title, observed.isoformat()),
            "source_platform": "CanadaBuys tender notices",
            "source_url": notice_url
            or "https://canadabuys.canada.ca/en/procurement-and-contracting-data",
            "procurement_identifier": solicitation or reference,
            "buyer": buyer,
            "buyer_type": "Canadian public-sector procurement entity",
            "jurisdiction": first(
                row,
                "contractingEntityAddressProvince-entiteContractanteAdresseProvince-eng",
                "regionsOfDelivery-regionsLivraison-eng",
            ),
            "title": title,
            "product_family": product_family,
            "status": status,
            "notice_date": observed.isoformat(),
            "close_date": first(row, "tenderClosingDate-appelOffresDateCloture"),
            "award_date": "",
            "start_date": first(row, "expectedContractStartDate-dateDebutContratPrevue"),
            "end_date": first(row, "expectedContractEndDate-dateFinContratPrevue"),
            "value_type": "unknown_tender_value",
            "original_value": "",
            "original_currency": "",
            "normalized_value_cad": "",
            "estimated_quantity": "",
            "unit": "",
            "winner": "",
            "other_bidders": "",
            "contract_vehicle": first(row, "noticeType-avisType-eng"),
            "mandatory_requirements": requirements[:1000],
            "specification_summary": clean(f"{title}; {product_family}; {description}")[:1600],
            "delivery_requirements": first(
                row,
                "regionsOfDelivery-regionsLivraison-eng",
            ),
            "payment_terms": "",
            "bid_security": "",
            "performance_security": "",
            "incumbent_evidence": "",
            "last_verified_date": date.today().isoformat(),
            "analyst_notes": (
                f"Normalized from {path.name}; tender values and committed "
                "quantities are not supplied in the bulk row."
            ),
        }
        normalized.append(normalized_row)

        for key in tender_keys(row):
            existing = index.get(key)
            if not existing or normalized_row["notice_date"] >= existing["notice_date"]:
                index[key] = normalized_row

    return index, normalized


def normalize_contract_row(
    path: Path,
    row: dict[str, str],
    source_type: str,
    tender_index: dict[str, dict[str, str]],
) -> dict[str, str] | None:
    category = english_category(row)
    if not is_goods(category):
        return None

    observed = event_date(row)
    if not observed:
        return None

    currency = first(row, "contractCurrency-contratMonnaie") or "CAD"
    amount = parse_money(first(row, "totalContractValue-valeurTotaleContrat"))
    value_type = "total_contract_value"
    if amount is None or amount <= 0:
        amount = parse_money(first(row, "contractAmount-montantContrat"))
        value_type = "contract_amount"
    if amount is None or amount <= 0:
        return None

    title = first(row, "title-titre-eng", "title-titre-fra")
    solicitation = first(row, "solicitationNumber-numeroSollicitation")
    reference = first(row, "referenceNumber-numeroReference")
    contract_number = first(row, "contractNumber-numeroContrat")
    supplier = first(
        row,
        "supplierStandardizedName-nomNormaliseFournisseur-eng",
        "supplierLegalName-nomLegalFournisseur-eng",
        "supplierOperatingName-nomCommercialFournisseur-eng",
    )

    tender: dict[str, str] | None = None
    for key in (solicitation, reference):
        if clean(key).casefold() in tender_index:
            tender = tender_index[clean(key).casefold()]
            break

    product_family = first(
        row,
        "unspscDescription-eng",
        "gsinDescription-nibsDescription-eng",
        "unspsc",
        "gsin-nibs",
    )
    description = first(
        row,
        "awardDescription-descriptionAttribution-eng",
        "tenderDescription-descriptionAppelOffres-eng",
        "awardDescription-descriptionAttribution-fra",
        "tenderDescription-descriptionAppelOffres-fra",
    )
    requirements = "; ".join(
        value
        for value in (
            first(row, "procurementMethod-methodeApprovisionnement-eng"),
            first(row, "selectionCriteria-criteresSelection-eng"),
            first(row, "limitedTenderingReason-raisonAppelOffresLimite-eng"),
            first(row, "tradeAgreements-accordsCommerciaux-eng"),
        )
        if value
    )
    status = first(
        row,
        "contractStatus-statutContrat-eng",
        "awardStatus-attributionStatut-eng",
        "instrumentType-typeInstrument-eng",
    )
    amendment_number = first(row, "amendmentNumber-numeroModification")
    award_date = first(row, "contractAwardDate-dateAttributionContrat")
    parsed_award_date = parse_date(award_date)
    future_award_flag = bool(parsed_award_date and parsed_award_date > date.today())
    source_url = (
        tender["source_url"]
        if tender
        else "https://canadabuys.canada.ca/en/procurement-and-contracting-data"
    )
    record_key = contract_number or reference or solicitation

    return {
        "record_id": "canadabuys_"
        + source_type
        + "_"
        + stable_id(record_key, supplier, str(amount), observed.isoformat()),
        "source_platform": (
            "CanadaBuys contract history"
            if source_type == "contract"
            else "CanadaBuys award notices"
        ),
        "source_url": source_url,
        "procurement_identifier": contract_number or solicitation or reference,
        "buyer": first(
            row,
            "contractingEntityName-nomEntitContractante-eng",
            "endUserEntitiesName-nomEntitesUtilisateurFinal-eng",
        ),
        "buyer_type": "Canadian public-sector procurement entity",
        "jurisdiction": first(
            row,
            "contractingEntityAddressProvince-entiteContractanteAdresseProvince-eng",
            "regionsOfDelivery-regionsLivraison-eng",
        ),
        "title": title,
        "product_family": product_family,
        "status": status,
        "notice_date": first(row, "publicationDate-datePublication"),
        "close_date": tender["close_date"] if tender else "",
        "award_date": award_date,
        "start_date": first(row, "contractStartDate-contratDateDebut"),
        "end_date": first(row, "contractEndDate-dateFinContrat"),
        "value_type": value_type,
        "original_value": f"{amount:.2f}",
        "original_currency": currency,
        "normalized_value_cad": f"{amount:.2f}" if currency.casefold() == "cad" else "",
        "estimated_quantity": "",
        "unit": "",
        "winner": supplier,
        "other_bidders": "",
        "contract_vehicle": first(row, "instrumentType-typeInstrument-eng"),
        "mandatory_requirements": requirements[:1000],
        "specification_summary": clean(f"{title}; {product_family}; {description}")[:1600],
        "delivery_requirements": first(
            row,
            "regionsOfDelivery-regionsLivraison-eng",
        ),
        "payment_terms": "",
        "bid_security": "",
        "performance_security": "",
        "incumbent_evidence": (f"Awarded supplier in this record: {supplier}" if supplier else ""),
        "last_verified_date": date.today().isoformat(),
        "analyst_notes": (
            f"Normalized from {path.name}; amendment={amendment_number or 'none'}; "
            f"joined_to_tender={'yes' if tender else 'no'}; "
            f"future_award_date_flag={'yes' if future_award_flag else 'no'}. "
            "Bulk values require "
            "verification against the underlying notice before bid decisions."
        ),
        "_observed_date": observed.isoformat(),
        "_contract_number": contract_number,
        "_solicitation": solicitation,
        "_reference": reference,
        "_source_type": source_type,
    }


def dedup_key(row: dict[str, str]) -> str:
    if row["_contract_number"]:
        return "contract|" + row["_contract_number"].casefold()
    return "|".join(
        (
            row["_solicitation"].casefold(),
            row["winner"].casefold(),
            row["title"].casefold(),
        )
    )


def award_is_in_window(row: dict[str, str], cutoff: date, as_of: date) -> bool:
    """Use award date for demand analysis; fall back only when it is absent."""
    awarded = parse_date(row["award_date"])
    if awarded:
        return cutoff <= awarded <= as_of
    observed = parse_date(row["_observed_date"])
    return bool(observed and cutoff <= observed <= as_of)


def write_rows(path: Path, rows: Iterable[dict[str, str]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for raw in rows:
            writer.writerow({field: raw.get(field, "") for field in OUTPUT_FIELDS})
            count += 1
    return count


def family_key(row: dict[str, str]) -> str:
    family = clean(row["product_family"]).casefold()
    if family:
        return family[:180]
    return clean(row["title"]).casefold()[:180]


def keyword_match(row: dict[str, str]) -> bool:
    haystack = " ".join(
        (
            row["title"],
            row["product_family"],
        )
    ).casefold()
    return any(keyword in haystack for keyword in TARGET_KEYWORDS)


def excluded_family(row: dict[str, str]) -> bool:
    haystack = " ".join(
        (
            row["title"],
            row["product_family"],
            row["specification_summary"],
        )
    ).casefold()
    return any(keyword in haystack for keyword in EXCLUDED_FAMILY_KEYWORDS)


def build_curated_rows(
    contracts: list[dict[str, str]],
    tenders: list[dict[str, str]],
    *,
    entry_min: float,
    validation_max: float,
    reference_min: float,
    reference_max: float,
    scale_min: float,
    max_value: float,
    max_rows: int,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    eligible = [
        row
        for row in contracts
        if row["normalized_value_cad"]
        and entry_min <= float(row["normalized_value_cad"]) <= max_value
        and not excluded_family(row)
    ]
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in eligible:
        groups[family_key(row)].append(row)

    summaries: list[dict[str, Any]] = []
    for family, rows in groups.items():
        values = [float(row["normalized_value_cad"]) for row in rows]
        buyers = {row["buyer"].casefold() for row in rows if row["buyer"]}
        winners = {row["winner"].casefold() for row in rows if row["winner"]}
        validation_count = sum(entry_min <= value <= validation_max for value in values)
        reference_count = sum(reference_min <= value <= reference_max for value in values)
        scale_count = sum(value >= scale_min for value in values)
        summaries.append(
            {
                "product_family": family,
                "event_count": len(rows),
                "validation_event_count": validation_count,
                "reference_event_count": reference_count,
                "scale_event_count": scale_count,
                "complete_ladder_signal": int(
                    validation_count > 0 and reference_count > 0 and scale_count > 0
                ),
                "distinct_buyers": len(buyers),
                "distinct_winners": len(winners),
                "total_value_cad": round(sum(values), 2),
                "median_value_cad": round(statistics.median(values), 2),
                "max_value_cad": round(max(values), 2),
                "keyword_priority": int(any(keyword in family for keyword in TARGET_KEYWORDS)),
            }
        )

    summaries.sort(
        key=lambda item: (
            item["complete_ladder_signal"],
            item["scale_event_count"],
            item["distinct_buyers"],
            item["event_count"],
            item["keyword_priority"],
            item["total_value_cad"],
        ),
        reverse=True,
    )

    selected: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(row: dict[str, str]) -> None:
        if len(selected) >= max_rows or row["record_id"] in seen:
            return
        seen.add(row["record_id"])
        selected.append(row)

    # Put mandate-adjacent physical-product examples first so the bounded agent
    # context is not consumed by unrelated high-volume categories.
    for row in sorted(
        (row for row in eligible if keyword_match(row)),
        key=lambda row: (
            float(row["normalized_value_cad"]) >= scale_min,
            float(row["normalized_value_cad"]),
        ),
        reverse=True,
    ):
        add(row)

    # Preserve the largest evidence for the most repeated remaining families.
    for summary in summaries[:40]:
        family_rows = sorted(
            groups[summary["product_family"]],
            key=lambda row: float(row["normalized_value_cad"]),
            reverse=True,
        )
        scale_rows = [row for row in family_rows if float(row["normalized_value_cad"]) >= scale_min]
        entry_rows = [row for row in family_rows if float(row["normalized_value_cad"]) < scale_min]
        for row in scale_rows[:3]:
            add(row)
        for row in entry_rows[:1]:
            add(row)

    # Include recent matching tenders so the agents can inspect current specifications.
    for row in sorted(tenders, key=lambda item: item["notice_date"], reverse=True):
        if keyword_match(row):
            add(row)
        if len([item for item in selected if item["value_type"] == "unknown_tender_value"]) >= 30:
            break

    return selected[:max_rows], summaries


def write_category_summary(
    path: Path,
    summaries: list[dict[str, Any]],
    *,
    max_rows: int | None = None,
) -> None:
    fields = [
        "product_family",
        "event_count",
        "validation_event_count",
        "reference_event_count",
        "scale_event_count",
        "complete_ladder_signal",
        "distinct_buyers",
        "distinct_winners",
        "total_value_cad",
        "median_value_cad",
        "max_value_cad",
        "keyword_priority",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries if max_rows is None else summaries[:max_rows])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", default="2023-07-26")
    parser.add_argument("--as-of", default=date.today().isoformat())
    parser.add_argument("--entry-min", type=float, default=10000)
    parser.add_argument("--validation-max", type=float, default=50000)
    parser.add_argument("--reference-min", type=float, default=50000)
    parser.add_argument("--reference-max", type=float, default=300000)
    parser.add_argument("--scale-min", type=float, default=250000)
    parser.add_argument("--max-value", type=float, default=2500000)
    parser.add_argument("--max-agent-rows", type=int, default=220)
    args = parser.parse_args()

    cutoff = parse_date(args.cutoff)
    as_of = parse_date(args.as_of)
    if cutoff is None or as_of is None:
        raise ValueError("--cutoff and --as-of must be YYYY-MM-DD")
    if cutoff > as_of:
        raise ValueError("--cutoff cannot be later than --as-of")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tender_index, tenders = load_tenders(cutoff, as_of)

    latest: dict[str, dict[str, str]] = {}
    raw_contract_rows = 0
    raw_award_rows = 0

    for path, source_row in iter_csv_files("canadabuys_contracts_*.csv"):
        normalized = normalize_contract_row(path, source_row, "contract", tender_index)
        if (
            not normalized
            or normalized["_observed_date"] < cutoff.isoformat()
            or normalized["_observed_date"] > as_of.isoformat()
            or not award_is_in_window(normalized, cutoff, as_of)
        ):
            continue
        raw_contract_rows += 1
        key = dedup_key(normalized)
        if key not in latest or normalized["_observed_date"] >= latest[key]["_observed_date"]:
            latest[key] = normalized

    # Award notices broaden coverage beyond PSPC contract history. Contract
    # history wins when the same contract is present in both sources.
    contract_keys = set(latest)
    for path, source_row in iter_csv_files("canadabuys_awards_*.csv"):
        normalized = normalize_contract_row(path, source_row, "award", tender_index)
        if (
            not normalized
            or normalized["_observed_date"] < cutoff.isoformat()
            or normalized["_observed_date"] > as_of.isoformat()
            or not award_is_in_window(normalized, cutoff, as_of)
        ):
            continue
        raw_award_rows += 1
        key = dedup_key(normalized)
        if key in contract_keys:
            continue
        if key not in latest or normalized["_observed_date"] >= latest[key]["_observed_date"]:
            latest[key] = normalized

    contracts = sorted(
        latest.values(),
        key=lambda row: (
            row["_observed_date"],
            float(row["normalized_value_cad"] or 0),
        ),
        reverse=True,
    )
    curated, summaries = build_curated_rows(
        contracts,
        tenders,
        entry_min=args.entry_min,
        validation_max=args.validation_max,
        reference_min=args.reference_min,
        reference_max=args.reference_max,
        scale_min=args.scale_min,
        max_value=args.max_value,
        max_rows=args.max_agent_rows,
    )

    full_count = write_rows(FULL_EVENTS_PATH, contracts)
    tender_count = write_rows(TENDER_EVENTS_PATH, tenders)
    curated_count = write_rows(EVIDENCE_PATH, curated)
    write_category_summary(SUMMARY_PATH, summaries)
    agent_summaries = sorted(
        summaries,
        key=lambda item: (
            item["keyword_priority"],
            item["complete_ladder_signal"],
            item["scale_event_count"],
            item["distinct_buyers"],
            item["event_count"],
            item["total_value_cad"],
        ),
        reverse=True,
    )
    write_category_summary(AGENT_SUMMARY_PATH, agent_summaries, max_rows=100)

    scale_events = sum(
        bool(row["normalized_value_cad"])
        and args.scale_min <= float(row["normalized_value_cad"]) <= args.max_value
        for row in contracts
    )
    entry_events = sum(
        bool(row["normalized_value_cad"])
        and args.entry_min <= float(row["normalized_value_cad"]) < args.scale_min
        for row in contracts
    )

    summary = f"""# CanadaBuys Evidence Preparation Summary

- Cutoff date: {cutoff.isoformat()}
- As-of date: {as_of.isoformat()}
- Raw qualifying goods contract-history rows before deduplication: {raw_contract_rows:,}
- Raw qualifying goods award-notice rows before deduplication: {raw_award_rows:,}
- Deduplicated normalized contract/award records: {full_count:,}
- Normalized goods tender notices: {tender_count:,}
- Entry-ladder records from CAD {args.entry_min:,.0f} to below CAD {args.scale_min:,.0f}: {entry_events:,}
- Scale-potential records from CAD {args.scale_min:,.0f} to CAD {args.max_value:,.0f}: {scale_events:,}
- Product-family summary rows: {len(summaries):,}
- Agent-ready stratified evidence rows: {curated_count:,}

## Important limitations

- CanadaBuys bulk contract values are evidence of purchasing activity, not
  evidence of gross margin or accessibility to a new bidder.
- Contract history and award notices can overlap; the preparation process
  prefers contract history and suppresses matching award rows.
- Amendment handling retains the latest observed record for a contract key; the
  underlying notice must be checked before treating the value as final.
- Bulk rows do not consistently provide bidder counts, quantities, payment
  terms, bid security, performance security, or detailed specifications.
- Source dates are preserved. Award dates later than the preparation date are
  flagged as source-data anomalies and excluded from analytical counts.
- Award/contract records are screened by award date; records for older awards
  merely republished or amended inside the observation window are not counted
  as new demand.
- Non-CAD records retain original value but have no normalized CAD value.
- The active standing-offer endpoint returned HTTP 502 during this preparation
  and is not represented in the normalized row counts.
"""
    RUN_SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
