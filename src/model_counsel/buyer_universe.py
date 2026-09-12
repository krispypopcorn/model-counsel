"""Create an agent-sized buyer-universe extract from Statistics Canada table 33-10-1095-01."""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "research_inputs" / "raw" / "statcan_33101095_business_counts_2025" / "33101095.csv"
OUTPUT = ROOT / "research_inputs" / "buyer_universe_summary.csv"

TARGET_NAICS = {
    "311": "Food manufacturing",
    "493": "Warehousing and storage",
    "622": "Hospitals",
    "623": "Nursing and residential care facilities",
    "721111": "Hotels",
    "8123": "Dry cleaning and laundry services",
}

GEOGRAPHIES = {
    "Canada",
    "Ontario",
    "Quebec",
    "British Columbia",
    "Alberta",
}


def naics_code(label: str) -> str:
    match = re.search(r"\[(\d{3,6})\]\s*$", label)
    return match.group(1) if match else ""


def main() -> None:
    source_rows: list[dict[str, str]] = []
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        for source_row in csv.DictReader(handle):
            code = naics_code(source_row["North American Industry Classification System (NAICS)"])
            if code not in TARGET_NAICS or source_row["GEO"] not in GEOGRAPHIES:
                continue
            source_rows.append(source_row | {"_naics_code": code})

    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for source_row in source_rows:
        key = (source_row["GEO"], source_row["_naics_code"])
        grouped.setdefault(key, []).append(source_row)

    rows: list[dict[str, str | int]] = []
    size_floor = {
        "20 to 49 employees": 20,
        "50 to 99 employees": 50,
        "100 to 199 employees": 100,
        "200 to 499 employees": 200,
        "500 plus employees": 500,
    }
    for (geography, code), group in sorted(grouped.items()):
        values = {item["Employment size"]: int(item["VALUE"]) for item in group if item["VALUE"]}

        def at_least(employee_floor: int) -> int:
            return sum(
                values.get(label, 0)
                for label, floor in size_floor.items()
                if floor >= employee_floor
            )

        rows.append(
            {
                "reference_period": group[0]["REF_DATE"],
                "geography": geography,
                "naics_code": code,
                "buyer_industry": TARGET_NAICS[code],
                "establishments_total_with_employees": values.get("Total, with employees", 0),
                "establishments_20_plus_employees": at_least(20),
                "establishments_50_plus_employees": at_least(50),
                "establishments_100_plus_employees": at_least(100),
                "establishments_200_plus_employees": at_least(200),
                "establishments_500_plus_employees": at_least(500),
                "source_table": "Statistics Canada 33-10-1095-01",
                "source_url": ("https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3310109501"),
                "interpretation_warning": (
                    "Statistical locations, not unique buying organizations or spend."
                ),
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
