from __future__ import annotations

from pathlib import Path

from model_counsel.procurement import (
    dedup_key,
    normalize_contract_row,
    parse_date,
    parse_money,
    stable_id,
)


def test_parse_helpers() -> None:
    assert parse_date("2025-03-14").isoformat() == "2025-03-14"
    assert parse_date("not-a-date") is None
    assert parse_money("CAD $1,234.50") == 1234.50
    assert parse_money("") is None
    assert stable_id("a", "b") == stable_id("a", "b")


def test_contract_normalization_and_deduplication() -> None:
    row = {
        "procurementCategory-categorieApprovisionnement": "Goods",
        "publicationDate-datePublication": "2025-03-14",
        "totalContractValue-valeurTotaleContrat": "125,000.00",
        "contractCurrency-contratMonnaie": "CAD",
        "title-titre-eng": "Synthetic container supply",
        "contractNumber-numeroContrat": "SYN-100",
        "supplierStandardizedName-nomNormaliseFournisseur-eng": "Example Supplier",
        "contractingEntityName-nomEntitContractante-eng": "Fictional Buyer",
        "contractAwardDate-dateAttributionContrat": "2025-03-10",
    }
    normalized = normalize_contract_row(Path("synthetic.csv"), row, "contract", {})
    assert normalized is not None
    assert normalized["normalized_value_cad"] == "125000.00"
    assert normalized["winner"] == "Example Supplier"
    assert normalized["value_type"] == "total_contract_value"
    assert dedup_key(normalized) == "contract|syn-100"
