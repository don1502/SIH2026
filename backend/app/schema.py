"""Shared schema for the cleaned crime-intelligence dataset.

Defines node types, their display names, and the logic to detect an uploaded
CSV's table type from its columns. Used by both the knowledge-graph builder
and the ML inference path so the same relationships are produced everywhere.
"""
from __future__ import annotations

import pandas as pd

# Node type -> (id column, name column, extra property columns)
NODE_TYPES: dict[str, dict] = {
    "person": {"table": "persons", "id": "person_id", "name": "full_name",
               "props": ["age", "gender", "role", "risk_score"], "label": "Person"},
    "organization": {"table": "organizations", "id": "org_id", "name": "org_name",
                     "props": ["type", "threat_level"], "label": "Organization"},
    "location": {"table": "locations", "id": "location_id", "name": "location_id",
                 "props": ["type", "latitude", "longitude"], "label": "Location"},
    "phone": {"table": "phones", "id": "phone_id", "name": "phone_number",
              "props": ["imei", "service_provider"], "label": "Phone"},
    "account": {"table": "accounts", "id": "account_id", "name": "account_number",
                "props": ["bank_name", "balance"], "label": "Account"},
    "vehicle": {"table": "vehicles", "id": "vehicle_id", "name": "registration_number",
                "props": ["vehicle_type", "color"], "label": "Vehicle"},
    "case": {"table": "cases", "id": "case_id", "name": "case_title",
             "props": ["crime_type", "status", "date_registered"], "label": "Case"},
}

# Signature columns used to recognize each cleaned table from an uploaded file.
TABLE_SIGNATURES: dict[str, set[str]] = {
    "persons": {"person_id", "full_name"},
    "organizations": {"org_id", "org_name"},
    "locations": {"location_id", "latitude"},
    "phones": {"phone_id", "phone_number"},
    "accounts": {"account_id", "account_number"},
    "vehicles": {"vehicle_id", "registration_number"},
    "cases": {"case_id", "case_title"},
    "person_phones": {"person_id", "phone_id"},
    "person_accounts": {"person_id", "account_id"},
    "person_vehicles": {"person_id", "vehicle_id"},
    "person_locations": {"person_id", "location_id"},
    "person_organizations": {"person_id", "org_id"},
    "case_persons": {"case_id", "person_id", "involvement_type"},
    "evidence": {"evidence_id", "case_id", "evidence_type"},
    "events": {"event_id", "event_title"},
    "call_records": {"caller_number", "receiver_number"},
    "transactions": {"sender_account", "receiver_account"},
}

THREAT_LEVEL = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
SUSPECT_INVOLVEMENT = {"Prime Suspect", "Accused", "Co-conspirator", "Suspect"}


def detect_table(df: pd.DataFrame) -> str | None:
    """Return the cleaned-table name whose signature columns are all present."""
    cols = set(df.columns)
    best: str | None = None
    best_size = 0
    for table, sig in TABLE_SIGNATURES.items():
        if sig.issubset(cols) and len(sig) > best_size:
            best, best_size = table, len(sig)
    return best


def resolve_phone_ids(call_records: pd.DataFrame, phones: pd.DataFrame) -> pd.DataFrame:
    """Map caller/receiver phone numbers to phone ids -> src_id/dst_id."""
    lookup = dict(zip(phones["phone_number"].astype(str), phones["phone_id"]))
    out = call_records.copy()
    out["src_id"] = out["caller_number"].astype(str).map(lookup)
    out["dst_id"] = out["receiver_number"].astype(str).map(lookup)
    return out.dropna(subset=["src_id", "dst_id"])


def resolve_account_ids(transactions: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    """Map sender/receiver account numbers to account ids -> src_id/dst_id."""
    lookup = dict(zip(accounts["account_number"].astype(str), accounts["account_id"]))
    out = transactions.copy()
    out["src_id"] = out["sender_account"].astype(str).map(lookup)
    out["dst_id"] = out["receiver_account"].astype(str).map(lookup)
    return out.dropna(subset=["src_id", "dst_id"])
