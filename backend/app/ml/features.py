"""Feature engineering for suspect prediction.

Person-level features are aggregated across the multi-hop graph: a person's
phone-call behaviour flows through their OWNED phones, and their financial
behaviour through their OWNED accounts. This is where the predictive signal
lives, since persons have no direct call/transaction records.

The same function is used for training (full cleaned dataset) and inference
(an uploaded subset of tables), degrading gracefully when tables are missing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.schema import SUSPECT_INVOLVEMENT, THREAT_LEVEL, resolve_account_ids, resolve_phone_ids

NUMERIC_FEATURES = [
    "age", "risk_score",
    "n_phones", "n_accounts", "n_vehicles", "n_orgs", "n_locations", "n_cases",
    "org_threat_max", "org_threat_sum",
    "call_total", "call_duration_total", "call_distinct_contacts",
    "txn_total", "txn_amount_total", "txn_distinct_counterparts", "balance_total",
    "comm_degree",
]
CATEGORICAL_FEATURES = ["gender", "role"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _owner_map(junction: pd.DataFrame | None, id_col: str) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    if junction is None:
        return mapping
    for r in junction.to_dict("records"):
        mapping.setdefault(r[id_col], []).append(r["person_id"])
    return mapping


def build_person_features(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    persons = tables.get("persons")
    if persons is None or persons.empty:
        return pd.DataFrame(columns=["person_id"] + ALL_FEATURES).set_index("person_id")

    feat = pd.DataFrame(index=persons["person_id"].values)
    feat.index.name = "person_id"

    # --- attributes ---
    p = persons.set_index("person_id")
    feat["age"] = _num(p.get("age", pd.Series(index=p.index))).reindex(feat.index)
    feat["risk_score"] = _num(p.get("risk_score", pd.Series(index=p.index))).reindex(feat.index)
    feat["gender"] = p.get("gender", pd.Series("Unknown", index=p.index)).reindex(feat.index).fillna("Unknown")
    feat["role"] = p.get("role", pd.Series("Unknown", index=p.index)).reindex(feat.index).fillna("Unknown")

    # --- ownership / association counts ---
    def count_by_person(table: str) -> pd.Series:
        df = tables.get(table)
        if df is None or df.empty:
            return pd.Series(0, index=feat.index)
        return df.groupby("person_id").size().reindex(feat.index).fillna(0)

    feat["n_phones"] = count_by_person("person_phones")
    feat["n_accounts"] = count_by_person("person_accounts")
    feat["n_vehicles"] = count_by_person("person_vehicles")
    feat["n_orgs"] = count_by_person("person_organizations")
    feat["n_locations"] = count_by_person("person_locations")
    feat["n_cases"] = count_by_person("case_persons")

    # --- organization threat exposure ---
    feat["org_threat_max"] = 0.0
    feat["org_threat_sum"] = 0.0
    po, orgs = tables.get("person_organizations"), tables.get("organizations")
    if po is not None and orgs is not None and not po.empty:
        threat = orgs.assign(t=orgs["threat_level"].map(THREAT_LEVEL).fillna(0))[["org_id", "t"]]
        merged = po.merge(threat, on="org_id", how="left")
        agg = merged.groupby("person_id")["t"].agg(["max", "sum"])
        feat["org_threat_max"] = agg["max"].reindex(feat.index).fillna(0)
        feat["org_threat_sum"] = agg["sum"].reindex(feat.index).fillna(0)

    # --- phone-call behaviour via owned phones ---
    _add_call_features(feat, tables)
    # --- financial behaviour via owned accounts ---
    _add_txn_features(feat, tables)
    # --- person-person communication degree ---
    _add_comm_degree(feat, tables)

    # account balances
    pa, accts = tables.get("person_accounts"), tables.get("accounts")
    feat["balance_total"] = 0.0
    if pa is not None and accts is not None and not pa.empty:
        bal = accts.assign(b=_num(accts.get("balance", 0)))[["account_id", "b"]]
        merged = pa.merge(bal, on="account_id", how="left")
        feat["balance_total"] = merged.groupby("person_id")["b"].sum().reindex(feat.index).fillna(0)

    for col in NUMERIC_FEATURES:
        if col not in feat.columns:
            feat[col] = 0.0
        feat[col] = _num(feat[col]).fillna(0.0)

    return feat[ALL_FEATURES]


def _add_call_features(feat: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    feat["call_total"] = 0.0
    feat["call_duration_total"] = 0.0
    feat["call_distinct_contacts"] = 0.0
    calls, phones = tables.get("call_records"), tables.get("phones")
    pp = tables.get("person_phones")
    if calls is None or phones is None or pp is None or calls.empty:
        return
    resolved = resolve_phone_ids(calls, phones)
    resolved["duration"] = _num(resolved.get("duration_seconds", 0)).fillna(0)
    owner = _owner_map(pp, "phone_id")

    counts: dict[str, int] = {}
    durations: dict[str, float] = {}
    contacts: dict[str, set] = {}
    for r in resolved.to_dict("records"):
        for src_owner in owner.get(r["src_id"], []):
            counts[src_owner] = counts.get(src_owner, 0) + 1
            durations[src_owner] = durations.get(src_owner, 0) + r["duration"]
            contacts.setdefault(src_owner, set()).add(r["dst_id"])
        for dst_owner in owner.get(r["dst_id"], []):
            counts[dst_owner] = counts.get(dst_owner, 0) + 1
            durations[dst_owner] = durations.get(dst_owner, 0) + r["duration"]
            contacts.setdefault(dst_owner, set()).add(r["src_id"])

    feat["call_total"] = pd.Series(counts).reindex(feat.index).fillna(0)
    feat["call_duration_total"] = pd.Series(durations).reindex(feat.index).fillna(0)
    feat["call_distinct_contacts"] = pd.Series({k: len(v) for k, v in contacts.items()}).reindex(feat.index).fillna(0)


def _add_txn_features(feat: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    feat["txn_total"] = 0.0
    feat["txn_amount_total"] = 0.0
    feat["txn_distinct_counterparts"] = 0.0
    txns, accounts = tables.get("transactions"), tables.get("accounts")
    pa = tables.get("person_accounts")
    if txns is None or accounts is None or pa is None or txns.empty:
        return
    resolved = resolve_account_ids(txns, accounts)
    resolved["amt"] = _num(resolved.get("amount", 0)).fillna(0)
    owner = _owner_map(pa, "account_id")

    counts: dict[str, int] = {}
    amounts: dict[str, float] = {}
    counterparts: dict[str, set] = {}
    for r in resolved.to_dict("records"):
        for src_owner in owner.get(r["src_id"], []):
            counts[src_owner] = counts.get(src_owner, 0) + 1
            amounts[src_owner] = amounts.get(src_owner, 0) + r["amt"]
            counterparts.setdefault(src_owner, set()).add(r["dst_id"])
        for dst_owner in owner.get(r["dst_id"], []):
            counts[dst_owner] = counts.get(dst_owner, 0) + 1
            amounts[dst_owner] = amounts.get(dst_owner, 0) + r["amt"]
            counterparts.setdefault(dst_owner, set()).add(r["src_id"])

    feat["txn_total"] = pd.Series(counts).reindex(feat.index).fillna(0)
    feat["txn_amount_total"] = pd.Series(amounts).reindex(feat.index).fillna(0)
    feat["txn_distinct_counterparts"] = pd.Series({k: len(v) for k, v in counterparts.items()}).reindex(feat.index).fillna(0)


def _add_comm_degree(feat: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    feat["comm_degree"] = 0.0
    calls, phones = tables.get("call_records"), tables.get("phones")
    txns, accounts = tables.get("transactions"), tables.get("accounts")
    phone_owner = _owner_map(tables.get("person_phones"), "phone_id")
    account_owner = _owner_map(tables.get("person_accounts"), "account_id")

    neighbors: dict[str, set] = {}

    def link(a: str, b: str) -> None:
        if a != b:
            neighbors.setdefault(a, set()).add(b)
            neighbors.setdefault(b, set()).add(a)

    if calls is not None and phones is not None and not calls.empty:
        for r in resolve_phone_ids(calls, phones).to_dict("records"):
            for a in phone_owner.get(r["src_id"], []):
                for b in phone_owner.get(r["dst_id"], []):
                    link(a, b)
    if txns is not None and accounts is not None and not txns.empty:
        for r in resolve_account_ids(txns, accounts).to_dict("records"):
            for a in account_owner.get(r["src_id"], []):
                for b in account_owner.get(r["dst_id"], []):
                    link(a, b)

    feat["comm_degree"] = pd.Series({k: len(v) for k, v in neighbors.items()}).reindex(feat.index).fillna(0)


def build_labels(tables: dict[str, pd.DataFrame]) -> pd.Series:
    """Person-level suspect label from case involvement (training only)."""
    cp = tables.get("case_persons")
    if cp is None or cp.empty:
        return pd.Series(dtype=int)
    is_sus = cp.assign(s=cp["involvement_type"].isin(SUSPECT_INVOLVEMENT).astype(int))
    return is_sus.groupby("person_id")["s"].max()
