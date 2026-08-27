"""Suspect prediction and graph construction for uploaded / sample case data."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from app.config import settings
from app.data_access import load_cleaned_tables
from app.ml.features import ALL_FEATURES, NUMERIC_FEATURES, build_person_features
from app.ml.train import METRICS_PATH, MODEL_PATH
from app.schema import NODE_TYPES, resolve_account_ids, resolve_phone_ids

INDICATOR_LABELS = {
    "call_total": "high call volume",
    "call_duration_total": "long total call time",
    "call_distinct_contacts": "many distinct call contacts",
    "txn_total": "high transaction count",
    "txn_amount_total": "large money movement",
    "txn_distinct_counterparts": "many financial counterparts",
    "balance_total": "high account balances",
    "org_threat_max": "linked to high-threat organization",
    "org_threat_sum": "multiple threat-org links",
    "comm_degree": "central in the network",
    "n_accounts": "controls many accounts",
    "n_phones": "controls many phones",
    "n_cases": "tied to multiple cases",
}


@lru_cache(maxsize=1)
def _load_model():
    path = Path(settings.models_path) / MODEL_PATH
    if not path.exists():
        raise FileNotFoundError("Model not trained. Run: python -m app.ml.train")
    return joblib.load(path)


@lru_cache(maxsize=1)
def _load_metrics() -> dict:
    path = Path(settings.models_path) / METRICS_PATH
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def model_metrics() -> dict:
    m = _load_metrics()
    return {k: v for k, v in m.items() if k != "population_stats"}


def _indicators(feat_row: pd.Series, stats: dict) -> list[str]:
    scored = []
    for col in NUMERIC_FEATURES:
        if col not in INDICATOR_LABELS or col not in stats:
            continue
        mean, std = stats[col]["mean"], stats[col]["std"] or 1.0
        z = (float(feat_row.get(col, 0)) - mean) / std
        if z > 0.6:
            scored.append((z, INDICATOR_LABELS[col]))
    scored.sort(reverse=True)
    return [label for _, label in scored[:4]]


def predict_suspects(tables: dict[str, pd.DataFrame], top_k: int = 25) -> dict:
    model = _load_model()
    stats = _load_metrics().get("population_stats", {})
    features = build_person_features(tables)
    if features.empty:
        return {"suspects": [], "graph": {"nodes": [], "edges": []}, "summary": {}}

    proba = model.predict_proba(features[ALL_FEATURES])[:, 1]
    scores = pd.Series(proba, index=features.index)

    persons = tables.get("persons")
    pinfo = persons.set_index("person_id") if persons is not None else pd.DataFrame()

    ranked = scores.sort_values(ascending=False)
    suspects = []
    for pid, prob in ranked.items():
        info = pinfo.loc[pid].to_dict() if pid in pinfo.index else {}
        suspects.append(
            {
                "person_id": pid,
                "name": info.get("full_name", pid),
                "age": info.get("age"),
                "gender": info.get("gender"),
                "role": info.get("role"),
                "risk_score": info.get("risk_score"),
                "suspect_probability": round(float(prob), 4),
                "is_suspect": bool(prob >= 0.5),
                "indicators": _indicators(features.loc[pid], stats),
            }
        )

    proba_map = {pid: float(prob) for pid, prob in scores.items()}
    graph = build_graph_elements(tables, proba_map)
    summary = {
        "persons_scored": int(len(scores)),
        "flagged_suspects": int((scores >= 0.5).sum()),
        "top_probability": round(float(ranked.iloc[0]), 4) if len(ranked) else 0.0,
    }
    return {"suspects": suspects[:top_k], "graph": graph, "summary": summary}


def build_graph_elements(tables: dict[str, pd.DataFrame], proba_map: dict[str, float]) -> dict:
    nodes: dict[str, dict] = {}

    def add_node(node_id: str, node_type: str, label: str, extra: dict | None = None):
        if node_id in nodes:
            return
        data = {"id": node_id, "label": label, "type": node_type}
        if node_type == "Person":
            prob = proba_map.get(node_id)
            data["suspect_probability"] = round(prob, 4) if prob is not None else None
            data["is_suspect"] = bool(prob is not None and prob >= 0.5)
        if extra:
            data.update(extra)
        nodes[node_id] = {"data": data}

    for _, spec in NODE_TYPES.items():
        df = tables.get(spec["table"])
        if df is None:
            continue
        for r in df.to_dict("records"):
            add_node(r[spec["id"]], spec["label"], str(r.get(spec["name"], r[spec["id"]])))

    edges: list[dict] = []

    def add_edge(src: str, dst: str, label: str, extra: dict | None = None):
        if src in nodes and dst in nodes:
            data = {"id": f"e{len(edges)}", "source": src, "target": dst, "label": label}
            if extra:
                data.update(extra)
            edges.append({"data": data})

    for jt, col in [("person_phones", "phone_id"), ("person_accounts", "account_id"),
                    ("person_vehicles", "vehicle_id")]:
        df = tables.get(jt)
        if df is not None:
            for r in df.to_dict("records"):
                add_edge(r["person_id"], r[col], "OWNS")

    for jt, col in [("person_organizations", "org_id"), ("person_locations", "location_id")]:
        df = tables.get(jt)
        if df is not None:
            for r in df.to_dict("records"):
                add_edge(r["person_id"], r[col], "ASSOCIATED_WITH")

    cp = tables.get("case_persons")
    if cp is not None:
        for r in cp.to_dict("records"):
            add_edge(r["person_id"], r["case_id"], "PARTICIPATED_IN",
                     {"involvement": r.get("involvement_type", "")})

    calls, phones = tables.get("call_records"), tables.get("phones")
    if calls is not None and phones is not None and not calls.empty:
        for r in resolve_phone_ids(calls, phones).to_dict("records"):
            add_edge(r["src_id"], r["dst_id"], "CALLED")

    txns, accounts = tables.get("transactions"), tables.get("accounts")
    if txns is not None and accounts is not None and not txns.empty:
        for r in resolve_account_ids(txns, accounts).to_dict("records"):
            add_edge(r["src_id"], r["dst_id"], "TRANSACTED_WITH",
                     {"amount": float(r.get("amount") or 0)})

    return {"nodes": list(nodes.values()), "edges": edges}


def sample_case(case_id: str | None = None) -> dict[str, pd.DataFrame]:
    """Assemble a self-contained set of tables for one case, for demoing the
    upload-and-predict flow without requiring the user to have files.
    """
    tables = load_cleaned_tables()
    cases = tables["cases"]
    cp_all = tables["case_persons"]
    if case_id is None:
        case_id = cases["case_id"].sample(1, random_state=None).iloc[0]

    cp = cp_all[cp_all["case_id"] == case_id]
    person_ids = set(cp["person_id"])

    def filt(table: str, col: str, keep: set) -> pd.DataFrame:
        df = tables.get(table)
        return df[df[col].isin(keep)].copy() if df is not None else pd.DataFrame()

    persons = tables["persons"][tables["persons"]["person_id"].isin(person_ids)].copy()
    pp = filt("person_phones", "person_id", person_ids)
    pa = filt("person_accounts", "person_id", person_ids)
    pv = filt("person_vehicles", "person_id", person_ids)
    po = filt("person_organizations", "person_id", person_ids)
    pl = filt("person_locations", "person_id", person_ids)

    owned_phone_ids = set(pp["phone_id"]) if not pp.empty else set()
    owned_account_ids = set(pa["account_id"]) if not pa.empty else set()

    all_phones = tables["phones"]
    all_accounts = tables["accounts"]
    owned_phone_numbers = set(
        all_phones[all_phones["phone_id"].isin(owned_phone_ids)]["phone_number"]
    )
    owned_account_numbers = set(
        all_accounts[all_accounts["account_id"].isin(owned_account_ids)]["account_number"]
    )

    # Include the full call / transaction activity of the persons' phones and
    # accounts (not just intra-case links), matching the training distribution.
    calls = tables["call_records"]
    calls = calls[
        calls["caller_number"].isin(owned_phone_numbers)
        | calls["receiver_number"].isin(owned_phone_numbers)
    ].copy()
    txns = tables["transactions"]
    txns = txns[
        txns["sender_account"].isin(owned_account_numbers)
        | txns["receiver_account"].isin(owned_account_numbers)
    ].copy()

    # Phone / account nodes: owned ones plus counterparts seen in the activity.
    phone_numbers = owned_phone_numbers | set(calls["caller_number"]) | set(calls["receiver_number"])
    account_numbers = (
        owned_account_numbers | set(txns["sender_account"]) | set(txns["receiver_account"])
    )
    phones = all_phones[all_phones["phone_number"].isin(phone_numbers)].copy()
    accounts = all_accounts[all_accounts["account_number"].isin(account_numbers)].copy()
    vehicles = filt("vehicles", "vehicle_id", set(pv["vehicle_id"]) if not pv.empty else set())
    orgs = filt("organizations", "org_id", set(po["org_id"]) if not po.empty else set())
    locations = filt("locations", "location_id", set(pl["location_id"]) if not pl.empty else set())

    return {
        "persons": persons, "phones": phones, "accounts": accounts, "vehicles": vehicles,
        "organizations": orgs, "locations": locations,
        "cases": cases[cases["case_id"] == case_id].copy(),
        "person_phones": pp, "person_accounts": pa, "person_vehicles": pv,
        "person_organizations": po, "person_locations": pl, "case_persons": cp,
        "call_records": calls, "transactions": txns,
    }
