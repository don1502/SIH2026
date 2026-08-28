"""Unsupervised anomaly detection for money flows and call bursts.

Three IsolationForest models score how anomalous each transaction, account and
phone is, with no labels required. Per-entity reasons are derived from how far
each feature deviates from the population norm, and person-level scores are
aggregated from the person's owned accounts and phones.

The same models are used to score the full dataset (for dashboard rankings and
Neo4j write-back) and any uploaded subset of tables.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config import settings
from app.data_access import load_cleaned_tables
from app.db.neo4j_client import get_driver
from app.schema import resolve_account_ids, resolve_phone_ids

META_PATH = "anomaly_meta.json"

TXN_NUMERIC = ["log_amount", "amount_z", "hour", "is_offhours", "sender_fanout", "receiver_fanin"]
TXN_CATEGORICAL = ["transaction_type"]
ACCOUNT_NUMERIC = [
    "txn_count", "total_out", "total_in", "mean_amount", "std_amount", "max_amount",
    "distinct_counterparts", "fan_out", "fan_in", "fan_ratio", "throughput",
    "balance", "balance_throughput_ratio", "offhours_ratio",
]
PHONE_NUMERIC = [
    "call_count", "total_duration", "mean_duration", "distinct_contacts",
    "offhours_ratio", "max_calls_in_hour",
]

LEVELS = {
    "txn": {"file": "anomaly_txn.joblib", "numeric": TXN_NUMERIC, "categorical": TXN_CATEGORICAL},
    "account": {"file": "anomaly_account.joblib", "numeric": ACCOUNT_NUMERIC, "categorical": []},
    "phone": {"file": "anomaly_phone.joblib", "numeric": PHONE_NUMERIC, "categorical": []},
}

TXN_REASONS = {
    "amount_z": "amount far above account norm",
    "log_amount": "unusually large transfer",
    "is_offhours": "off-hours transaction",
    "sender_fanout": "sender spreads funds widely",
    "receiver_fanin": "receiver aggregates many inflows",
}
ACCOUNT_REASONS = {
    "total_out": "high outgoing volume",
    "total_in": "high incoming volume",
    "max_amount": "very large single transfer",
    "distinct_counterparts": "many financial counterparts",
    "fan_out": "sends to many accounts",
    "fan_in": "receives from many accounts",
    "throughput": "high money throughput",
    "balance_throughput_ratio": "throughput dwarfs balance (pass-through)",
    "offhours_ratio": "frequent off-hours activity",
    "txn_count": "high transaction velocity",
}
PHONE_REASONS = {
    "call_count": "high call volume",
    "total_duration": "long total call time",
    "distinct_contacts": "many distinct contacts",
    "max_calls_in_hour": "call burst within one hour",
    "offhours_ratio": "frequent off-hours calls",
    "mean_duration": "unusually long calls",
}
REASONS = {"txn": TXN_REASONS, "account": ACCOUNT_REASONS, "phone": PHONE_REASONS}


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _hours(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.hour


# --------------------------------------------------------------- feature builders
def build_txn_features(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    txns, accounts = tables.get("transactions"), tables.get("accounts")
    if txns is None or accounts is None or txns.empty:
        return pd.DataFrame(columns=TXN_NUMERIC + TXN_CATEGORICAL), pd.DataFrame()

    r = resolve_account_ids(txns, accounts).reset_index(drop=True)
    r["amount"] = _num(r.get("amount", 0)).fillna(0)
    r["hour"] = _hours(r.get("timestamp", "")).fillna(12)
    r["is_offhours"] = (r["hour"] < 6).astype(int)

    stats = r.groupby("src_id")["amount"].agg(["mean", "std"])
    r = r.merge(stats.rename(columns={"mean": "s_mean", "std": "s_std"}), left_on="src_id", right_index=True, how="left")
    r["amount_z"] = (r["amount"] - r["s_mean"]) / (r["s_std"].fillna(0) + 1.0)
    r["log_amount"] = np.log1p(r["amount"])
    r["sender_fanout"] = r["src_id"].map(r.groupby("src_id").size())
    r["receiver_fanin"] = r["dst_id"].map(r.groupby("dst_id").size())
    r["transaction_type"] = r.get("transaction_type", "Unknown").fillna("Unknown")

    if "transaction_id" in r.columns:
        idx = r["transaction_id"].astype(str).values
    else:
        idx = np.array([f"TXN_{i}" for i in range(len(r))])
    feats = r[TXN_NUMERIC + TXN_CATEGORICAL].copy()
    feats.index = idx
    for c in TXN_NUMERIC:
        feats[c] = _num(feats[c]).fillna(0)

    meta = pd.DataFrame(
        {
            "transaction_id": idx,
            "sender_account": r.get("sender_account", pd.Series([""] * len(r))).values,
            "receiver_account": r.get("receiver_account", pd.Series([""] * len(r))).values,
            "amount": r["amount"].values,
            "transaction_type": r["transaction_type"].values,
            "timestamp": r.get("timestamp", pd.Series([""] * len(r))).values,
        },
        index=idx,
    )
    return feats, meta


def build_account_features(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    accounts = tables.get("accounts")
    txns = tables.get("transactions")
    if accounts is None or accounts.empty:
        return pd.DataFrame(columns=ACCOUNT_NUMERIC), pd.DataFrame()

    ids = accounts["account_id"].values
    feats = pd.DataFrame(0.0, index=ids, columns=ACCOUNT_NUMERIC)
    feats.index.name = "account_id"

    if txns is not None and not txns.empty:
        r = resolve_account_ids(txns, accounts).copy()
        r["amount"] = _num(r.get("amount", 0)).fillna(0)
        r["offhours"] = (_hours(r.get("timestamp", "")).fillna(12) < 6).astype(int)

        out = r.groupby("src_id")["amount"].agg(out_count="count", total_out="sum")
        inn = r.groupby("dst_id")["amount"].agg(in_count="count", total_in="sum")
        mx = pd.concat([r.groupby("src_id")["amount"].max(), r.groupby("dst_id")["amount"].max()], axis=1).max(axis=1)
        allamt_mean = pd.concat([r[["src_id", "amount"]].rename(columns={"src_id": "a"}),
                                 r[["dst_id", "amount"]].rename(columns={"dst_id": "a"})]).groupby("a")["amount"]
        mean_amt = allamt_mean.mean()
        std_amt = allamt_mean.std()
        offh = pd.concat([r[["src_id", "offhours"]].rename(columns={"src_id": "a"}),
                          r[["dst_id", "offhours"]].rename(columns={"dst_id": "a"})]).groupby("a")["offhours"].mean()

        cps: dict[str, set] = {}
        for row in r[["src_id", "dst_id"]].itertuples(index=False):
            cps.setdefault(row.src_id, set()).add(row.dst_id)
            cps.setdefault(row.dst_id, set()).add(row.src_id)
        distinct_cp = pd.Series({k: len(v) for k, v in cps.items()})

        feats["fan_out"] = out["out_count"].reindex(ids).fillna(0)
        feats["fan_in"] = inn["in_count"].reindex(ids).fillna(0)
        feats["total_out"] = out["total_out"].reindex(ids).fillna(0)
        feats["total_in"] = inn["total_in"].reindex(ids).fillna(0)
        feats["txn_count"] = feats["fan_out"] + feats["fan_in"]
        feats["max_amount"] = mx.reindex(ids).fillna(0)
        feats["mean_amount"] = mean_amt.reindex(ids).fillna(0)
        feats["std_amount"] = std_amt.reindex(ids).fillna(0)
        feats["distinct_counterparts"] = distinct_cp.reindex(ids).fillna(0)
        feats["offhours_ratio"] = offh.reindex(ids).fillna(0)

    feats["fan_ratio"] = feats["fan_out"] / (feats["fan_in"] + 1.0)
    feats["throughput"] = feats["total_out"] + feats["total_in"]
    feats["balance"] = _num(accounts.set_index("account_id").get("balance")).reindex(ids).fillna(0).values
    feats["balance_throughput_ratio"] = feats["throughput"] / (feats["balance"].abs() + 1.0)

    meta = accounts.set_index("account_id").reindex(ids)[["account_number", "bank_name"]].copy()
    return feats, meta


def build_phone_features(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    phones = tables.get("phones")
    calls = tables.get("call_records")
    if phones is None or phones.empty:
        return pd.DataFrame(columns=PHONE_NUMERIC), pd.DataFrame()

    ids = phones["phone_id"].values
    feats = pd.DataFrame(0.0, index=ids, columns=PHONE_NUMERIC)
    feats.index.name = "phone_id"

    if calls is not None and not calls.empty:
        r = resolve_phone_ids(calls, phones).copy()
        r["duration"] = _num(r.get("duration_seconds", 0)).fillna(0)
        ts = pd.to_datetime(r.get("timestamp", ""), errors="coerce", utc=True)
        r["hour"] = ts.dt.hour.fillna(12)
        r["offhours"] = (r["hour"] < 6).astype(int)
        r["hour_bucket"] = ts.dt.floor("h")

        long = pd.concat([
            r[["src_id", "duration", "offhours", "dst_id", "hour_bucket"]].rename(
                columns={"src_id": "p", "dst_id": "other"}),
            r[["dst_id", "duration", "offhours", "src_id", "hour_bucket"]].rename(
                columns={"dst_id": "p", "src_id": "other"}),
        ])
        agg = long.groupby("p")["duration"].agg(call_count="count", total_duration="sum", mean_duration="mean")
        offh = long.groupby("p")["offhours"].mean()
        contacts = long.groupby("p")["other"].nunique()
        burst = long.groupby(["p", "hour_bucket"]).size().groupby(level=0).max()

        feats["call_count"] = agg["call_count"].reindex(ids).fillna(0)
        feats["total_duration"] = agg["total_duration"].reindex(ids).fillna(0)
        feats["mean_duration"] = agg["mean_duration"].reindex(ids).fillna(0)
        feats["distinct_contacts"] = contacts.reindex(ids).fillna(0)
        feats["offhours_ratio"] = offh.reindex(ids).fillna(0)
        feats["max_calls_in_hour"] = burst.reindex(ids).fillna(0)

    meta = phones.set_index("phone_id").reindex(ids)[["phone_number", "service_provider"]].copy()
    return feats, meta


# ------------------------------------------------------------------- model plumbing
def _build_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    transformers = [("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                      ("scale", StandardScaler())]), numeric)]
    if categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    pre = ColumnTransformer(transformers)
    model = IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=-1)
    return Pipeline([("pre", pre), ("clf", model)])


def _active_mask(level: str, feats: pd.DataFrame) -> pd.Series:
    """Entities that actually participate, so anomalies reflect unusual behaviour
    among real actors rather than active-vs-inactive."""
    if level == "account" and "txn_count" in feats:
        return feats["txn_count"] > 0
    if level == "phone" and "call_count" in feats:
        return feats["call_count"] > 0
    return pd.Series(True, index=feats.index)


def _rank_scores(pipe: Pipeline, X: pd.DataFrame, extra: pd.Series | None = None, extra_weight: float = 0.35) -> np.ndarray:
    raw = -pipe.score_samples(X)  # higher = more anomalous
    if_rank = pd.Series(raw, index=X.index).rank(pct=True)
    if extra is None:
        return if_rank.values
    extra_rank = extra.reindex(X.index).fillna(0).clip(lower=0).rank(pct=True)
    blended = (1.0 - extra_weight) * if_rank + extra_weight * extra_rank
    return pd.concat([blended, extra_rank], axis=1).max(axis=1).values


def _score_active(level: str, feats: pd.DataFrame) -> pd.Series:
    """Score only active entities; returns a Series of anomaly scores (0-1)."""
    active = feats[_active_mask(level, feats)]
    if active.empty:
        return pd.Series(dtype=float)
    pipe = _get_pipeline(level, active)
    if pipe is None:
        return pd.Series(dtype=float)
    extra = None
    if level == "phone" and "max_calls_in_hour" in active:
        extra = active["max_calls_in_hour"].astype(float) + 0.5 * active.get("call_count", 0).astype(float)
    elif level == "account" and "throughput" in active:
        extra = active["throughput"].astype(float)
    return pd.Series(_rank_scores(pipe, active, extra), index=active.index)


@lru_cache(maxsize=1)
def _load_meta() -> dict:
    path = Path(settings.models_path) / META_PATH
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=3)
def _load_model(level: str):
    path = Path(settings.models_path) / LEVELS[level]["file"]
    if not path.exists():
        return None
    return joblib.load(path)


def _get_pipeline(level: str, X: pd.DataFrame):
    """Return the trained pipeline, or fit an ephemeral one on X as a fallback."""
    pipe = _load_model(level)
    if pipe is not None:
        return pipe
    if len(X) < 10:
        return None
    pipe = _build_pipeline(LEVELS[level]["numeric"], LEVELS[level]["categorical"])
    pipe.fit(X)
    return pipe


def _reasons(level: str, feat_row: pd.Series, stats: dict) -> list[str]:
    out = []
    reason_map = REASONS[level]
    for col in LEVELS[level]["numeric"]:
        if col not in reason_map or col not in stats:
            continue
        mean, std = stats[col]["mean"], stats[col]["std"] or 1.0
        val = float(feat_row.get(col, 0))
        if col == "is_offhours":
            if val >= 1:
                out.append((5.0, reason_map[col]))
            continue
        z = (val - mean) / std
        if z > 1.0:
            out.append((z, reason_map[col]))
    out.sort(reverse=True)
    return [r for _, r in out[:3]]


# ------------------------------------------------------------------ scoring helpers
def _owner_names(tables: dict[str, pd.DataFrame], junction: str, id_col: str) -> dict[str, str]:
    j, persons = tables.get(junction), tables.get("persons")
    if j is None or persons is None:
        return {}
    name = dict(zip(persons["person_id"], persons["full_name"]))
    owners: dict[str, str] = {}
    for r in j[[id_col, "person_id"]].itertuples(index=False):
        owners.setdefault(getattr(r, id_col), name.get(r.person_id, r.person_id))
    return owners


def score_transactions(tables: dict[str, pd.DataFrame], limit: int = 20) -> list[dict]:
    feats, meta = build_txn_features(tables)
    if feats.empty:
        return []
    pipe = _get_pipeline("txn", feats)
    if pipe is None:
        return []
    extra = feats["log_amount"].astype(float) + feats.get("amount_z", 0).clip(lower=0)
    scores = _rank_scores(pipe, feats, extra)
    stats = _load_meta().get("txn", {})
    acct_owner = _owner_names(tables, "person_accounts", "account_id")
    # map account_number -> owner name via accounts table
    accounts = tables.get("accounts")
    num_to_owner = {}
    if accounts is not None:
        for r in accounts[["account_id", "account_number"]].itertuples(index=False):
            num_to_owner[r.account_number] = acct_owner.get(r.account_id, "")

    order = np.argsort(scores)[::-1][:limit]
    results = []
    for i in order:
        idx = feats.index[i]
        m = meta.loc[idx]
        results.append({
            "transaction_id": str(m["transaction_id"]),
            "sender_account": str(m["sender_account"]),
            "receiver_account": str(m["receiver_account"]),
            "sender_owner": num_to_owner.get(str(m["sender_account"]), ""),
            "receiver_owner": num_to_owner.get(str(m["receiver_account"]), ""),
            "amount": round(float(m["amount"]), 2),
            "transaction_type": str(m["transaction_type"]),
            "timestamp": str(m["timestamp"]),
            "anomaly_score": round(float(scores[i]), 4),
            "reasons": _reasons("txn", feats.loc[idx], stats),
        })
    return results


def score_calls(tables: dict[str, pd.DataFrame], limit: int = 20) -> list[dict]:
    feats, meta = build_phone_features(tables)
    if feats.empty:
        return []
    scores = _score_active("phone", feats)
    if scores.empty:
        return []
    stats = _load_meta().get("phone", {})
    owner = _owner_names(tables, "person_phones", "phone_id")

    ranked = scores.sort_values(ascending=False)
    active_max = int(feats.loc[scores.index, "call_count"].max())
    min_calls = 5 if active_max >= 5 else 2  # relax for small uploaded sets
    results = []
    for idx, sc in ranked.items():
        f = feats.loc[idx]
        if f["call_count"] < min_calls:
            continue
        m = meta.loc[idx] if idx in meta.index else None
        results.append({
            "phone_id": str(idx),
            "phone_number": str(m["phone_number"]) if m is not None else str(idx),
            "owner": owner.get(str(idx), ""),
            "call_count": int(f["call_count"]),
            "distinct_contacts": int(f["distinct_contacts"]),
            "max_calls_in_hour": int(f["max_calls_in_hour"]),
            "total_duration": int(f["total_duration"]),
            "anomaly_score": round(float(sc), 4),
            "reasons": _reasons("phone", f, stats),
        })
        if len(results) >= limit:
            break
    return results


def compute_person_scores(tables: dict[str, pd.DataFrame]) -> pd.Series:
    """Person-level anomaly (0-1) aggregated from owned accounts and phones."""
    acc_feats, _ = build_account_features(tables)
    ph_feats, _ = build_phone_features(tables)
    acc_scores = _score_active("account", acc_feats) if not acc_feats.empty else pd.Series(dtype=float)
    ph_scores = _score_active("phone", ph_feats) if not ph_feats.empty else pd.Series(dtype=float)

    persons = tables.get("persons")
    if persons is None or persons.empty:
        return pd.Series(dtype=float)
    result = pd.Series(0.0, index=persons["person_id"].values)

    pa = tables.get("person_accounts")
    if pa is not None and not acc_scores.empty:
        pa2 = pa.assign(s=pa["account_id"].map(acc_scores)).dropna(subset=["s"])
        amax = pa2.groupby("person_id")["s"].max()
        result = pd.concat([result, amax], axis=1).fillna(0).max(axis=1)
    pp = tables.get("person_phones")
    if pp is not None and not ph_scores.empty:
        pp2 = pp.assign(s=pp["phone_id"].map(ph_scores)).dropna(subset=["s"])
        pmax = pp2.groupby("person_id")["s"].max()
        result = pd.concat([result, pmax], axis=1).fillna(0).max(axis=1)
    return result.fillna(0.0)


def _person_reasons(pid: str, tables: dict[str, pd.DataFrame],
                    acc_feats: pd.DataFrame, ph_feats: pd.DataFrame) -> list[str]:
    stats = _load_meta()
    reasons: list[str] = []
    pa = tables.get("person_accounts")
    if pa is not None and not acc_feats.empty:
        owned = pa.loc[pa["person_id"] == pid, "account_id"]
        for aid in owned:
            if aid in acc_feats.index:
                reasons.extend(_reasons("account", acc_feats.loc[aid], stats.get("account", {})))
    pp = tables.get("person_phones")
    if pp is not None and not ph_feats.empty:
        owned = pp.loc[pp["person_id"] == pid, "phone_id"]
        for phid in owned:
            if phid in ph_feats.index:
                reasons.extend(_reasons("phone", ph_feats.loc[phid], stats.get("phone", {})))
    seen: list[str] = []
    for r in reasons:
        if r not in seen:
            seen.append(r)
    return seen[:4]


def score_persons(tables: dict[str, pd.DataFrame], limit: int = 20) -> list[dict]:
    scores = compute_person_scores(tables)
    if scores.empty:
        return []
    persons_df = tables.get("persons")
    if persons_df is None:
        return []
    persons = persons_df.set_index("person_id")
    acc_feats, _ = build_account_features(tables)
    ph_feats, _ = build_phone_features(tables)
    top = scores.sort_values(ascending=False).head(limit)
    out = []
    for pid, sc in top.items():
        info = persons.loc[pid].to_dict() if pid in persons.index else {}
        out.append({
            "person_id": pid,
            "name": info.get("full_name", pid),
            "role": info.get("role"),
            "risk_score": info.get("risk_score"),
            "anomaly_score": round(float(sc), 4),
            "reasons": _person_reasons(pid, tables, acc_feats, ph_feats),
        })
    return out


def score_all(tables: dict[str, pd.DataFrame], limit: int = 15) -> dict:
    empty = {"transactions": [], "calls": [], "persons": []}
    try:
        return {
            "transactions": score_transactions(tables, limit),
            "calls": score_calls(tables, limit),
            "persons": score_persons(tables, limit),
        }
    except Exception:
        return empty


# --------------------------------------------------------------------- dashboard API
def top_transactions(limit: int = 20) -> list[dict]:
    return score_transactions(load_cleaned_tables(), limit)


def top_calls(limit: int = 20) -> list[dict]:
    return score_calls(load_cleaned_tables(), limit)


def top_persons(limit: int = 20) -> list[dict]:
    return score_persons(load_cleaned_tables(), limit)


# --------------------------------------------------------------------- train / write
def _meta_stats(feats: pd.DataFrame, numeric: list[str]) -> dict:
    return {c: {"mean": float(feats[c].mean()), "std": float(feats[c].std() or 1.0)} for c in numeric}


def train() -> dict:
    tables = load_cleaned_tables()
    models_dir = Path(settings.models_path)
    models_dir.mkdir(parents=True, exist_ok=True)

    builders = {"txn": build_txn_features, "account": build_account_features, "phone": build_phone_features}
    meta: dict = {}
    summary: dict = {}
    for level, builder in builders.items():
        feats, _ = builder(tables)
        active = feats[_active_mask(level, feats)]
        cfg = LEVELS[level]
        pipe = _build_pipeline(cfg["numeric"], cfg["categorical"])
        pipe.fit(active)
        joblib.dump(pipe, models_dir / cfg["file"])
        meta[level] = _meta_stats(active, cfg["numeric"])
        summary[level] = int(len(active))

    with open(models_dir / META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    _load_meta.cache_clear()
    _load_model.cache_clear()
    return {"trained": summary}


def write_back_person_anomaly(scores: pd.Series | None = None) -> int:
    if scores is None:
        scores = compute_person_scores(load_cleaned_tables())
    rows = [{"id": pid, "anomaly": float(sc)} for pid, sc in scores.items()]
    query = ("UNWIND $rows AS row MATCH (p:Entity {id: row.id}) "
             "SET p.anomaly_score = row.anomaly")
    with get_driver().session() as session:
        for i in range(0, len(rows), 1000):
            session.run(query, rows=rows[i : i + 1000])
    return len(rows)


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
    print("person nodes scored:", write_back_person_anomaly())
