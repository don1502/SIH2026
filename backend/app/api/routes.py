"""FastAPI routes for the Criminal Intelligence Knowledge Graph + suspect ML."""
from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.api import graph_service as gs
from app.ml import anomaly as ml_anomaly
from app.ml import predict as ml_predict
from app.schema import detect_table

router = APIRouter()


# ---------------------------------------------------------------- graph reads
@router.get("/stats")
def stats():
    return gs.graph_stats()


@router.get("/entities/search")
def search(q: str = Query(..., min_length=1), limit: int = 20):
    return gs.search_entities(q, limit)


@router.get("/entities/{entity_id}")
def profile(entity_id: str):
    result = gs.entity_profile(entity_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")
    return result


@router.get("/entities/{entity_id}/subgraph")
def entity_subgraph(entity_id: str, hops: int = 1, limit: int = 250):
    return gs.subgraph(entity_id, hops=hops, limit=limit)


@router.get("/cases/{case_id}")
def case(case_id: str):
    result = gs.case_details(case_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")
    return result


# ----------------------------------------------------------------- analytics
@router.get("/analytics/summary")
def analytics_summary():
    return {"communities": gs.community_sizes(), "graph": gs.graph_stats()}


@router.get("/analytics/top")
def analytics_top(metric: str = "pagerank", limit: int = 20):
    valid = ("pagerank", "betweenness", "degree", "anomaly_score")
    if metric not in valid:
        raise HTTPException(status_code=400, detail=f"metric must be one of {valid}")
    return gs.top_by_metric(metric, limit)


# ---------------------------------------------------------------- anomalies
@router.get("/anomalies/transactions")
def anomalies_transactions(limit: int = 20):
    return ml_anomaly.top_transactions(limit)


@router.get("/anomalies/calls")
def anomalies_calls(limit: int = 20):
    return ml_anomaly.top_calls(limit)


@router.get("/anomalies/persons")
def anomalies_persons(limit: int = 20):
    return ml_anomaly.top_persons(limit)


@router.get("/analytics/communities")
def analytics_communities():
    return gs.community_sizes()


# ---------------------------------------------------------------- ML / predict
@router.get("/ml/metrics")
def ml_metrics():
    try:
        return ml_predict.model_metrics()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/predict/sample")
def predict_sample(case_id: str | None = None):
    """Predict suspects for one case from the dataset (demo without upload)."""
    try:
        tables = ml_predict.sample_case(case_id)
        result = ml_predict.predict_suspects(tables)
        result["anomalies"] = ml_anomaly.score_all(tables)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/predict/upload")
async def predict_upload(files: list[UploadFile] = File(...)):
    """Accept multiple CSV files, detect their table types, build a graph and
    predict suspects.
    """
    tables: dict[str, pd.DataFrame] = {}
    recognized: dict[str, str] = {}
    unrecognized: list[str] = []

    for f in files:
        content = await f.read()
        try:
            df = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
        except Exception:
            unrecognized.append(f.filename)
            continue
        table = detect_table(df)
        if table is None:
            unrecognized.append(f.filename)
            continue
        tables[table] = df
        recognized[f.filename] = table

    if "persons" not in tables:
        raise HTTPException(
            status_code=400,
            detail="A persons file (with person_id, full_name) is required.",
        )

    try:
        result = ml_predict.predict_suspects(tables)
        result["anomalies"] = ml_anomaly.score_all(tables)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    result["ingestion"] = {"recognized": recognized, "unrecognized": unrecognized}
    return result


# ----------------------------------------------------------------------- admin
@router.post("/admin/pipeline")
def run_pipeline():
    from app.analytics import metrics as analytics
    from app.graph.import_graph import bootstrap
    from app.ingestion.load_csv import load_all_csvs

    ingest = load_all_csvs()
    graph = bootstrap()
    a = analytics.compute_analytics()
    written = analytics.write_back_to_neo4j(a)
    from app.ml import anomaly as anomaly_mod
    from app.ml.train import train as train_suspect

    suspect = train_suspect()
    anomaly_trained = anomaly_mod.train()
    anomaly_written = anomaly_mod.write_back_person_anomaly()
    return {
        "ingested_tables": len(ingest),
        "ingested_rows": sum(ingest.values()),
        "graph": graph,
        "nodes_scored": written,
        "suspect_model": suspect.get("test"),
        "anomaly_model": anomaly_trained,
        "anomaly_persons_scored": anomaly_written,
    }


@router.post("/admin/train")
def train_model():
    from app.ml.train import train

    return train()


@router.post("/admin/train-anomaly")
def train_anomaly():
    return {
        "trained": ml_anomaly.train(),
        "persons_scored": ml_anomaly.write_back_person_anomaly(),
    }
