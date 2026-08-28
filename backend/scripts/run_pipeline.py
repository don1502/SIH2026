"""End-to-end build on the cleaned dataset:
ingest CSVs -> build graph -> analytics -> train suspect model.

Run from the backend/ directory:
    python -m scripts.run_pipeline
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.analytics.metrics import compute_analytics, write_back_to_neo4j  # noqa: E402
from app.graph.import_graph import bootstrap  # noqa: E402
from app.ingestion.load_csv import load_all_csvs  # noqa: E402
from app.ml import anomaly  # noqa: E402
from app.ml.train import train  # noqa: E402


def main() -> None:
    t0 = time.time()

    print("[1/5] Ingesting cleaned CSVs into PostgreSQL...")
    ingest = load_all_csvs()
    print(f"      {len(ingest)} tables, {sum(ingest.values())} rows")

    print("[2/5] Building Neo4j knowledge graph...")
    graph = bootstrap()
    print(f"      {graph['nodes']} nodes, {graph['relationships']} relationships {graph['by_type']}")

    print("[3/5] Computing analytics and writing scores back to Neo4j...")
    analytics = compute_analytics()
    written = write_back_to_neo4j(analytics)
    print(f"      scored {written} person nodes")

    print("[4/5] Training suspect-prediction model...")
    metrics = train()
    print(f"      test ROC-AUC={metrics['test']['roc_auc']} "
          f"PR-AUC={metrics['test']['pr_auc']} F1={metrics['test']['f1']}")

    print("[5/5] Training anomaly-detection models...")
    trained = anomaly.train()
    scored = anomaly.write_back_person_anomaly()
    print(f"      {trained['trained']}; wrote anomaly_score to {scored} persons")

    print(f"\nPipeline complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
