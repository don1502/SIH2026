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
from app.ml.train import train  # noqa: E402


def main() -> None:
    t0 = time.time()

    print("[1/4] Ingesting cleaned CSVs into PostgreSQL...")
    ingest = load_all_csvs()
    print(f"      {len(ingest)} tables, {sum(ingest.values())} rows")

    print("[2/4] Building Neo4j knowledge graph...")
    graph = bootstrap()
    print(f"      {graph['nodes']} nodes, {graph['relationships']} relationships {graph['by_type']}")

    print("[3/4] Computing analytics and writing scores back to Neo4j...")
    analytics = compute_analytics()
    written = write_back_to_neo4j(analytics)
    print(f"      scored {written} person nodes")

    print("[4/4] Training suspect-prediction model...")
    metrics = train()
    print(f"      test ROC-AUC={metrics['test']['roc_auc']} "
          f"PR-AUC={metrics['test']['pr_auc']} F1={metrics['test']['f1']}")

    print(f"\nPipeline complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
