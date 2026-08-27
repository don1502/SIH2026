"""Ingest all dataset CSVs into PostgreSQL staging tables.

Each CSV file becomes a table named after the file (e.g. persons.csv -> persons).
This is the raw structured staging layer; entity resolution and graph build
read from these tables.
"""
from __future__ import annotations

import pandas as pd

from app.config import settings
from app.db.postgres import get_engine


def load_all_csvs() -> dict[str, int]:
    """Load every CSV in the dataset csv/ folder into a Postgres table.

    Returns a mapping of table_name -> row_count.
    """
    engine = get_engine()
    results: dict[str, int] = {}

    csv_files = sorted(settings.cleaned_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {settings.cleaned_path}")

    for csv_path in csv_files:
        table = csv_path.stem.lower()
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        df.to_sql(table, engine, if_exists="replace", index=False, chunksize=5000)
        results[table] = len(df)

    return results


if __name__ == "__main__":
    counts = load_all_csvs()
    total = sum(counts.values())
    print(f"Loaded {len(counts)} tables, {total} rows into PostgreSQL:")
    for table, n in sorted(counts.items()):
        print(f"  {table:<32} {n:>7}")
