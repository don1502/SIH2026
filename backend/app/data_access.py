"""Load the cleaned dataset tables as pandas DataFrames."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from app.config import settings

CLEANED_TABLES = [
    "persons", "organizations", "locations", "phones", "accounts", "vehicles",
    "cases", "person_phones", "person_accounts", "person_vehicles",
    "person_locations", "person_organizations", "case_persons", "evidence",
    "events", "call_records", "transactions",
]


@lru_cache(maxsize=1)
def load_cleaned_tables() -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for name in CLEANED_TABLES:
        path = settings.cleaned_path / f"{name}.csv"
        if path.exists():
            tables[name] = pd.read_csv(path, dtype=str, keep_default_na=False)
    return tables
