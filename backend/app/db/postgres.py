from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.config import settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(settings.postgres_url, pool_pre_ping=True, future=True)


def ping() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
