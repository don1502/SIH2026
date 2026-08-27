from functools import lru_cache

from neo4j import Driver, GraphDatabase

from app.config import settings


@lru_cache(maxsize=1)
def get_driver() -> Driver:
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


def run_query(query: str, **params):
    with get_driver().session() as session:
        result = session.run(query, **params)
        return [record.data() for record in result]


def ping() -> bool:
    try:
        run_query("RETURN 1 AS ok")
        return True
    except Exception:
        return False
