"""Build the Neo4j knowledge graph from the cleaned crime-intelligence dataset.

Nodes: Person, Organization, Location, Phone, Account, Vehicle, Case.
Edges: OWNS, ASSOCIATED_WITH, PARTICIPATED_IN, CALLED, TRANSACTED_WITH.

call_records reference phones by number and transactions reference accounts by
account_number; both are resolved to node ids before edges are written.
"""
from __future__ import annotations

import pandas as pd

from app.data_access import load_cleaned_tables
from app.db.neo4j_client import get_driver, run_query
from app.schema import NODE_TYPES, resolve_account_ids, resolve_phone_ids


def reset_graph() -> None:
    run_query("MATCH (n) DETACH DELETE n")


def create_constraints() -> None:
    run_query(
        "CREATE CONSTRAINT entity_id IF NOT EXISTS "
        "FOR (e:Entity) REQUIRE e.id IS UNIQUE"
    )
    run_query("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")


def _batch(session, query: str, rows: list[dict], size: int = 1000) -> None:
    for i in range(0, len(rows), size):
        session.run(query, rows=rows[i : i + size])


def load_nodes(tables: dict[str, pd.DataFrame]) -> int:
    total = 0
    driver = get_driver()
    with driver.session() as session:
        for node_type, spec in NODE_TYPES.items():
            df = tables.get(spec["table"])
            if df is None or df.empty:
                continue
            rows = []
            for r in df.to_dict("records"):
                props = {"id": r[spec["id"]], "name": str(r.get(spec["name"], r[spec["id"]]))}
                for col in spec["props"]:
                    val = r.get(col, "")
                    if val not in (None, "", "nan"):
                        props[col] = val
                rows.append(props)
            query = (
                "UNWIND $rows AS row "
                "MERGE (e:Entity {id: row.id}) SET e += row "
                f"SET e:`{spec['label']}`"
            )
            _batch(session, query, rows)
            total += len(rows)
    return total


def _owns_edges(tables: dict[str, pd.DataFrame]) -> list[dict]:
    rows: list[dict] = []
    for jt, id_col in [
        ("person_phones", "phone_id"),
        ("person_accounts", "account_id"),
        ("person_vehicles", "vehicle_id"),
    ]:
        df = tables.get(jt)
        if df is None:
            continue
        for r in df.to_dict("records"):
            rows.append({"src": r["person_id"], "dst": r[id_col]})
    return rows


def _assoc_edges(tables: dict[str, pd.DataFrame]) -> list[dict]:
    rows: list[dict] = []
    for jt, id_col, role_col in [
        ("person_organizations", "org_id", "role_in_org"),
        ("person_locations", "location_id", "association_type"),
    ]:
        df = tables.get(jt)
        if df is None:
            continue
        for r in df.to_dict("records"):
            rows.append({"src": r["person_id"], "dst": r[id_col], "role": r.get(role_col, "")})
    return rows


def load_relationships(tables: dict[str, pd.DataFrame]) -> dict:
    driver = get_driver()
    counts: dict[str, int] = {}
    with driver.session() as session:
        owns = _owns_edges(tables)
        _batch(session,
               "UNWIND $rows AS row MATCH (a:Entity {id: row.src}) "
               "MATCH (b:Entity {id: row.dst}) MERGE (a)-[:OWNS]->(b)", owns)
        counts["OWNS"] = len(owns)

        assoc = _assoc_edges(tables)
        _batch(session,
               "UNWIND $rows AS row MATCH (a:Entity {id: row.src}) "
               "MATCH (b:Entity {id: row.dst}) "
               "MERGE (a)-[r:ASSOCIATED_WITH]->(b) SET r.role = row.role", assoc)
        counts["ASSOCIATED_WITH"] = len(assoc)

        cp = tables.get("case_persons")
        if cp is not None:
            rows = [{"src": r["person_id"], "dst": r["case_id"],
                     "involvement": r.get("involvement_type", "")}
                    for r in cp.to_dict("records")]
            _batch(session,
                   "UNWIND $rows AS row MATCH (a:Entity {id: row.src}) "
                   "MATCH (b:Entity {id: row.dst}) "
                   "MERGE (a)-[r:PARTICIPATED_IN]->(b) SET r.involvement = row.involvement",
                   rows)
            counts["PARTICIPATED_IN"] = len(rows)

        calls = tables.get("call_records")
        phones = tables.get("phones")
        if calls is not None and phones is not None:
            resolved = resolve_phone_ids(calls, phones)
            rows = [{"src": r["src_id"], "dst": r["dst_id"],
                     "timestamp": r.get("timestamp", ""),
                     "duration": float(r.get("duration_seconds") or 0)}
                    for r in resolved.to_dict("records")]
            _batch(session,
                   "UNWIND $rows AS row MATCH (a:Entity {id: row.src}) "
                   "MATCH (b:Entity {id: row.dst}) "
                   "MERGE (a)-[r:CALLED {timestamp: row.timestamp}]->(b) "
                   "SET r.duration = row.duration", rows)
            counts["CALLED"] = len(rows)

        txns = tables.get("transactions")
        accounts = tables.get("accounts")
        if txns is not None and accounts is not None:
            resolved = resolve_account_ids(txns, accounts)
            rows = [{"src": r["src_id"], "dst": r["dst_id"],
                     "timestamp": r.get("timestamp", ""),
                     "amount": float(r.get("amount") or 0),
                     "ttype": r.get("transaction_type", "")}
                    for r in resolved.to_dict("records")]
            _batch(session,
                   "UNWIND $rows AS row MATCH (a:Entity {id: row.src}) "
                   "MATCH (b:Entity {id: row.dst}) "
                   "MERGE (a)-[r:TRANSACTED_WITH {timestamp: row.timestamp}]->(b) "
                   "SET r.amount = row.amount, r.ttype = row.ttype", rows)
            counts["TRANSACTED_WITH"] = len(rows)

    return counts


def bootstrap() -> dict:
    tables = load_cleaned_tables()
    reset_graph()
    create_constraints()
    n_nodes = load_nodes(tables)
    rel_counts = load_relationships(tables)
    return {"nodes": n_nodes, "relationships": sum(rel_counts.values()), "by_type": rel_counts}


if __name__ == "__main__":
    result = bootstrap()
    print(f"Graph bootstrapped: {result['nodes']} nodes, {result['relationships']} relationships")
    print(result["by_type"])
