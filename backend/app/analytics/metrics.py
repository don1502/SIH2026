"""Graph analytics on the cleaned dataset.

Builds a person-to-person projection (two people are linked if their owned
phones called each other or their owned accounts transacted), then runs
PageRank, betweenness and Louvain community detection and writes the scores
back to the Neo4j Person nodes.
"""
from __future__ import annotations

import community as community_louvain
import networkx as nx
import pandas as pd

from app.data_access import load_cleaned_tables
from app.db.neo4j_client import get_driver
from app.schema import resolve_account_ids, resolve_phone_ids


def _owner_map(junction: pd.DataFrame, id_col: str) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    if junction is None:
        return mapping
    for r in junction.to_dict("records"):
        mapping.setdefault(r[id_col], []).append(r["person_id"])
    return mapping


def build_person_graph(tables: dict[str, pd.DataFrame] | None = None) -> nx.Graph:
    tables = tables or load_cleaned_tables()
    graph = nx.Graph()
    for pid in tables.get("persons", pd.DataFrame({"person_id": []}))["person_id"]:
        graph.add_node(pid)

    phone_owner = _owner_map(tables.get("person_phones"), "phone_id")
    account_owner = _owner_map(tables.get("person_accounts"), "account_id")

    def add_pair(a: str, b: str, w: float) -> None:
        if a == b:
            return
        if graph.has_edge(a, b):
            graph[a][b]["weight"] += w
        else:
            graph.add_edge(a, b, weight=w)

    calls, phones = tables.get("call_records"), tables.get("phones")
    if calls is not None and phones is not None:
        for r in resolve_phone_ids(calls, phones).to_dict("records"):
            for a in phone_owner.get(r["src_id"], []):
                for b in phone_owner.get(r["dst_id"], []):
                    add_pair(a, b, 1.0)

    txns, accounts = tables.get("transactions"), tables.get("accounts")
    if txns is not None and accounts is not None:
        for r in resolve_account_ids(txns, accounts).to_dict("records"):
            for a in account_owner.get(r["src_id"], []):
                for b in account_owner.get(r["dst_id"], []):
                    add_pair(a, b, 1.5)

    return graph


def compute_analytics(tables: dict[str, pd.DataFrame] | None = None) -> dict:
    graph = build_person_graph(tables)
    pagerank = nx.pagerank(graph, weight="weight") if graph.number_of_edges() else {}
    betweenness = (
        nx.betweenness_centrality(graph, weight="weight", normalized=True, k=min(200, graph.number_of_nodes()))
        if graph.number_of_edges()
        else {}
    )
    partition = (
        community_louvain.best_partition(graph, weight="weight", random_state=42)
        if graph.number_of_edges()
        else {}
    )
    degree = dict(graph.degree())
    return {
        "graph": graph,
        "pagerank": pagerank,
        "betweenness": betweenness,
        "partition": partition,
        "degree": degree,
    }


def write_back_to_neo4j(analytics: dict) -> int:
    rows = [
        {
            "id": node,
            "pagerank": float(analytics["pagerank"].get(node, 0.0)),
            "betweenness": float(analytics["betweenness"].get(node, 0.0)),
            "community": int(analytics["partition"].get(node, -1)),
            "degree": int(analytics["degree"].get(node, 0)),
        }
        for node in analytics["graph"].nodes()
    ]
    query = (
        "UNWIND $rows AS row MATCH (p:Entity {id: row.id}) "
        "SET p.pagerank = row.pagerank, p.betweenness = row.betweenness, "
        "    p.community = row.community, p.degree = row.degree"
    )
    with get_driver().session() as session:
        for i in range(0, len(rows), 1000):
            session.run(query, rows=rows[i : i + 1000])
    return len(rows)


def community_summary(analytics: dict) -> dict:
    sizes: dict[int, int] = {}
    for c in analytics["partition"].values():
        sizes[c] = sizes.get(c, 0) + 1
    return {"num_communities": len(sizes), "sizes": dict(sorted(sizes.items(), key=lambda x: x[1], reverse=True))}


def run_all() -> dict:
    analytics = compute_analytics()
    written = write_back_to_neo4j(analytics)
    return {"nodes_scored": written, "communities": community_summary(analytics)}


if __name__ == "__main__":
    import json

    print(json.dumps(run_all(), indent=2))
