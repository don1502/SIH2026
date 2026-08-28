"""Read helpers that turn Neo4j / Postgres results into API + Cytoscape shapes."""
from __future__ import annotations

import pandas as pd

from app.db.neo4j_client import run_query
from app.db.postgres import get_engine

_TYPE_LABELS = (
    "Person", "Organization", "Location", "Phone", "Account", "Vehicle", "Case",
)


def search_entities(query: str, limit: int = 20) -> list[dict]:
    cypher = (
        "MATCH (e:Entity) "
        "WHERE toLower(coalesce(e.name, e.id)) CONTAINS toLower($q) "
        "   OR toLower(e.id) CONTAINS toLower($q) "
        "RETURN e.id AS id, e.name AS name, labels(e) AS labels, "
        "       e.role AS role, e.risk_score AS risk_score, e.pagerank AS pagerank "
        "ORDER BY coalesce(e.pagerank, 0) DESC "
        "LIMIT $limit"
    )
    return run_query(cypher, q=query, limit=limit)


def entity_profile(entity_id: str) -> dict | None:
    node = run_query(
        "MATCH (e:Entity {id: $id}) "
        "RETURN e.id AS id, properties(e) AS props, labels(e) AS labels",
        id=entity_id,
    )
    if not node:
        return None

    neighbors = run_query(
        "MATCH (e:Entity {id: $id})-[r]-(n:Entity) "
        "RETURN type(r) AS rel_type, "
        "       startNode(r).id = $id AS outgoing, "
        "       n.id AS neighbor_id, n.name AS neighbor_name, "
        "       labels(n) AS neighbor_labels, properties(r) AS rel_props "
        "LIMIT 200",
        id=entity_id,
    )

    rel_counts = run_query(
        "MATCH (e:Entity {id: $id})-[r]-() "
        "RETURN type(r) AS rel_type, count(r) AS count ORDER BY count DESC",
        id=entity_id,
    )

    return {
        "id": node[0]["id"],
        "labels": node[0]["labels"],
        "properties": node[0]["props"],
        "relationship_counts": rel_counts,
        "neighbors": neighbors,
    }


def subgraph(entity_id: str, hops: int = 1, limit: int = 250) -> dict:
    hops = 1 if hops <= 1 else 2

    node_rows = run_query(
        f"MATCH path = (e:Entity {{id: $id}})-[*1..{hops}]-(m:Entity) "
        "UNWIND nodes(path) AS node "
        "RETURN DISTINCT node.id AS id, node.name AS name, labels(node) AS labels, "
        "       node.role AS role, node.risk_score AS risk_score, "
        "       node.pagerank AS pagerank, node.community AS community, "
        "       node.anomaly_score AS anomaly_score "
        "LIMIT $limit",
        id=entity_id,
        limit=limit * 3,
    )

    edge_rows = run_query(
        f"MATCH path = (e:Entity {{id: $id}})-[*1..{hops}]-(m:Entity) "
        "UNWIND relationships(path) AS rel "
        "RETURN DISTINCT elementId(rel) AS eid, startNode(rel).id AS source, "
        "       endNode(rel).id AS target, type(rel) AS label, properties(rel) AS props "
        "LIMIT $limit",
        id=entity_id,
        limit=limit,
    )

    nodes = [_node_element(r, entity_id) for r in node_rows]
    node_ids = {r["id"] for r in node_rows}
    edges = []
    for r in edge_rows:
        if r["source"] not in node_ids or r["target"] not in node_ids:
            continue
        data = {"id": r["eid"], "source": r["source"], "target": r["target"], "label": r["label"]}
        data.update(r.get("props") or {})
        edges.append({"data": data})
    return {"nodes": nodes, "edges": edges}


def _node_element(row: dict, center_id: str) -> dict:
    labels = row.get("labels") or []
    label = next((k for k in _TYPE_LABELS if k in labels), "Entity")
    return {
        "data": {
            "id": row["id"],
            "label": row.get("name") or row["id"],
            "type": label,
            "role": row.get("role"),
            "risk_score": row.get("risk_score"),
            "pagerank": row.get("pagerank"),
            "community": row.get("community"),
            "anomaly_score": row.get("anomaly_score"),
            "is_anomalous": bool((row.get("anomaly_score") or 0) >= 0.8),
            "is_center": row["id"] == center_id,
        }
    }


def case_details(case_id: str) -> dict | None:
    cases = pd.read_sql(
        "SELECT * FROM cases WHERE case_id = %(cid)s", get_engine(), params={"cid": case_id}
    )
    if cases.empty:
        return None
    evidence = pd.read_sql(
        "SELECT evidence_id, evidence_type, description FROM evidence WHERE case_id = %(cid)s",
        get_engine(), params={"cid": case_id},
    )
    persons = pd.read_sql(
        "SELECT cp.person_id, p.full_name, cp.involvement_type "
        "FROM case_persons cp LEFT JOIN persons p ON cp.person_id = p.person_id "
        "WHERE cp.case_id = %(cid)s",
        get_engine(), params={"cid": case_id},
    )
    out = cases.to_dict("records")[0]
    out["evidence"] = evidence.to_dict("records")
    out["persons"] = persons.to_dict("records")
    return out


def top_by_metric(metric: str, limit: int = 20) -> list[dict]:
    rows = run_query(
        f"MATCH (p:Person) WHERE p.{metric} IS NOT NULL "
        f"RETURN p.id AS id, p.name AS name, p.{metric} AS score, "
        "       p.community AS community, p.role AS role, p.risk_score AS risk_score "
        f"ORDER BY p.{metric} DESC LIMIT $limit",
        limit=limit,
    )
    return [
        {
            "id": r["id"],
            "name": r["name"] or r["id"],
            "score": round(float(r["score"]), 6),
            "community": r["community"],
            "role": r["role"],
            "risk_score": r["risk_score"],
        }
        for r in rows
    ]


def community_sizes() -> dict:
    rows = run_query(
        "MATCH (p:Person) WHERE p.community IS NOT NULL AND p.community >= 0 "
        "RETURN p.community AS community, count(*) AS count ORDER BY count DESC"
    )
    return {
        "num_communities": len(rows),
        "sizes": {str(r["community"]): r["count"] for r in rows},
    }


def graph_stats() -> dict:
    counts = run_query("MATCH (n:Entity) RETURN labels(n) AS labels, count(*) AS count")
    label_counts: dict[str, int] = {}
    for row in counts:
        for label in row["labels"]:
            if label != "Entity":
                label_counts[label] = label_counts.get(label, 0) + row["count"]
    rel_counts = run_query(
        "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(*) AS count ORDER BY count DESC"
    )
    total_nodes = run_query("MATCH (n:Entity) RETURN count(n) AS c")[0]["c"]
    total_rels = run_query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
    return {
        "total_nodes": total_nodes,
        "total_relationships": total_rels,
        "node_labels": label_counts,
        "relationship_types": rel_counts,
    }
