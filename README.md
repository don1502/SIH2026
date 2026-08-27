# Criminal Intelligence Knowledge Graph (SIH26189)

An evidence-backed **Criminal Intelligence Knowledge Graph** that fuses fragmented
crime data (FIRs, CDRs, financial transactions, vehicles, locations, cases) into a
unified Neo4j graph, resolves duplicate identities across sources, and applies graph
analytics to surface hidden leaders, cross-network brokers, communities, and
suspicious patterns — with full provenance for every relationship.

> Built on the synthetic `SIH26189_Criminal_Network_Dataset_v2` benchmark.
> Signals are investigative leads, **not** proof of guilt.

## Architecture

```
27 CSV tables ──► PostgreSQL (staging)
                     │
                     ├─► Entity Resolution (name + DOB + alias matching)
                     │
                     └─► Neo4j Knowledge Graph  ◄── enrichment (transfers, vehicles, cases)
                              │
                              ├─► Graph analytics (PageRank, betweenness, Louvain,
                              │     cross-network brokerage)  → scores written back to Neo4j
                              │
                              └─► FastAPI  ──►  React + Cytoscape.js dashboard
```

## Stack

| Layer | Technology |
|-------|------------|
| Frontend | React + TypeScript + Vite + Cytoscape.js |
| Backend | Python + FastAPI |
| Graph DB | Neo4j 5 |
| Staging DB | PostgreSQL 16 |
| Analytics | networkx, python-louvain, scikit-learn, rapidfuzz |

## Results (vs held-out ground truth)

| Task | Metric | Score |
|------|--------|-------|
| Entity resolution | Precision / Recall / F1 | 0.97 / 0.69 / 0.81 |
| Community detection | NMI / ARI | 0.98 / 0.99 |
| Hidden leader detection | Precision@6 | 1.00 |
| Cross-network broker detection | Precision@8 | 1.00 |

Ground-truth files are used for evaluation only, never during discovery.

## Prerequisites

- Docker Desktop (Neo4j + PostgreSQL)
- Python 3.10+
- Node.js 20+

## Setup

### 1. Start infrastructure

```bash
docker compose up -d
```

Neo4j: http://localhost:7474 (neo4j / sihpassword) · PostgreSQL: localhost:5433

### 2. Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
copy ..\.env.example ..\.env       # then adjust if needed

python -m scripts.run_pipeline     # ingest -> graph -> enrich -> analytics
python -m scripts.evaluate         # print evaluation report

uvicorn app.main:app --reload      # API at http://localhost:8000
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev                        # dashboard at http://localhost:5173
```

## Key API endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Service + DB status |
| `GET /api/stats` | Graph node/edge counts |
| `GET /api/entities/search?q=` | Search entities |
| `GET /api/entities/{id}` | Entity profile + neighbors |
| `GET /api/entities/{id}/subgraph?hops=1` | Cytoscape subgraph |
| `GET /api/evidence/{evidence_id}` | Evidence → source provenance chain |
| `GET /api/analytics/summary` | Leader / broker / community evaluation |
| `GET /api/analytics/top?metric=pagerank` | Ranked entities |
| `GET /api/er/evaluate` | Entity-resolution metrics |
| `GET /api/er/duplicates` | Discovered duplicate identity clusters |

## Project layout

```
SIH2026/
├── docker-compose.yml            # Neo4j + PostgreSQL
├── SIH26189_Criminal_Network_Dataset_v2/
├── backend/
│   ├── app/
│   │   ├── config.py
│   │   ├── main.py               # FastAPI app
│   │   ├── db/                   # postgres + neo4j clients
│   │   ├── ingestion/            # CSV -> PostgreSQL
│   │   ├── graph/                # import_graph + enrich
│   │   ├── er/                   # entity resolution
│   │   ├── analytics/            # centrality, communities, brokerage
│   │   └── api/                  # routes + graph_service
│   └── scripts/
│       ├── run_pipeline.py
│       └── evaluate.py
└── frontend/                     # React + Cytoscape.js dashboard
```
