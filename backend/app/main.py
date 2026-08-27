from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.db import neo4j_client, postgres

app = FastAPI(
    title="Criminal Intelligence Knowledge Graph API",
    description="SIH26189 - evidence-backed graph analytics for investigators",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "postgres": postgres.ping(),
        "neo4j": neo4j_client.ping(),
    }
