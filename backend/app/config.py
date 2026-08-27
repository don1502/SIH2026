from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    dataset_dir: str = str(REPO_ROOT / "SIH26189_Criminal_Network_Dataset_v2")
    cleaned_dir: str = str(REPO_ROOT / "cleaned_dataset")
    graph_ml_dir: str = str(REPO_ROOT / "graph_ml_export")
    models_dir: str = str(REPO_ROOT / "backend" / "models")

    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_user: str = "sih"
    postgres_password: str = "sih"
    postgres_db: str = "sih"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "sihpassword"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def dataset_path(self) -> Path:
        p = Path(self.dataset_dir)
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()

    @property
    def csv_dir(self) -> Path:
        return self.dataset_path / "csv"

    @property
    def neo4j_dir(self) -> Path:
        return self.dataset_path / "neo4j"

    def _resolve(self, raw: str) -> Path:
        p = Path(raw)
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()

    @property
    def cleaned_path(self) -> Path:
        return self._resolve(self.cleaned_dir)

    @property
    def graph_ml_path(self) -> Path:
        return self._resolve(self.graph_ml_dir)

    @property
    def models_path(self) -> Path:
        return self._resolve(self.models_dir)

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
