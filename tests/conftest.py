import pytest

from config import get_settings
from scripts.seed_database import build_database


@pytest.fixture(autouse=True)
def seeded_db(tmp_path, monkeypatch):
    """Every test runs against its own freshly seeded temp database."""
    path = tmp_path / "test.db"
    build_database(path)
    monkeypatch.setenv("DB_PATH", str(path))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    get_settings.cache_clear()
    from agent.context import reset_retriever_cache
    reset_retriever_cache()
    yield path
    reset_retriever_cache()
    get_settings.cache_clear()
