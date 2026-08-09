import pytest

from aicache.matching.embedder import Embedder


@pytest.fixture(scope="session")
def embedder() -> Embedder:
    """One model load for the whole test session — sentence-transformers
    load time (~1-2s) would otherwise dominate every test that needs it."""
    return Embedder()
