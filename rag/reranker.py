"""Second-stage re-ranking with a cross-encoder (reads query + passage together, so it is far more
precise than the first-stage vector similarity). Falls back to vector order if the model is unavailable."""
import logging
from typing import Optional

log = logging.getLogger(__name__)


class NoopReranker:
    name = "none (vector order)"

    def score(self, query: str, texts: list[str]) -> Optional[list[float]]:
        return None


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder   # imported lazily: heavy dependency
        self.name = model_name
        self._model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> Optional[list[float]]:
        return [float(s) for s in self._model.predict([(query, t) for t in texts])]


def get_reranker(enabled: bool, model_name: str):
    if not enabled:
        return NoopReranker()
    try:
        return CrossEncoderReranker(model_name)
    except Exception as exc:   # package missing, model download blocked, ...
        log.warning("Reranker unavailable (%s). Using vector similarity order.", exc)
        return NoopReranker()