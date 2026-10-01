"""Optional local cross-encoder reranker between RRF fusion and the judge (v0.16).

Pipeline: lexical + dense (bge-m3) → RRF → cross-encoder over the leading ranks → a smaller
top-k for the judge. A cross-encoder reads the duty and the passage together, which a
bi-encoder similarity cannot; it is used to ORDER passages and to shrink the judge window,
never as evidence and never as a hard gate on its own.

It runs in this process (sentence-transformers / torch on CPU or GPU): policy text does not
leave the machine. Model weights come from the local Hugging Face cache; development
downloads require RERANK_ALLOW_DOWNLOAD=on, and server mode prohibits downloads. Without the library or the weights the pilot
says so in the packet and keeps the RRF order; it never fails a run for a missing reranker.
"""
import math
import os
import time
from hashlib import sha256
from weakref import WeakValueDictionary

from regchain.model_router import ModelPolicyError, cache_namespace, current_model_scope, model_mode

VERSION = 'cross-encoder-v1'
DEFAULT_MODEL = 'BAAI/bge-reranker-v2-m3'
# How many of the fused ranks are rescored. Beyond this the RRF order stands.
RERANK_CANDIDATES = 20
# Weights can be shared; the wrappers carrying mutable call telemetry are scoped to
# a single run and disappear once the run no longer holds them.
_INSTANCES = WeakValueDictionary()
_ENCODERS = {}
# Cross-encoder scores are deterministic for a pinned model, so a duty/passage pair scored in an
# earlier analysis for the same tenant/company is not scored again (v0.18). Keyed by the model, its max_length
# (a longer pair is cut there) and hashes of the two texts, so no policy text is held as a key;
# bounded like the embedding cache.
SCORE_CACHE_LIMIT = 50000
_SCORE_CACHE = {}


class RerankUnavailable(ValueError):
    """Operator-facing; the pilot continues without reranking."""


class CrossEncoderReranker:
    name = 'cross-encoder'

    def __init__(self, model: str = DEFAULT_MODEL, device: str | None = None, max_length: int = 512,
                 allow_download: bool = False, *, _encoder=None):
        cache_namespace()
        self._model_scope = current_model_scope()
        self._model_mode = model_mode()
        if self._model_mode == 'server' and allow_download:
            raise ModelPolicyError('Server rerankers require local weights; model downloads are disabled')
        self.allow_download = allow_download
        if _encoder is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RerankUnavailable('sentence-transformers is not installed; the RRF order is used') from exc
            try:
                _encoder = CrossEncoder(model, max_length=max_length, device=device or None, local_files_only=not allow_download)
            except Exception as exc:  # the library raises many types for a missing or corrupt model
                raise RerankUnavailable(f'reranker model "{model}" is not available locally ({type(exc).__name__}); the RRF order is used'
                                        + (' — set RERANK_ALLOW_DOWNLOAD=on to fetch it once'
                                           if not allow_download and self._model_mode == 'development' else '')) from exc
        self.encoder = _encoder
        self.model = model
        self.max_length = max_length
        self.device = str(getattr(self.encoder, 'device', device or 'cpu'))
        self._model_identity = (self.model, self.max_length, self.device)
        self.calls = []

    def manifest(self) -> dict:
        self._validate_context()
        return {'method': VERSION, 'model': self.model, 'max_length': self.max_length, 'device': self.device,
                'candidates': RERANK_CANDIDATES, 'note': 'Cross-encoder relevance orders passages for the judge; it is not evidence.'}

    def _validate_context(self):
        namespace = cache_namespace()
        if self._model_scope != current_model_scope() or self._model_mode != model_mode():
            raise ModelPolicyError('Reranker instances cannot be reused across model scopes or modes')
        if self._model_mode == 'server' and self.allow_download:
            raise ModelPolicyError('Server rerankers require local weights; model downloads are disabled')
        if self._model_identity != (self.model, self.max_length, self.device):
            raise ModelPolicyError('Reranker model identity changed after construction')
        return namespace

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance of each passage to the query, in [0, 1] (the model's sigmoid output)."""
        self._validate_context()
        if not texts:
            return []
        started = time.monotonic()
        scores = self.encoder.predict([(query, text) for text in texts], batch_size=8, show_progress_bar=False)
        self.calls.append({'pairs': len(texts), 'elapsed_ms': int((time.monotonic() - started) * 1000)})
        return [float(score) for score in scores]


def cached_scores(reranker, query: str, texts: list[str]):
    """(scores, cache hits) for `texts` against `query`; only pairs not scored before reach the reranker.

    A reranker without a model name (a test double, an ad-hoc scorer) has no identity to key on
    and is always asked. A result of the wrong length or with a non-finite score raises before
    anything is stored, so a fault is never served from memory. Hits are appended to the
    reranker's own `calls` as {'pairs': 0, 'cache_hits': n, 'elapsed_ms': 0}.
    """
    namespace = cache_namespace()
    if isinstance(reranker, CrossEncoderReranker):
        reranker._validate_context()
    model = getattr(reranker, 'model', None)
    if not isinstance(model, str) or not model:
        return list(reranker.score(query, texts)), 0
    head = (model, namespace, getattr(reranker, 'max_length', None), sha256(query.encode('utf-8')).hexdigest())
    keys = [(*head, sha256(text.encode('utf-8')).hexdigest()) for text in texts]
    known = {key: _SCORE_CACHE[key] for key in keys if key in _SCORE_CACHE}
    missing = {key: text for key, text in zip(keys, texts) if key not in known}
    if missing:
        fresh = [float(score) for score in reranker.score(query, list(missing.values()))]
        if len(fresh) != len(missing) or any(not math.isfinite(score) for score in fresh):
            raise ValueError('reranker returned an unexpected result')
        known.update(zip(missing, fresh))
        _SCORE_CACHE.update(zip(missing, fresh))
        if len(_SCORE_CACHE) > SCORE_CACHE_LIMIT:
            for stale in list(_SCORE_CACHE)[:len(_SCORE_CACHE) - SCORE_CACHE_LIMIT]:
                _SCORE_CACHE.pop(stale, None)
    hits = sum(1 for key in keys if key not in missing)
    if hits and isinstance(getattr(reranker, 'calls', None), list):
        reranker.calls.append({'pairs': 0, 'cache_hits': hits, 'elapsed_ms': 0})
    return [known[key] for key in keys], hits


def configured_reranker():
    """Optional local reranker; missing weights fall back, policy violations fail closed."""
    model = (os.getenv('RERANK_MODEL') or '').strip()
    if not model or model.lower() in ('off', 'none', '0'):
        return None, {'status': 'off', 'reason': 'RERANK_MODEL is not configured'}
    allow = (os.getenv('RERANK_ALLOW_DOWNLOAD') or '').strip().lower() == 'on'
    device = (os.getenv('RERANK_DEVICE') or '').strip() or None
    cache_namespace()
    mode = model_mode()
    if mode == 'server' and allow:
        raise ModelPolicyError('Server rerankers require local weights; model downloads are disabled')
    key = (model, device, allow, mode, current_model_scope())
    encoder_key = (model, device, allow, 512)
    try:
        instance = _INSTANCES.get(key)
        if instance is None:
            instance = CrossEncoderReranker(model, device=device, allow_download=allow,
                                           _encoder=_ENCODERS.get(encoder_key))
            # Retain only the current configuration's weights between runs.
            _ENCODERS.clear()
            _ENCODERS[encoder_key] = instance.encoder
            _INSTANCES[key] = instance
        return instance, {'status': 'active', 'model': model, 'device': instance.device}
    except RerankUnavailable as exc:
        return None, {'status': 'unavailable', 'model': model, 'reason': str(exc)}
