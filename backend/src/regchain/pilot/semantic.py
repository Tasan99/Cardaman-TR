"""Local embedding retrieval for private policy passages.

Word overlap cannot see a policy that meets a duty in other words, and one vector for a
whole passage dilutes a single contradicting sentence inside a long procedure. Passages
are therefore scored by their best sentence as well as their whole text, then fused with
the lexical ranking so exact terms still count. Equal scores share a rank: the lexical
scorer produces many ties, and breaking them by hash order let chance decide what the
model saw.

This orders what the model and the reviewer read first. A similarity is not evidence of
coverage and is not a calibrated probability, so no score threshold is applied.
"""
import math
import os
import re
from fractions import Fraction
from hashlib import sha256

import httpx

from regchain.extraction.grounding import MODAL
from regchain.model_router import (ModelPolicyError, cache_namespace, current_model_scope, model_mode,
                                   model_digests_match, model_names_match,
                                   validate_inventory_entry, validate_model, validate_ollama_endpoint)
from .policies import lexical_score, lexical_terms
from .rerank import RERANK_CANDIDATES, cached_scores

VERSION = 'hybrid-rrf-v1'
RERANKED_VERSION = 'hybrid-rrf-rerank-v1'
RRF_K = 60
BATCH = 32
# Above this the per-sentence pass is skipped and recorded as skipped, never silently.
MAX_SENTENCES = 4000
RUNNERS_UP = 4
# The engine asks for one duty's order several times in a row (ranked, scores, select); it is
# computed once. Bounded: each entry holds a few lists as long as the passage set.
ORDER_MEMO = 64
# A fused rank can suppress one retrieval channel. Keep the existing judge window and
# add a bounded, independent contribution from each channel; never replace its head.
UNION_SEMANTIC_LIMIT = 10
UNION_LEXICAL_LIMIT = 10
SENTENCE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'(])')
HEADING = re.compile(r'\s{0,3}#{1,6}\s')
MODALITY = {'MUST': 'must', 'MUST_NOT': 'must not', 'SHOULD': 'should', 'SHOULD_NOT': 'should not',
            'MAY': 'may', 'MAY_NOT': 'may not'}
# Embeddings are deterministic for a pinned model, so a passage seen in an earlier run of this
# process is not embedded again (the second run of the same policy set costs no embedding time).
EMBED_CACHE_LIMIT = 20000
_EMBED_CACHE = {}
# v0.16 evidence gate, structural only. Measured live: a PDF running header fragment ("Kurum İçi
# Sınırsız Kullanım / Kişisel Veri"), a torn line ("edürleri, • Programın ... denetimi,") and page
# numbers reached the judge and were judged UNCLEAR, which blocked a favourable verdict on their
# own. A similarity floor is deliberately NOT applied by default: a contradiction rarely shares the
# duty's words and was measured to rank low (9 to 18 of 18), so a floor would hide exactly what
# the isolated contradiction check exists to find. EVIDENCE_MIN_SIMILARITY turns one on.
PAGE_CRUMB = re.compile(r'^[\s\d./–\-—|:]+$|^(?:sayfa|page)\s*\d+(?:\s*/\s*\d+)?$', re.I)
SENTENCE_END = re.compile(r'[.!?;:]["”’)]?\s*$')
SENTENCE_MARK = re.compile(r'[.!?;:]')
HEADING_WORDS_MAX = 8
TORN_WORDS_MAX = 14


class EmbeddingFailure(ValueError):
    """Operator-facing; never carries policy text."""


class OllamaEmbedder:
    name = 'ollama'

    def __init__(self, model: str, model_digest: str, base_url: str, timeout: float = 120, cache: bool = False):
        # One central policy guards chat and private embeddings alike. Validate before
        # constructing a client or looking in a process-wide private-content cache.
        origin = validate_ollama_endpoint(base_url)
        validate_model(model, model_digest, kind='embedding')
        cache_namespace()
        self._model_scope = current_model_scope()
        self._model_mode = model_mode()
        if not 10 <= timeout <= 3600:
            raise ValueError('EMBED_TIMEOUT_SECONDS must be between 10 and 3600')
        self.model = model
        self.model_digest = model_digest
        self.model_version = model + '@' + model_digest
        self.timeout = timeout
        self.base_url = origin
        self._model_identity = (model, model_digest, origin)
        # The process-wide vector cache is used only when asked for (the workspace asks); a
        # direct instance embeds every text it is given, which is what a measurement expects.
        self.use_cache = cache
        self.cache_hits = 0

    def manifest(self):
        self._validate_context()
        return {'provider': self.name, 'model_version': self.model_version, 'adapter': 'ollama-embed-v1'}

    def _validate_context(self):
        namespace = cache_namespace()
        if self._model_scope != current_model_scope() or self._model_mode != model_mode():
            raise ModelPolicyError('Embedding instances cannot be reused across model scopes or modes')
        validate_ollama_endpoint(self.base_url)
        validate_model(self.model, self.model_digest, kind='embedding')
        if (self._model_identity != (self.model, self.model_digest, self.base_url)
                or self.model_version != self.model + '@' + self.model_digest):
            raise ModelPolicyError('Embedding model identity changed after construction')
        return namespace

    def embed(self, texts):
        namespace = self._validate_context()
        if not texts:
            return []
        keys = [(self.model_version, namespace, sha256(text.encode('utf-8')).hexdigest()) for text in texts]
        if self.use_cache:
            missing = [text for text, key in dict(zip(texts, keys)).items() if key not in _EMBED_CACHE]
        else:
            missing = list(texts)
        self.cache_hits += len(texts) - len(missing)
        vectors = []
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=False, trust_env=False) as client:
                # Verify pinned local model identity rather than trusting a mutable tag.
                tags = client.get(self.base_url + '/api/tags')
                tags.raise_for_status()
                matching = [item for item in tags.json().get('models', [])
                            if isinstance(item, dict) and model_names_match(item.get('name'), self.model)]
                for item in matching:
                    validate_inventory_entry(item)
                if not any(model_digests_match(item.get('digest'), self.model_digest) for item in matching):
                    raise EmbeddingFailure('Configured embedding model digest does not match installed Ollama model')
                for start in range(0, len(missing), BATCH):
                    batch = missing[start:start+BATCH]
                    # truncate=False: an over-long passage is an error, not a silently shortened vector.
                    response = client.post(self.base_url + '/api/embed',
                        json={'model': self.model, 'input': batch, 'truncate': False})
                    response.raise_for_status()
                    rows = response.json().get('embeddings')
                    if not isinstance(rows, list) or len(rows) != len(batch):
                        raise EmbeddingFailure('Embedding provider returned an unexpected number of vectors')
                    vectors.extend(rows)
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, (EmbeddingFailure, ModelPolicyError)):
                raise
            raise EmbeddingFailure('Ollama embedding request failed; check service, model and configuration') from exc
        if not self.use_cache:
            dimension = len(vectors[0]) if isinstance(vectors[0], list) else 0
            return [unit(vector, dimension) for vector in vectors]
        if vectors:
            dimension = len(vectors[0]) if isinstance(vectors[0], list) else 0
            fresh = [unit(vector, dimension) for vector in vectors]
            for text, vector in zip(missing, fresh):
                _EMBED_CACHE[(self.model_version, namespace, sha256(text.encode('utf-8')).hexdigest())] = vector
            if len(_EMBED_CACHE) > EMBED_CACHE_LIMIT:
                for stale in list(_EMBED_CACHE)[:len(_EMBED_CACHE) - EMBED_CACHE_LIMIT]:
                    del _EMBED_CACHE[stale]
        return [_EMBED_CACHE[key] for key in keys]


def unit(vector, dimension):
    if (not isinstance(vector, list) or len(vector) != dimension or not dimension
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector)):
        raise EmbeddingFailure('Embedding provider returned a malformed vector')
    norm = math.sqrt(math.sumprod(vector, vector))
    if not norm:
        raise EmbeddingFailure('Embedding provider returned a zero vector')
    return [x / norm for x in vector]


def configured_embedder():
    try:
        timeout = float(os.getenv('EMBED_TIMEOUT_SECONDS') or 120)
    except ValueError as exc:
        raise ValueError('EMBED_TIMEOUT_SECONDS must be a number of seconds') from exc
    if not (os.getenv('EMBED_MODEL') or '').strip():
        raise ValueError('Anlamsal arama için EMBED_MODEL yapılandırılmamış; '
                         'yalnızca kelime eşleşmesini seç veya yerel bir embedding modeli kur.')
    return OllamaEmbedder(os.getenv('EMBED_MODEL', ''), os.getenv('EMBED_MODEL_DIGEST', ''),
                          os.getenv('OLLAMA_BASE_URL', 'http://host.docker.internal:11434'), timeout, cache=True)


def heading_only(chunk) -> bool:
    # A Markdown heading names a section; it cannot state a control. The blank-line
    # chunker makes each one a passage, and they crowd real text out of a small budget.
    lines = [line for line in chunk['text'].splitlines() if line.strip()]
    return chunk['filename'].lower().endswith('.md') and bool(lines) and all(HEADING.match(line) for line in lines)


def judgeable(chunk) -> bool:
    """A passage that could state a control: not a heading, and at least three words.

    PDF extraction leaves crumbs such as a lone "R" marker or half a word cut at a block
    boundary. The same rule decides what is indexed, what the judge reads and what the
    review screen counts as unread, so the three never disagree.
    """
    return not heading_only(chunk) and len(chunk['text'].split()) >= 3


def relevance_floor() -> float:
    """Optional similarity floor for the evidence gate; 0 (the default) means no floor."""
    try:
        return max(0.0, min(1.0, float(os.getenv('EVIDENCE_MIN_SIMILARITY') or 0)))
    except ValueError:
        return 0.0


def evidence_gate(chunk, similarity=None, lexical=None) -> str | None:
    """Why a retrieved passage is kept from the judge, or None when it is worth judging.

    Structural crumbs only, by rule: a page number or date line, a heading-like line (few
    words, no sentence end, no duty wording) and a torn short fragment that starts mid-word
    or mid-sentence. A control register row is never a crumb. The optional similarity floor
    applies only to a passage that also shares no word with the duty.
    """
    text = ' '.join(chunk['text'].split())
    if chunk.get('locator') == 'control_row':
        return None
    if PAGE_CRUMB.match(text):
        return 'PAGE_CRUMB'
    words = text.split()
    ends = bool(SENTENCE_END.search(text))
    modal = bool(MODAL.search(text))
    # A heading has no sentence punctuation at all; a short line with a full stop inside is text.
    if len(words) <= HEADING_WORDS_MAX and not SENTENCE_MARK.search(text) and not modal:
        return 'HEADING_FRAGMENT'
    if len(words) <= TORN_WORDS_MAX and not ends and not modal and text[:1].islower():
        return 'TORN_FRAGMENT'
    floor = relevance_floor()
    if floor and similarity is not None and similarity < floor and not lexical:
        return 'LOW_RELEVANCE'
    return None


def shared_ranks(scores):
    """Competition ranking: equal scores share the best rank."""
    first = {}
    for index, score in enumerate(sorted(scores, reverse=True), 1):
        first.setdefault(score, index)
    return [first[score] for score in scores]


def candidate_union(order, semantic_order, lexical_order, limit,
                    semantic_limit=UNION_SEMANTIC_LIMIT, lexical_limit=UNION_LEXICAL_LIMIT):
    """Add channel candidates without changing the existing head or admitting new IDs.

    ``lexical_order`` must contain only passages with positive lexical overlap. The
    returned pool can exceed ``limit``: this is a candidate expansion, not a top-k
    reranking claim. Added passages follow the original fused/reranked order.
    """
    if any(type(n) is not int or n < 0 for n in (limit, semantic_limit, lexical_limit)):
        raise ValueError('Candidate window limits must be non-negative integers')
    head = list(dict.fromkeys(order[:limit]))
    members = set(semantic_order[:semantic_limit]) | set(lexical_order[:lexical_limit])
    seen = set(head)
    return list(dict.fromkeys([*head, *(i for i in order if i in members and i not in seen)]))


def query_text(candidate) -> str:
    action = candidate.get('required_action') or candidate.get('prohibited_action') or ''
    parts = [*candidate.get('conditions', []), candidate.get('subject', ''),
             MODALITY.get(candidate.get('modality'), ''), action, *candidate.get('exceptions', [])]
    return ' '.join(part.strip() for part in parts if part and part.strip())


class PolicyIndex:
    """Embeds every passage once per analysis; each duty then costs one query vector.

    With a reranker (v0.16) the leading RRF ranks are rescored by a cross-encoder for every
    duty; a reranker fault for one duty is recorded and that duty keeps the RRF order. Scores
    of a pair seen in an earlier analysis come from the process-wide cache (v0.18) and are
    counted in rerank_cache_hits.
    """

    def __init__(self, policies, embedder, reranker=None):
        cache_namespace()
        self._model_scope = current_model_scope()
        self._model_mode = model_mode()
        self.embedder = embedder
        self.reranker = reranker
        self.rerank_cache = {}
        self.rerank_failures = []
        self.rerank_cache_hits = 0
        self.queries = {}
        self.orders = {}
        chunks = [chunk for policy in policies for chunk in policy['chunks']]
        titled = [chunk for chunk in chunks if not heading_only(chunk)]
        self.candidates = [chunk for chunk in titled if judgeable(chunk)]
        self.headings_excluded = len(chunks) - len(titled)
        self.fragments_excluded = len(titled) - len(self.candidates)
        sentences, owners = [], []
        for index, chunk in enumerate(self.candidates):
            for sentence in SENTENCE.split(chunk['text']):
                if sentence != chunk['text'] and len(sentence.split()) >= 4:
                    sentences.append(sentence)
                    owners.append(index)
        self.sentence_level = len(sentences) <= MAX_SENTENCES
        if not self.sentence_level:
            sentences, owners = [], []
        vectors = embedder.embed([chunk['text'] for chunk in self.candidates] + sentences)
        self.dimension = len(vectors[0]) if vectors else 0
        if any(len(vector) != self.dimension for vector in vectors):
            raise EmbeddingFailure('Embedding provider returned vectors of different sizes')
        self.vectors = [[vector] for vector in vectors[:len(self.candidates)]]
        for owner, vector in zip(owners, vectors[len(self.candidates):]):
            self.vectors[owner].append(vector)
        self.sentences = len(sentences)

    def _validate_context(self):
        cache_namespace()
        if self._model_scope != current_model_scope() or self._model_mode != model_mode():
            raise ModelPolicyError('Policy indexes cannot be reused across model scopes or modes')

    def manifest(self):
        self._validate_context()
        reranker = ({'status': 'active', **self.reranker.manifest(), 'failures': len(self.rerank_failures)} if self.reranker is not None
                    else {'status': 'off'})
        return {'method': RERANKED_VERSION if self.reranker is not None else VERSION, 'embedding': self.embedder.manifest(),
                'dimension': self.dimension,
                'rrf_k': RRF_K, 'passages_indexed': len(self.candidates), 'headings_excluded': self.headings_excluded,
                'fragments_excluded': self.fragments_excluded,
                'sentence_level': self.sentence_level, 'sentences_indexed': self.sentences,
                'reranker': reranker,
                'candidate_expansion': {'method': 'additive-channel-union-v1',
                                        'semantic_limit': UNION_SEMANTIC_LIMIT, 'lexical_limit': UNION_LEXICAL_LIMIT,
                                        'preserves_existing_head': True, 'zero_overlap_lexical_excluded': True},
                'evidence_gate': {'version': 'structural-v1', 'similarity_floor': f'{relevance_floor():.2f}'},
                'note': 'Similarity orders passages; it is neither coverage evidence nor a calibrated probability.'}

    def prepare(self, candidates):
        """Embed every duty's query in one batch; select() then needs no embedder call."""
        self._validate_context()
        missing = [text for text in dict.fromkeys(query_text(c) for c in candidates) if text not in self.queries]
        if missing and self.candidates:
            self.queries.update(zip(missing, self.embedder.embed(missing)))

    def rerank(self, query, ranked):
        """Cross-encoder scores for the leading fused ranks, by candidate index; {} without a reranker."""
        self._validate_context()
        if self.reranker is None or not ranked:
            return {}
        top = tuple(ranked[:RERANK_CANDIDATES])
        key = (query, top)
        if key not in self.rerank_cache:
            try:
                scores, hits = cached_scores(self.reranker, query, [self.candidates[i]['text'] for i in top])
                if len(scores) != len(top) or any(not math.isfinite(s) for s in scores):
                    raise ValueError('reranker returned an unexpected result')
                self.rerank_cache[key] = dict(zip(top, scores))
                self.rerank_cache_hits += hits
            except ModelPolicyError:
                raise
            except Exception as exc:  # a reranker fault must never end a run; the RRF order stands
                self.rerank_failures.append(type(exc).__name__)
                self.rerank_cache[key] = {}
        return self.rerank_cache[key]

    def order(self, candidate):
        """(ranked, similarity, semantic_rank, lexical_rank, lexical, fused, rerank) for one duty.

        Memoised per query text and action (the two inputs of the ranking); callers get copies,
        so the result is the same whichever caller asked first.
        """
        self._validate_context()
        key = (query_text(candidate), candidate.get('required_action') or candidate.get('prohibited_action') or '')
        if key not in self.orders:
            self.orders[key] = self._order(candidate)
            for stale in list(self.orders)[:-ORDER_MEMO]:
                del self.orders[stale]
        ranked, similarity, semantic_rank, lexical_rank, lexical, fused, rerank = self.orders[key]
        return list(ranked), list(similarity), list(semantic_rank), list(lexical_rank), list(lexical), list(fused), dict(rerank)

    def _order(self, candidate):
        self.prepare([candidate])
        query = self.queries[query_text(candidate)]
        if len(query) != self.dimension:
            raise EmbeddingFailure('Query and passage embeddings have different sizes')
        similarity = [max(math.sumprod(query, vector) for vector in group) for group in self.vectors]
        terms = lexical_terms(candidate.get('required_action') or candidate.get('prohibited_action') or '')
        lexical = [lexical_score(terms, chunk['text']) for chunk in self.candidates]
        semantic_rank, lexical_rank = shared_ranks(similarity), shared_ranks(lexical)
        fused = [Fraction(1, RRF_K + semantic_rank[i]) + (Fraction(1, RRF_K + lexical_rank[i]) if lexical[i] else 0)
                 for i in range(len(self.candidates))]
        ranked = sorted(range(len(self.candidates)),
                        key=lambda i: (-fused[i], -similarity[i], self.candidates[i]['source_id']))
        rerank = self.rerank(query_text(candidate), ranked)
        if rerank:
            head = sorted(rerank, key=lambda i: (-rerank[i], -fused[i], self.candidates[i]['source_id']))
            ranked = [*head, *ranked[len(head):]]
        return ranked, similarity, semantic_rank, lexical_rank, lexical, fused, rerank

    def ranked(self, candidate):
        """Every candidate passage, best first; the isolated conflict check reads past the shown ones."""
        self._validate_context()
        return [self.candidates[i] for i in self.order(candidate)[0]] if self.candidates else []

    def candidate_window(self, candidate, limit, semantic_limit=UNION_SEMANTIC_LIMIT,
                         lexical_limit=UNION_LEXICAL_LIMIT):
        """Existing leading ranks plus independent semantic/positive-lexical candidates.

        Structural filtering and scoped query preparation are exactly those of ranked().
        This does not change scores(), ranked(), select() or the evidence gate. A caller
        that reads all small-corpus passages should keep doing so. Control-row expansion
        remains the caller's responsibility and must not be replaced by this union.
        """
        self._validate_context()
        if not self.candidates:
            candidate_union([], [], [], limit, semantic_limit, lexical_limit)
            return []
        order, similarity, _, _, lexical, _, _ = self.order(candidate)
        semantic_order = sorted(order, key=lambda i: (-similarity[i], self.candidates[i]['source_id']))
        lexical_order = sorted((i for i in order if lexical[i] > 0),
                               key=lambda i: (-lexical[i], self.candidates[i]['source_id']))
        union = candidate_union(order, semantic_order, lexical_order, limit, semantic_limit, lexical_limit)
        return [self.candidates[i] for i in union]

    def scores(self, candidate):
        """Every retrieval signal per passage, for the debug view: {source_id: row}."""
        self._validate_context()
        if not self.candidates:
            return {}
        order, similarity, semantic_rank, lexical_rank, lexical, fused, rerank = self.order(candidate)
        rerank_rank = {i: position for position, i in enumerate(sorted(rerank, key=lambda i: -rerank[i]), 1)}
        rows = {}
        for position, i in enumerate(order, 1):
            rows[self.candidates[i]['source_id']] = {
                'rank': position, 'similarity': f'{similarity[i]:.4f}', 'semantic_rank': semantic_rank[i],
                'lexical_score': lexical[i], 'lexical_rank': lexical_rank[i] if lexical[i] else None,
                'rrf': f'{float(fused[i]):.5f}',
                'rerank_score': f'{rerank[i]:.4f}' if i in rerank else None, 'rerank_rank': rerank_rank.get(i)}
        return rows

    def select(self, candidate, limit=6, max_chars=6500):
        self._validate_context()
        if not self.candidates:
            return [], {'shown': [], 'not_shown': []}
        order, similarity, semantic_rank, lexical_rank, lexical, fused, rerank = self.order(candidate)
        rerank_rank = {i: position for position, i in enumerate(sorted(rerank, key=lambda i: -rerank[i]), 1)}

        def row(position, i):
            # Evidence forbids floats: a similarity is persisted as a decimal string.
            return {'source_id': self.candidates[i]['source_id'], 'rank': position,
                    'semantic_rank': semantic_rank[i], 'lexical_rank': lexical_rank[i] if lexical[i] else None,
                    'similarity': f'{similarity[i]:.4f}', 'lexical_score': lexical[i], 'rrf': f'{float(fused[i]):.5f}',
                    'rerank_score': f'{rerank[i]:.4f}' if i in rerank else None, 'rerank_rank': rerank_rank.get(i)}
        chosen, shown, not_shown, used = [], [], [], 0
        # Only the leading ranks compete. Without this window a large top passage left
        # room that was filled from the bottom of the ranking: on a 118-page PDF the
        # model was shown rank 184, a single letter, because it happened to fit.
        for position, i in enumerate(order[:limit + RUNNERS_UP], 1):
            chunk = self.candidates[i]
            if len(chosen) < limit and used + len(chunk['text']) <= max_chars:
                chosen.append(chunk)
                shown.append(row(position, i))
                used += len(chunk['text'])
            else:
                # What the model did NOT read, so a reviewer can see the near misses.
                not_shown.append(row(position, i))
        return chosen, {'shown': shown, 'not_shown': not_shown}
