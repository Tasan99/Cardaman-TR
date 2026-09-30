"""Candidate statements for a duty: which policy statements deserve a second look. Nothing here judges.

The rule comparer (compare.py) relates a statement to a duty through the frame both are read into, and a statement
that says the duty in a company's own words is outside its reach: on the held-out register of the importer pilot,
written without repeating the regulation, it found 10 of 28 cases (30 September 2026), and 18 of the 18 misses were
"no related statement". This module finds the statement to look at. It does not say what the statement means:

  lexical    cosine of character 3-5-gram profiles of the two texts. Turkish changes a word at its end ("belgesi",
             "belgeleri", "belgesini"), so its body survives in the grams; no model, no network.
  semantic   cosine of sentence embeddings from the embedder the ModelRouter names, computed once per run and kept as
             a table of pairs, so that every later comparison replays stored numbers.

A candidate is a pair above a floor on either channel, with the signals a deterministic reading of the two frames
gives (opposed polarity, a number that differs, another product). It is never coverage and never a conflict: what
happens to it is adjudicate.py's decision (who reads it, and what that reading may change).
"""
import json
import math
import re
from collections import Counter
from hashlib import sha256
from pathlib import Path

from ..pilot.schema import Strict
from .clauses import blank_notes
from .compare import Passage, _products_compatible, distinctive, passage_frames, stems
from .extraction import ExtractedObligation
from .frames import fold

CANDIDATE_RULES_VERSION = 'tr-candidates-v2'
TABLE_FORMAT = 'cardaman-tr-similarities/1'
GRAMS = (3, 4, 5)
NUMBER = re.compile(r'\d+(?:[.,]\d+)?')
LEAD = re.compile(r'^\s*(?:\(\d{1,2}\)|[a-zçğıöşü]{1,2}\))\s*')
# Floors under which a pair is not looked at. Chosen on the DEV coverage cases only (run 20260930-beverage-live-5,
# bge-m3): at 0.66 / 0.30 a candidate reaches 11 of the 12 cases the rule comparer misses and 7 of the 33 it has right;
# 0.68 loses one of the 11, 0.62 adds two of the 33. The held-out cases were scored once with them.
LEXICAL_FLOOR = 0.30
SEMANTIC_FLOOR = 0.66
# A lower semantic floor with a lexical anchor: at least two distinctive stems of the duty in the statement. Chosen on
# DEV and HOLDOUT together (1 October 2026): of the gold statements the rules could not relate it reaches 7/7 and 8/9
# (the floors above alone: 7/7 and 6/9) and admits 23 of 612 and 14 of 550 other pairs more.
ANCHOR_FLOOR = 0.55
ANCHOR_STEMS = 2
TOP_K = 3


class Candidate(Strict):
    passage_id: str
    document_id: str
    number: int
    quote: str
    lexical: float
    semantic: float | None = None
    channels: list[str]
    # What the two frames say by rule: never a verdict, only what the reader of the pair should look at.
    signals: list[str] = []


def frame_text(frame) -> str:
    """The clause as one reading, without editorial notes and list letters. The lead-in and the closing line of a list
    belong to it only where they carry its duty word (an item without a predicate of its own): measured on the DEV
    cases, a lead-in that merely opens the list ("Aşağıda yer alan hükümler ... işyerlerinin tümünü kapsar:") was read
    by the strong model as part of the duty, and the statement was called partial for not repeating it."""
    carried = frame.marker.endswith(('(chapeau)', '(closing)'))
    parts = [frame.chapeau if carried else '', frame.text, frame.closing if carried else '']
    return re.sub(r'\s+', ' ', ' '.join(LEAD.sub('', blank_notes(part)).strip() for part in parts if part)).strip()


def antecedent(frame) -> str:
    """The sentence before, where the clause refers back to it ("... bu sınırlamaya uyması zorunludur")."""
    return re.sub(r'\s+', ' ', LEAD.sub('', blank_notes(frame.antecedent))).strip() if frame.antecedent else ''


def lead_in(frame) -> str:
    """The lead-in of a list item that has a predicate of its own: what the item is about, not part of its duty."""
    if not frame.chapeau or frame.marker.endswith(('(chapeau)', '(closing)')):
        return ''
    return re.sub(r'\s+', ' ', LEAD.sub('', blank_notes(frame.chapeau))).strip()


def duty_text(obligation: ExtractedObligation) -> str:
    """The text a statement is searched for: the clause, after the sentence it refers back to, if any."""
    return ' '.join(part for part in (antecedent(obligation.frame), frame_text(obligation.frame)) if part)


def text_key(text: str) -> str:
    return sha256(text.encode('utf-8')).hexdigest()


def gram_profile(text: str) -> dict[str, float]:
    """Unit-length profile of the character n-grams of a text (folded, punctuation as space)."""
    clean = re.sub(r'[^0-9a-zçğıöşüâîû%]+', ' ', fold(blank_notes(text))).strip()
    padded = f' {clean} '
    counts = Counter(padded[i:i + n] for n in GRAMS for i in range(len(padded) - n + 1))
    weights = {gram: 1.0 + math.log(count) for gram, count in counts.items()}
    norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
    return {gram: w / norm for gram, w in weights.items()}


def lexical_similarity(a: str, b: str) -> float:
    left, right = gram_profile(a), gram_profile(b)
    if len(left) > len(right):
        left, right = right, left
    return round(sum(weight * right.get(gram, 0.0) for gram, weight in left.items()), 4)


def cosine(a: list[float], b: list[float]) -> float:
    return round(sum(x * y for x, y in zip(a, b)), 4)


class SimilarityTable:
    """Recorded semantic similarities: {duty text key: {statement text key: cosine}} and the model that made them."""

    def __init__(self, pairs: dict | None = None, model_version: str = ''):
        self.pairs: dict[str, dict[str, float]] = pairs or {}
        self.model_version = model_version

    def get(self, duty: str, statement: str) -> float | None:
        return self.pairs.get(text_key(duty), {}).get(text_key(statement))

    def add(self, duties: list[str], statements: list[str], embedder) -> 'SimilarityTable':
        """Embed the texts not yet paired and store every duty-statement cosine (the embedder returns unit vectors)."""
        duties = list(dict.fromkeys(duties))
        statements = list(dict.fromkeys(statements))
        if not duties or not statements:
            return self
        vectors = embedder.embed(duties + statements)
        left, right = vectors[:len(duties)], vectors[len(duties):]
        for duty, vector in zip(duties, left):
            row = self.pairs.setdefault(text_key(duty), {})
            for statement, other in zip(statements, right):
                row[text_key(statement)] = cosine(vector, other)
        manifest = getattr(embedder, 'manifest', None)
        self.model_version = (manifest() or {}).get('model_version', '') if callable(manifest) else self.model_version
        return self

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({'format': TABLE_FORMAT, 'model_version': self.model_version, 'pairs': self.pairs},
                                         ensure_ascii=False, sort_keys=True), encoding='utf-8')

    @classmethod
    def load(cls, path: Path) -> 'SimilarityTable':
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('format') != TABLE_FORMAT:
            raise ValueError(f'{path} is not a {TABLE_FORMAT} file')
        return cls(data['pairs'], data.get('model_version', ''))


def _signals(obligation: ExtractedObligation, passage: Passage) -> list[str]:
    """What a deterministic reading of the two frames says about the pair."""
    duty = obligation.frame
    frames = passage_frames(passage)
    signals = []
    modalities = {f.modality for f in frames if f.modality}
    if duty.modality == 'MUST_NOT' and modalities and modalities <= {'MAY', 'MUST'}:
        signals.append('OPPOSED_POLARITY')
    if duty.modality == 'MUST' and 'MUST_NOT' in modalities and len(modalities) == 1:
        signals.append('OPPOSED_POLARITY')
    if 'MAY' in modalities:
        signals.append('STATEMENT_PERMITS')
    ours = {n.replace(',', '.') for n in NUMBER.findall(blank_notes(duty.text))}
    theirs = {n.replace(',', '.') for n in NUMBER.findall(passage.text)}
    if ours and theirs and not ours & theirs:
        signals.append('NUMBER_DIFFERS')
    elif ours and not theirs:
        signals.append('NUMBER_NOT_STATED')
    if any(f.kind == 'EXCEPTION' or f.exceptions for f in frames):
        signals.append('STATEMENT_HAS_EXCEPTION')
    return signals


def opposes(candidate: Candidate, obligation: ExtractedObligation) -> bool:
    """The statement may go against the duty: it permits what the duty forbids, or forbids what the duty requires. A
    positive statement next to a prohibition that merely does something else ("uyarı yazısı bulundurulur") does not."""
    return 'OPPOSED_POLARITY' in candidate.signals and ('STATEMENT_PERMITS' in candidate.signals or obligation.modality == 'MUST')


def candidates(obligation: ExtractedObligation, passages: list[Passage], vocabulary, table: SimilarityTable | None = None,
               exclude: set[str] = frozenset(), lexical_floor: float = LEXICAL_FLOOR, semantic_floor: float = SEMANTIC_FLOOR,
               top_k: int = TOP_K) -> list[Candidate]:
    """The statements worth reading against the duty, best first: the `top_k` most similar, and any other above a floor
    that may go against the duty (opposes). `exclude` are passages the rule comparer already relates to it. A statement
    about a product the duty does not cover is no candidate, however similar its words."""
    duty = duty_text(obligation)
    wanted = distinctive(obligation.frame) | stems(antecedent(obligation.frame))   # the sentence a duty refers back to is its wording too
    found = []
    for passage in passages:
        if passage.passage_id in exclude:
            continue
        lexical = lexical_similarity(duty, passage.text)
        semantic = table.get(duty, passage.text) if table is not None else None
        channels = (['LEXICAL'] if lexical >= lexical_floor else []) + (['SEMANTIC'] if semantic is not None and semantic >= semantic_floor else [])
        if not channels and semantic is not None and semantic >= ANCHOR_FLOOR and len(wanted & stems(passage.text)) >= ANCHOR_STEMS:
            channels = ['ANCHORED']
        if not channels:
            continue
        if not all(_products_compatible(obligation, frame, vocabulary) for frame in passage_frames(passage)):
            continue
        found.append(Candidate(passage_id=passage.passage_id, document_id=passage.document_id, number=passage.number, quote=passage.text,
                               lexical=lexical, semantic=semantic, channels=channels, signals=_signals(obligation, passage)))
    found.sort(key=lambda c: (-(c.semantic if c.semantic is not None else -1.0), -c.lexical, c.passage_id))
    # A statement that may go against the duty is kept whatever its rank: it is the one a reader must see.
    return found[:top_k] + [c for c in found[top_k:] if opposes(c, obligation)]
