"""Amendment notes of the consolidated text, read per clause.

The Mevzuat Bilgi Sistemi prints, inside the consolidated wording, what each amending instrument did and when:
"(Ek cümle:11/6/2026-7584/2 md.)", "(Değişik:RG-20/9/2023-32315)", "(Mülga: 27/3/1969-1137/37-1 md.)". These notes are
the official text's own record of change, so "which clauses changed since a date" can be answered from one stored
version, without a second one to diff against (change.py uses both).

A note belongs to the unit it names: a sentence note to the sentence that follows it, a fıkra note (or a note with no
unit at the head of a fıkra) to every clause of the fıkra, a note on its own line at the head of an article to the
whole article. The old wording is not in the text; a note says that and when a clause changed, never what it said.
"""
import re
from datetime import date
from typing import Literal

from ..pilot.schema import Strict
from .clauses import NOTE, Clause, blank_notes, split_clauses

ACTIONS = {'Ek': 'ADDED', 'Değişik': 'CHANGED', 'Mülga': 'REPEALED', 'İptal': 'ANNULLED', 'Yeniden düzenleme': 'REENACTED'}
UNITS = (('cümle', 'SENTENCE'), ('fıkra', 'FIKRA'), ('bent', 'BENT'), ('madde', 'ARTICLE'), ('ibare', 'PHRASE'), ('ifade', 'PHRASE'),
         ('başlık', 'HEADING'), ('paragraf', 'FIKRA'))
PART = re.compile(r'(Ek|Değişik|Mülga|İptal|Yeniden düzenleme)\s*([a-zçğıöşü ]{0,30}?)\s*:\s*(?:R\.?\s?G\.?\s*[-:.]?\s*)?'
                  r'(\d{1,2})/(\d{1,2})/(\d{4})\s*[-–]?\s*([^;)]*)')
MARKER = re.compile(r'^\s*(?:\(\d{1,2}\)|[a-zçğıöşü]{1,2}\))?\s*$')


class AmendmentNote(Strict):
    action: Literal['ADDED', 'CHANGED', 'REPEALED', 'ANNULLED', 'REENACTED']
    unit: Literal['ARTICLE', 'FIKRA', 'BENT', 'SENTENCE', 'PHRASE', 'HEADING', 'UNSPECIFIED']
    date: date
    instrument: str
    text: str
    start: int
    end: int


def notes_in(text: str, base: int = 0) -> list[AmendmentNote]:
    """Every dated note in a text; one parenthesis may hold several ("Mülga: ...; Yeniden düzenleme: ...")."""
    out = []
    for note in NOTE.finditer(text):
        for part in PART.finditer(note.group()):
            try:
                when = date(int(part.group(5)), int(part.group(4)), int(part.group(3)))
            except ValueError:
                continue
            word = part.group(2).strip()
            unit = next((name for stem, name in UNITS if stem in word), 'UNSPECIFIED')
            out.append(AmendmentNote(action=ACTIONS[part.group(1)], unit=unit, date=when, instrument=part.group(6).strip(' .'),
                                     text=note.group(), start=base + note.start(), end=base + note.end()))
    return out


def clause_amendments(section: dict, clauses: list[Clause] | None = None) -> dict[str, list[AmendmentNote]]:
    """{clause ref: the notes that apply to it}, own and inherited."""
    clauses = split_clauses(section) if clauses is None else clauses
    text = section['text']
    out: dict[str, list[AmendmentNote]] = {c.ref: [] for c in clauses}
    if not clauses:
        return out
    lines = [(line['start'], line['end']) for line in section.get('lines') or []] or [(0, len(text))]
    note_only = [(a, b) for a, b in lines if not blank_notes(text[a:b]).strip()]
    first = min(c.start for c in clauses)

    def give(targets, note):
        for target in targets:
            if note not in out[target.ref]:
                out[target.ref].append(note)

    for note in notes_in(text):
        # A note on a line of its own above the wording: the whole article.
        if note.start < first and any(a <= note.start < b for a, b in note_only):
            give(clauses, note)
            continue
        inside = next((c for c in clauses if c.start <= note.start < c.end), None)
        clause = inside or next((c for c in clauses if c.start >= note.end), None)
        if clause is None:
            continue
        chapeau = next((c for c in clauses if c.chapeau_start is not None and c.chapeau_start <= note.start < c.chapeau_start + len(c.chapeau)), None)
        if inside is None and chapeau is not None:
            give([c for c in clauses if c.chapeau_start == chapeau.chapeau_start], note)      # a note in the lead-in of a list
            continue
        # Leading: nothing but a paragraph marker stands between the start of the clause and the note.
        leading = inside is None or bool(MARKER.match(blank_notes(text[clause.start:note.start])))
        if note.unit in ('SENTENCE', 'PHRASE', 'HEADING') or not leading or clause.sentence not in (None, 1):
            give([clause], note)
        elif clause.bent is not None and note.unit != 'FIKRA':
            give([c for c in clauses if (c.fikra, c.bent) == (clause.fikra, clause.bent)], note)
        else:
            give([c for c in clauses if c.fikra == clause.fikra], note)
    return out


def latest(notes: list[AmendmentNote]) -> AmendmentNote | None:
    return max(notes, key=lambda n: n.date) if notes else None
