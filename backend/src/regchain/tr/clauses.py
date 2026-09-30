"""Clause segmentation of a Turkish article: fıkra, bent and sentence, each an exact span of the stored text.

A duty sits in one sentence of one paragraph ("Kanun 4250 md. 6, beşinci fıkra, üçüncü cümle": no retail sale between
22:00 and 06:00), and its sanction, its exception and a later amendment all point at that place. The engine reads an
article as one text; this module cuts it where Turkish drafting cuts it:

  fıkra     a numbered paragraph "(3) ..." or, in older statutes, an unnumbered source paragraph (the parser's line spans)
  bent      a lettered item "a) ..." under a fıkra, with the fıkra's lead-in (chapeau) and closing predicate as context
  cümle     a sentence of a fıkra or a bent

A reference names the place the way a lawyer would: '<printed label>/f.3/b.c/c.2'. Every clause satisfies
article_text[start:end] == text, so a quote taken from a clause is a quote of the stored official text.
Nothing here decides what a clause means; frames.py reads the wording.
"""
import re
from typing import Literal

from ..pilot.schema import Strict

TR_LOWER = 'a-zçğıöşüâîû'
TR_UPPER = 'A-ZÇĞİÖŞÜÂÎÛ'
# Editorial notes of the consolidated text: "(Değişik:RG-18/9/2013-28769)", "(Ek cümle:11/6/2026-7584/2 md.)".
NOTE = re.compile(r'\(\s*(?:Ek|Değişik|Mülga|İptal|Yeniden düzenleme|Mükerrer|Ek ibare|Değişik ibare)[^()]*(?:\([^()]*\)[^()]*)*\)')
FIKRA = re.compile(r'^\((\d{1,2})\)\s*')
BENT = re.compile(r'^([a-zçğıöşü]{1,2})\)\s+')
SUB_BENT = re.compile(r'^(\d{1,2})\)\s+')
MARKER_ONLY = re.compile(r'\(?\d{1,2}\)|[a-zçğıöşü]{1,2}\)')
DASH = re.compile(r'^[-–—]\s*\S')
# A closing line finishes the sentence its bents opened: "yükümlüdürler.", "ifade eder.", "kapsamaz.", "den fazla olamaz."
CLOSING = re.compile(r'^[%s][^:]{0,120}[.]?$' % TR_LOWER)
# A sentence opens with a capital, a quotation, a bracket or a date ("1/1/2020 tarihinden önce ...").
SENTENCE_END = re.compile(r'[.!?;](?=\s+(?:\(\s*(?:Ek|Değişik|Mülga)[^()]*\)\s*)?(?:[%s“"‘(]|\d{1,2}/\d{1,2}/\d{4}))' % TR_UPPER)
ABBREVIATION = re.compile(r'(?:\b(?:md|No|no|vb|vs|bkz|Dr|Prof|sayılı|Sk|Cad|Tic|Ltd|Şti|A\.Ş|T\.C)|\b[A-ZÇĞİÖŞÜ]|\d)$')
QUOTES = {'“': '”', '"': '"', '‘': '’'}

Kind = Literal['FIKRA', 'BENT', 'SENTENCE']


class Clause(Strict):
    ref: str
    label: str
    heading: str = ''
    fikra: int
    bent: str | None = None
    sentence: int | None = None
    start: int
    end: int
    text: str
    # The lead-in of the fıkra a bent hangs from ("Piyasaya sürenler;") and the predicate that closes the list
    # ("yükümlüdürler."); both exact spans of the article, empty for a free-standing sentence.
    chapeau: str = ''
    chapeau_start: int | None = None
    closing: str = ''
    closing_start: int | None = None
    numbered: bool = True
    # The text with editorial notes blanked out (same length, same offsets): what the wording rules read.
    reading: str = ''

    def composed(self) -> str:
        """Chapeau, clause and closing as one reading (not a span of the article)."""
        return ' '.join(part for part in (self.chapeau, self.text, self.closing) if part)


def blank_notes(text: str) -> str:
    """The text with each editorial note replaced by spaces of the same length, so offsets survive."""
    return NOTE.sub(lambda m: ' ' * (m.end() - m.start()), text)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of each sentence. A full stop inside a quotation, inside brackets, after an abbreviation or
    between digits does not end a sentence; a quoted label warning ("“Alkol ile karıştırılarak ... tavsiye edilmez.”")
    stays inside the sentence that prescribes it."""
    masked = blank_notes(text)
    spans, start, depth, closing = [], 0, 0, None
    index = 0
    while index < len(masked):
        char = masked[index]
        if closing is not None:
            if char == closing:
                closing = None
        elif char in QUOTES and (char != '"' or masked.count('"', index + 1) > 0):
            closing = QUOTES[char]
        elif char == '(':
            depth += 1
        elif char == ')' and depth:
            depth -= 1
        elif depth == 0 and char in '.!?;' and SENTENCE_END.match(masked, index):
            # ';' separates clauses of one sentence in Turkish drafting; only a full stop, '!' or '?' ends a sentence.
            if char != ';' and not ABBREVIATION.search(masked[start:index]):
                spans.append((start, index + 1))
                start = index + 1
                while start < len(masked) and masked[start].isspace():
                    start += 1
                index = start
                continue
        index += 1
    if masked[start:].strip():
        spans.append((start, len(text)))
    out = []
    for a, b in spans:
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if masked[a:b].strip():
            out.append((a, b))
    return out


def _line_kind(line: str) -> tuple[str, str]:
    """('NOTE' | 'FIKRA' | 'BENT' | 'SUB' | 'DASH' | 'CLOSING' | 'PARAGRAPH', marker)."""
    bare = blank_notes(line).strip()
    if not bare or MARKER_ONLY.fullmatch(bare):
        return 'NOTE', ''                                # "c) (Mülga: RG-3/12/2013-28840)": a repealed item leaves its marker
    numbered = FIKRA.match(bare)
    if numbered:
        return 'FIKRA', numbered.group(1)
    lettered = BENT.match(bare)
    if lettered:
        return 'BENT', lettered.group(1)
    if SUB_BENT.match(bare):
        return 'SUB', SUB_BENT.match(bare).group(1)
    if DASH.match(bare):
        return 'DASH', ''
    if CLOSING.match(bare) and len(bare) <= 120:
        return 'CLOSING', ''
    return 'PARAGRAPH', ''


def _lines(section: dict) -> list[tuple[int, int]]:
    spans = [(line['start'], line['end']) for line in section.get('lines') or []]
    return spans or [(0, len(section['text']))]


def split_clauses(section: dict) -> list[Clause]:
    """The clauses of one stored provision, in document order."""
    text, label = section['text'], section['printed_label']
    heading = section['heading_path'][2] if len(section.get('heading_path') or []) >= 3 else ''
    numbered = any(_line_kind(text[a:b])[0] == 'FIKRA' for a, b in _lines(section))
    # -- group the source lines into fıkra blocks --------------------------------------------------
    blocks = []                                   # {'number', 'lead': [(a, b)], 'bents': [{'letter', 'spans'}], 'closing': (a, b)}
    current, previous = None, None
    for a, b in _lines(section):
        kind, marker = _line_kind(text[a:b])
        if kind == 'NOTE':
            continue
        if kind == 'FIKRA' or (kind == 'PARAGRAPH' and (not numbered or current is None)) or current is None:
            if kind in ('BENT', 'SUB', 'DASH', 'CLOSING') and current is None:
                kind = 'PARAGRAPH'
            number = int(marker) if kind == 'FIKRA' else len(blocks) + 1
            current = {'number': number, 'lead': [(a, b)], 'bents': [], 'closing': None}
            blocks.append(current)
        elif kind == 'BENT':
            if current['closing'] is not None:
                # What looked like the list's closing predicate was the end of the bent before this one
                # ("c) ... bileşiminde; -İnositol 100 mg/L ... den fazla olamaz." and then "ç) ...").
                current['bents'][-1]['spans'].append(current['closing'])
                current['closing'] = None
            current['bents'].append({'letter': marker, 'spans': [(a, b)]})
        elif kind in ('SUB', 'DASH'):
            (current['bents'][-1]['spans'] if current['bents'] else current['lead']).append((a, b))
        elif kind == 'CLOSING' and current['bents'] and previous not in ('SUB', 'DASH'):
            current['closing'] = (a, b) if current['closing'] is None else (current['closing'][0], b)
        else:
            # A continuation paragraph (or the predicate that closes a dash list): it stays with what it follows.
            (current['bents'][-1]['spans'] if current['bents'] else current['lead']).append((a, b))
        previous = kind
    # -- emit clauses ------------------------------------------------------------------------------
    clauses = []

    def emit(start, end, fikra, bent=None, chapeau=None, closing=None):
        spans = sentence_spans(text[start:end])
        for position, (a, b) in enumerate(spans, 1):
            sentence = position if len(spans) > 1 else None
            ref = f'{label}/f.{fikra}' + (f'/b.{bent}' if bent else '') + (f'/c.{sentence}' if sentence else '')
            clauses.append(Clause(ref=ref, label=label, heading=heading, fikra=fikra, bent=bent, sentence=sentence,
                                  start=start + a, end=start + b, text=text[start + a:start + b],
                                  chapeau=text[chapeau[0]:chapeau[1]] if chapeau else '',
                                  chapeau_start=chapeau[0] if chapeau else None,
                                  closing=text[closing[0]:closing[1]] if closing else '',
                                  closing_start=closing[0] if closing else None, numbered=numbered,
                                  reading=blank_notes(text[start + a:start + b])))

    for block in blocks:
        lead = (block['lead'][0][0], block['lead'][-1][1])
        if not block['bents']:
            emit(lead[0], lead[1], block['number'])
            continue
        # The lead-in is the chapeau of every bent. When it is itself a full sentence before the list opens
        # ("(1) ... hükümleri uygulanır. Bu genel hükümlere ek olan hükümlere aşağıda yer verilmiştir."), its own
        # sentences are clauses too; the last one, the one the list hangs from, is the chapeau.
        sentences = sentence_spans(text[lead[0]:lead[1]])
        chapeau = (lead[0] + sentences[-1][0], lead[0] + sentences[-1][1]) if sentences else lead
        if len(sentences) > 1:
            for position, (a, b) in enumerate(sentences[:-1], 1):
                clauses.append(Clause(ref=f'{label}/f.{block["number"]}/c.{position}', label=label, heading=heading,
                                      fikra=block['number'], sentence=position, start=lead[0] + a, end=lead[0] + b,
                                      text=text[lead[0] + a:lead[0] + b], numbered=numbered,
                                      reading=blank_notes(text[lead[0] + a:lead[0] + b])))
        for bent in block['bents']:
            emit(bent['spans'][0][0], bent['spans'][-1][1], block['number'], bent['letter'], chapeau, block['closing'])
    # Two lists under one fıkra repeat a bent letter; the second keeps its place with a counter, so every reference
    # names one clause.
    seen: dict[str, int] = {}
    for index, clause in enumerate(clauses):
        seen[clause.ref] = seen.get(clause.ref, 0) + 1
        if seen[clause.ref] > 1:
            clauses[index] = clause.model_copy(update={'ref': f'{clause.ref}({seen[clause.ref]})'})
    return clauses


def clause_at(clauses: list[Clause], ref: str) -> Clause:
    found = [c for c in clauses if c.ref == ref]
    if len(found) != 1:
        raise KeyError(ref)
    return found[0]
