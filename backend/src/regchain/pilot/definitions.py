"""The regulation's own definitions article, read by rule (v0.19).

Measured in the C12 smoke run (24 September 2026, run 20260924-173120-v019-smoke-ollama, md. 28(2)):
the duty says "Başkanlığa bildirilir", the policy "MASAK'a bildirilir", and the thinking verifier called
it DIRECT_OPPOSITE because it did not know that "Başkanlık" is what the regulation's md. 3 defines as
"Mali Suçları Araştırma Kurulu Başkanlığı" (MASAK). The judge now reads the definitions of the terms
that occur in the duty or the passage. Turkish only: an FCA chapter's defined terms are Glossary links
without their wording, so it gets none ({}).
"""
import re

from .policies import fold

L = 'a-zçğıöşüâîû'
# "(1) Bu Yönetmelikte geçen;" / "Bu Kanunda geçen;" / "Bu Yönetmelikte yer alan;" opens the list.
OPENING = re.compile(r'Bu\s+(?:Yönetmelikte|Kanunda|Tebliğde|Yönergede)\s+(?:geçen|yer\s+alan)(?:\s+terimlerden)?\s*[;:,]')
# One list item: "c) Başkanlık: ..." or "d) (Değişik: 9/5/2014-2014/6381 K.) Denetim elemanı: ...".
ITEM = re.compile(r'(?:(?<=^)|(?<=[;:,.]\s)|(?<=\n))([a-zçğıöşü]{1,2})\)\s+((?:\([^()]*\)\s*)*)([^:;,()]{2,80}?):\s')
NOTE = re.compile(r'\((?:Ek|Değişik|Mülga|Yeniden düzenleme)[^()]*\)\s*')
CLOSING = re.compile(r',?\s*(?:ifade\s+eder|anlamına\s+gelir)\s*\.?\s*$')
MAX_TERMS = 8
MAX_CHARS = 200
# A final consonant softens before a vowel suffix: "Başkanlık" occurs as "Başkanlığa", "Kurum" as "Kuruma".
SOFT = {'k': '[kğ]', 'p': '[pb]', 't': '[td]', 'ç': '[çc]'}


def _is_definitions(section) -> bool:
    heading = fold(' '.join(str(h) for h in (section.get('heading_path') or [])[-1:]))
    return heading.strip() == 'tanımlar' or bool(OPENING.search(section.get('text') or ''))


def parse(text: str) -> dict:
    """{term: definition} of one definitions article's text; {} without the list opening."""
    opening = OPENING.search(text or '')
    if not opening:
        return {}
    listing = '; ' + text[opening.end():].strip()
    items = list(ITEM.finditer(listing))
    out = {}
    for index, match in enumerate(items):
        end = items[index + 1].start() if index + 1 < len(items) else len(listing)
        term = match.group(3).strip()
        definition = NOTE.sub('', listing[match.end():end]).strip()
        definition = CLOSING.sub('', definition).rstrip(' ,;.').strip()
        # Trivial: a one-letter term, an empty or echoed definition, or a note that the item was repealed.
        if len(term) < 3 or len(definition) < 4 or fold(definition) == fold(term) or fold(definition).startswith('mülga'):
            continue
        out.setdefault(term, definition)
    return out


def definitions_of(sections) -> dict:
    """{term: definition} of every definitions article among `sections` (the regulation analysed),
    computed once per analysis. The first definition of a term wins."""
    out = {}
    for section in sections or []:
        if _is_definitions(section):
            for term, definition in parse(section.get('text') or '').items():
                out.setdefault(term, definition)
    return out


def _pattern(term: str):
    folded = fold(term)
    stem, last = folded[:-1], folded[-1:]
    tail = SOFT.get(last, re.escape(last))
    return re.compile(r'(?<![%s])%s%s' % (L, re.escape(stem), tail))


def relevant(definitions: dict, *texts: str) -> dict:
    """The definitions whose term occurs in `texts` (the duty, then the passage), in order of first
    occurrence, at most MAX_TERMS entries of at most MAX_CHARS characters. A term inside a longer
    defined term found at the same place ("Başkan" in "Başkanlığa", where "Başkanlık" matches) is not
    counted."""
    if not definitions:
        return {}
    found = []
    for rank, text in enumerate(texts):
        folded = fold(text or '')
        for term in definitions:
            for match in _pattern(term).finditer(folded):
                found.append((rank, match.start(), -(match.end() - match.start()), match.end(), term))
    taken, kept = {}, []
    for rank, start, _, end, term in sorted(found):
        spans = taken.setdefault(rank, [])
        if any(a <= start and end <= b for a, b in spans):
            continue
        spans.append((start, end))
        if term not in kept:
            kept.append(term)
    return {term: definitions[term][:MAX_CHARS] for term in kept[:MAX_TERMS]}
