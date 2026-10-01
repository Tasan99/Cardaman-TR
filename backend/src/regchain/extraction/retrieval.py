"""Reference resolution across a pinned consolidated corpus, plus lexical definitions.

No other company's documents and no unchecked web content are queried. Resolution is
deterministic and lexical, not semantic embedding search. Every corpus version used is
pinned by identity and content hash so the retrieved context can be reproduced later.
"""
import math
import re
from collections import Counter
from dataclasses import dataclass


def turkish_references(text):
    # Imported lazily: grounding imports this module for operative_text.
    from .grounding import turkish_references as classify
    return classify(text)

RETRIEVER_VERSION = 'scoped-context-v3'
# Turkish article citations are resolved by article number within the same document.
TR_ARTICLE_REF = re.compile(r'(\d+(?:/[A-ZÇĞİÖŞÜ])?)\s*(?:inci|nci|ncı|ıncı|üncü|uncu|ncu|ncü|inci)\s+madde')
# The end of an article range: "6 ila 14 üncü maddeler", "6-14 üncü maddeler".
RANGE_BEFORE = re.compile(r'\d+(?:/[A-ZÇĞİÖŞÜ])?\s*(?:ila|ilâ|-|–)\s*$')
# Largest optional item (a neighbour or a lexical definition) that may join a packet.
OPTIONAL_CHARS = 4000
BOOKS = ('CONC|COBS|MCOB|SYSC|PRIN|SUP|GEN|DISP|ICOBS|BCOBS|CASS|FEES|PERG|COND|TC|SYSC|IPRU|MAR|CREDS')
REF = re.compile(r'\b(?:(' + BOOKS + r')\s+|(?:paragraph|section|rule)\s+)(\d+[A-Z]?(?:\.\d+[A-Z]*)+)', re.I)
GENERIC_REFERENCE = re.compile(r'\b(?:in accordance with|as defined in|subject to (?:rule|section))\b', re.I)
# An editorial Note records the predecessor guidance a provision derives from. It is
# not an operative cross-reference and is not resolvable inside the Handbook.
NOTE = re.compile(r'\[\s*Note\s*:[^\]]*\]', re.I)
DEFINITION_CUE = re.compile(r'\b(?:as defined in|has the meaning|means)\b', re.I)
TYPE_SUFFIX = re.compile(r'(?<=[0-9A-Z])[RGDE]$')
TYPE_NAMES = {'R': 'RULE', 'G': 'GUIDANCE', 'D': 'DIRECTION', 'E': 'EVIDENTIAL'}
STOP = {'the','and','that','with','this','from','must','shall','should','which','firm','firms','for','are','may','not','under','rule','paragraph'}


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r'[a-z]{3,}', text.lower()) if t not in STOP]


def clean_label(label: str) -> str:
    return re.sub(r'\s+', '', label.upper())


def split_label(label: str) -> tuple[str, str]:
    """Split a printed locator such as 'CONC 7.3.5D' into its book and numeric parts."""
    match = re.match(r'^\s*([A-Za-z]{2,6})\s+(.*)$', label or '')
    return (match[1].upper(), clean_label(match[2])) if match else ('', clean_label(label or ''))


def reference_keys(book: str, numeric: str) -> tuple[tuple[tuple[str, str], str], ...]:
    """Readings of a citation: as printed, and without a trailing provision-type letter.

    CONC 7.3.5D is a provision in its own right; CONC 7.3.5DR is that provision cited
    with its type; CONC 6.7.2R is provision 6.7.2 cited with its type. Each reading is
    returned with the type letter it assumes, so a reading that contradicts the
    regulator's own type marking can be discarded instead of guessed between.
    """
    key = clean_label(numeric)
    stripped = TYPE_SUFFIX.sub('', key)
    readings = [((book.upper(), key), '')]
    if stripped != key:
        readings.append(((book.upper(), stripped), key[-1]))
    return tuple(readings)


def admissible_reading(section: dict, letter: str) -> bool:
    """A type-stripped reading must not contradict the recorded provision type."""
    expected = TYPE_NAMES.get(letter)
    actual = section.get('legal_type') or 'UNKNOWN'
    return not (expected and actual != 'UNKNOWN' and actual != expected)


def operative_text(text: str) -> str:
    """Text with editorial Notes removed, for cross-reference detection only."""
    return NOTE.sub(' ', text)


@dataclass(frozen=True)
class ContextPacket:
    target_id: str
    items: tuple[dict, ...]
    unresolved: tuple[str, ...]
    truncated: bool
    source_kind: str
    quality_flags: tuple[str, ...]
    corpus: tuple[dict, ...] = ()
    missing_sources: tuple[str, ...] = ()
    # References owed by retrieved context, not by the paragraph under extraction.
    # Recorded for the reviewer; they do not by themselves block a candidate.
    dependency_gaps: tuple[str, ...] = ()
    # v0.19: the provision's own heading ("Üçüncü tarafa güven", "CONC 7.8 Jurisdictional requirements"),
    # given to the extraction model as context only. It is read from the target's heading_path, which the
    # packet records with the source, so the manifest below is unchanged.
    heading: str = ''

    def manifest(self) -> dict:
        return {'retriever_version': RETRIEVER_VERSION, 'target_id': self.target_id,
                'items': list(self.items), 'unresolved': list(self.unresolved),
                'truncated': self.truncated, 'source_kind': self.source_kind,
                'quality_flags': list(self.quality_flags),
                # Pinned corpus identity makes the retrieved context reproducible.
                'corpus': list(self.corpus), 'missing_sources': list(self.missing_sources),
                'dependency_gaps': list(self.dependency_gaps)}


def provision_heading(section: dict) -> str:
    """The heading of the provision itself: the last heading_path entry (a mevzuat article heading is the
    third entry and may be empty; a Handbook path ends with the section heading)."""
    path = section.get('heading_path') or []
    return ' '.join(str(path[-1]).split()) if path else ''


def implied_books(section: dict) -> set[str]:
    """Books named by a paragraph's own headings, for sources that print bare labels.

    A PDF instrument prints '7.3.5' under a 'CONC' heading while citing it as
    'CONC 7.3.5'. The heading is the source's own scope, not an inferred one.
    """
    scope = ' '.join(section.get('heading_path') or []) + ' ' + str(section.get('section_number') or '')
    return {book.upper() for book in re.findall(r'\b(' + BOOKS + r')\b', scope, re.I)}


def build_index(sections: list[dict]) -> tuple[dict, dict]:
    """Map printed locators to provisions, and chapter locators to their provisions.

    A locator that is ambiguous within the indexed corpus is recorded as ambiguous and
    is never resolved by picking one of the candidates.
    """
    provisions: dict[tuple[str, str], list[dict]] = {}
    chapters: dict[tuple[str, str], list[dict]] = {}
    for section in sections:
        label = section.get('printed_label')
        if not label:
            continue
        book, numeric = split_label(label)
        if not numeric:
            continue
        books = {book} if book else {''} | implied_books(section)
        for name in books:
            provisions.setdefault((name, numeric), []).append(section)
            parts = numeric.split('.')
            if len(parts) > 2:
                chapters.setdefault((name, '.'.join(parts[:2])), []).append(section)
    return provisions, chapters


def retrieve(target: dict, sections: list[dict], *, corpus: list[dict] | None = None,
             max_chars: int = 32000, max_items: int = 8) -> ContextPacket:
    version = str(target['version_id'])
    same_version = [s for s in sections if str(s['version_id']) == version]
    corpus = list(corpus or [])
    searchable = same_version + corpus
    provisions, chapters = build_index(searchable)
    # Chapters represented in the pinned corpus. A reference into any other chapter is
    # a missing source that can be fixed by ingesting it, not an unresolvable reference.
    available = {(book, '.'.join(numeric.split('.')[:2])) for book, numeric in provisions}
    positions = {str(s['id']): i for i, s in enumerate(same_version)}
    by_id = {str(s['id']): s for s in searchable}
    items: list[dict] = []
    unresolved: list[str] = []
    missing: list[str] = []
    dependency: list[str] = []
    chapter_queue: list[tuple] = []
    ranged: set[str] = set()
    cited: set[str] = set()
    selected = {str(target['id'])}
    remaining = max_chars - len(target['text'])
    truncated = remaining < 0

    def add(section: dict, reason: str, optional: bool = False) -> bool:
        nonlocal remaining, truncated
        sid = str(section['id'])
        if sid in selected:
            if not optional:
                for item in items:
                    if item['section_id']==sid: item['required']=True
            return True
        if section.get('source_kind') == 'AMENDMENT' or set(section.get('quality_flags') or []) & {
                'OMITTED_TEXT_MARKER', 'POSSIBLE_WRAPPED_LABEL', 'DELETED_PROVISION'}:
            return False
        # Optional context must never decide whether a provision can be read. A whole-chapter
        # source brought a 30,000-character annex table in as an optional "definition" for
        # every rule, and each prompt then failed the admission budget.
        if optional and len(section['text']) > OPTIONAL_CHARS:
            return False
        if len(items) >= max_items or len(section['text']) > remaining:
            # A whole-section citation that does not fit is reported as partly inlined.
            # An exact provision citation that does not fit still exhausts the budget.
            truncated = truncated or not optional
            return False
        selected.add(sid)
        remaining -= len(section['text'])
        items.append({'section_id': sid, 'version_id': str(section['version_id']), 'text': section['text'],
                      'source_hash': section['content_hash'], 'reason': reason,'required':not optional,
                      'section': section['section_number'], 'paragraph': section['paragraph_number'],
                      'page': section.get('page'), 'page_end': section.get('page_end'),
                      'printed_label': section.get('printed_label'),
                      'legal_type': section.get('legal_type', 'UNKNOWN'),
                      'effective_from': str(section['effective_from']) if section.get('effective_from') else None,
                      'source_kind': section.get('source_kind', 'UNKNOWN')})
        return True

    def lookup(book: str, numeric: str, source: dict) -> tuple[list[dict], bool, str]:
        """Return matching sections, whether the locator was found, and which index hit.

        Every admissible reading is collected before deciding: if two different
        provisions answer to the same citation, that is reported as ambiguous rather
        than settled by preferring one reading. A chapter locator such as CONC 5D.3
        legitimately names every provision in that section, so it is expanded instead.
        """
        found: list[dict] = []
        kind = 'none'
        for key, letter in reference_keys(book, numeric):
            for hit in provisions.get(key, ()):
                if admissible_reading(hit, letter) and hit not in found:
                    found.append(hit)
                    kind = 'provision'
            if kind == 'provision':
                continue
            if key in chapters:
                return chapters[key], True, 'chapter'
        if found:
            return found, True, kind
        if not book:
            # A bare 'paragraph 4.2' is local to its own document section.
            wanted = {key[1] for key, _ in reference_keys('', numeric)}
            local = [s for s in same_version
                     if clean_label(split_label(s.get('printed_label') or '')[1]) in wanted
                     and s['section_number'] == source['section_number']]
            if local:
                return local, True, 'local'
        return [], False, 'none'

    def resolve(source: dict, depth: int, visited: frozenset, primary: bool) -> None:
        """Depth 0 is the paragraph under extraction; deeper levels are its context.

        A gap in the paragraph's own references means its duty cannot be read safely.
        A gap owed by a retrieved context paragraph is recorded instead of blocking:
        the Handbook's reference graph is effectively the whole Handbook, and treating
        both cases alike would abstain on everything without saying anything useful.
        """
        sid = str(source['id'])
        if sid in visited:
            return
        visited = visited | {sid}
        gaps = unresolved if primary else dependency
        text = operative_text(source['text'])
        references = list(REF.finditer(text))
        for cue in GENERIC_REFERENCE.finditer(text):
            if not any(cue.start() <= ref.start() <= cue.end() + 32 for ref in references):
                gaps.append(cue.group() + ' [unidentified target]')
        for match, article, internal in turkish_references(text):
            # A citation of another statute is a citation, not a reference this snapshot
            # could resolve; the article number is looked up in this document only.
            if not internal:
                continue
            hits = [s for s in same_version if str(s.get('paragraph_number')) == article
                    and s.get('locator_kind') == 'mevzuat_madde']
            if str(target['id']) in {str(s['id']) for s in hits}:
                continue
            if len(hits) != 1:
                gaps.append(match.group() + (' [ambiguous within the pinned corpus]' if hits else ''))
                continue
            if depth >= 1 or not add(hits[0], 'exact_reference', optional=depth > 0):
                gaps.append(match.group())
                continue
            (ranged if primary and RANGE_BEFORE.search(text[:match.start()]) else cited).add(str(hits[0]['id']))
            resolve(hits[0], depth + 1, visited, False)
        for match in references:
            if re.match(r'\s*(?:and|or|to|–|-)\s*\d+\.', text[match.end():], re.I):
                gaps.append(match.group() + ' [compound reference requires review]')
            book, numeric = (match[1] or '').upper(), match[2]
            matches, found, kind = lookup(book, numeric, source)
            if not found:
                key = clean_label(numeric)
                if book and (book, '.'.join(key.split('.')[:2])) not in available:
                    # Actionable: the chapter exists, it simply has not been ingested.
                    (missing if primary else dependency).append(f'{book} {key}')
                gaps.append(match.group())
                continue
            if kind == 'chapter':
                # Deferred: exact provision citations must win the context budget.
                if primary:
                    chapter_queue.append((match.group(), matches, depth, visited))
                else:
                    dependency.append(match.group()+' [secondary section reference]')
                continue
            if len(matches) != 1:
                gaps.append(match.group() + ' [ambiguous within the pinned corpus]')
                continue
            match_section = matches[0]
            if str(match_section['id']) == str(target['id']):
                continue
            # The paragraph's own citations must fit. Citations reached through them
            # are best effort: they are named in the manifest when they do not fit.
            if depth >= 1 or not add(match_section, 'exact_reference', optional=depth > 0):
                gaps.append(match.group())
                continue
            resolve(match_section, depth + 1, visited, False)

    resolve(target, 0, frozenset(), True)
    # v0.19: "6 ila 14 üncü maddeler kapsamında alınan bilgiler" (Tedbirler md. 15) names a range of
    # articles as the scope of what it speaks about; only its last article was looked up, and that
    # article's own conditions were then demanded of md. 15's duty, which no answer could meet. The
    # article stays in the packet as context (the reference counts as resolved) but does not govern.
    for item in items:
        if item['section_id'] in ranged - cited:
            item['required'] = False
    position = positions.get(str(target['id']), -1)
    for index in (position - 1, position + 1):
        if 0 <= index < len(same_version):
            neighbor = same_version[index]
            if neighbor['section_number'] == target['section_number']:
                # A neighbour is conservative extra evidence, not a cited source.
                continuation=bool(re.match(r'^(?:unless|except|provided that|this (?:rule|requirement) does not apply)\b',neighbor['text'],re.I))
                add(neighbor, 'neighbor', optional=not continuation)
    # A glossary definition is the regulator's own definition link, not a lexical guess.
    # Only an explicit definitional cue pulls one in: every Handbook provision links
    # defined terms, and inlining all of them would exhaust the context budget.
    if DEFINITION_CUE.search(operative_text(target['text'])):
        wanted = {term.get('glossary_id') for term in (target.get('defined_terms') or [])}
        for section in corpus:
            if section.get('legal_type') == 'DEFINITION' and section['paragraph_number'] in wanted:
                add(section, 'glossary_definition')
    # Whole-section citations take what the budget has left after exact citations,
    # neighbours and definitions. What did not fit is named rather than hidden.
    for label, matches, depth, visited in chapter_queue:
        added = 0
        for match_section in matches:
            if str(match_section['id']) == str(target['id']):
                continue
            if depth < 3 and add(match_section, 'chapter_reference', optional=False):
                added += 1
                resolve(match_section, depth + 1, visited, False)
        if not added:
            unresolved.append(label + ' [section could not be inlined]')
        elif added < len([m for m in matches if str(m['id']) != str(target['id'])]):
            unresolved.append(label + ' [required section only partly inlined]')
    # TF-IDF ranking only among explicitly definition-like passages. Lexical hits
    # provide potential context; they never mark unresolved legal references resolved.
    query = Counter(tokens(target['text']))
    definitions = [s for s in same_version if re.search(r'\b(means|defined|definition)\b', s['text'], re.I)]
    frequencies = Counter(t for s in definitions for t in set(tokens(s['text'])))
    ranked = []
    for section in definitions:
        terms = Counter(tokens(section['text']))
        overlap = query.keys() & terms.keys()
        if len(overlap) < 2:
            continue
        score = sum((1 + math.log(terms[t])) * math.log(1 + len(definitions) / (1 + frequencies[t])) for t in overlap)
        ranked.append((score, str(section['id']), section))
    for _, __, section in sorted(ranked, key=lambda item: (-item[0], item[1]))[:2]:
        add(section, 'lexical_definition', optional=True)
    # Neighbours and definitions can themselves depend on references. Their mere
    # retrieval must not hide those missing sources from the generation gate.
    for item in list(items):
        resolve(by_id[item['section_id']], 1, frozenset(), False)
    pinned = {}
    for section in corpus:
        key = str(section['version_id'])
        pinned.setdefault(key, {'version_id': key, 'external_key': section.get('external_key'),
                                'normalized_hash': section.get('version_hash')})
    return ContextPacket(str(target['id']), tuple(items), tuple(dict.fromkeys(unresolved)), truncated,
                         target.get('source_kind', 'UNKNOWN'), tuple(target.get('quality_flags') or []),
                         tuple(pinned[key] for key in sorted(pinned)), tuple(dict.fromkeys(missing)),
                         tuple(dict.fromkeys(dependency)), provision_heading(target))
