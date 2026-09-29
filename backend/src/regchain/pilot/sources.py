"""Byte-retained regulatory snapshots; no synthetic edits masquerading as regulatory changes.

Two regulators share one snapshot format: the FCA Handbook (a chapter per page) and the
Turkish Mevzuat Bilgi Sistemi (a regulation per page). A "module" is an FCA book such as
COBS, or a Turkish kind group such as KANUN; a "chapter" is the FCA chapter number or the
Turkish regulation number. Provision labels stay the regulator's own.
"""
import json
import re
from datetime import datetime
from pathlib import Path

from regchain.evidence import digest
from regchain.ingestion.fetch import fetch, validate_url
from regchain.ingestion.handbook import parse_handbook
from regchain.ingestion.mevzuat import (FOOTNOTE_RULES, GROUPS, KINDS, MEVZUAT_PARSER_VERSION, identity, is_mevzuat, parse_mevzuat,
                                        preliminary, scope_rank, text_url)

MEVZUAT_VERSIONS = frozenset(FOOTNOTE_RULES)
from regchain.ingestion.models import Download, PARSER_VERSION
from .paths import extended

URLS = ['https://handbook.fca.org.uk/handbook/conc7/conc7s1',
        'https://handbook.fca.org.uk/handbook/conc7/conc7s3']
MODULE = re.compile(r'^[A-Z]{2,6}$')
CHAPTER = re.compile(r'^[0-9]{1,2}[A-Z]?$')
TR_GROUPS = frozenset(code for code, _ in KINDS.values())
TR_LABEL = re.compile(r'^(Kanun|Yönetmelik|Tebliğ|CBK|KHK|Tüzük) (\d+) md\. (\S+)$')
# Scope evidence sent with every proposal; the whole application section of a large
# chapter would crowd the policy passages out of the prompt.
SCOPE_CHARS = 3000
# Applicability is its own model call since v0.12, so Turkish scope articles (a 'Kapsam'
# article plus the often long 'Tanımlar' and 'Yükümlüler' lists) get a wider budget.
TR_SCOPE_CHARS = 7000


def is_turkish(module: str) -> bool:
    return (module or '').strip().upper() in TR_GROUPS


def chapter_url(module: str, chapter: str) -> str:
    """The FCA serves a whole chapter, every section and provision, on one page."""
    module, chapter = module.strip().upper(), chapter.strip().upper()
    if not MODULE.fullmatch(module) or not CHAPTER.fullmatch(chapter):
        raise ValueError('Regülasyon "COBS" gibi bir modül kodu ve "4" ya da "5D" gibi bir bölüm numarasıyla seçilir.')
    return f'https://handbook.fca.org.uk/handbook/{module.lower()}{chapter.lower()}'


def chapter_of(label: str):
    """('COBS', '4') for 'COBS 4.5A.4' and 'COBS 4 Annex 1R'; ('KANUN', '5549') for
    'Kanun 5549 md. 4'. A run covers one chapter or one Turkish regulation."""
    turkish = TR_LABEL.match(label.strip())
    if turkish:
        return GROUPS[turkish.group(1)], turkish.group(2)
    module, _, number = label.strip().partition(' ')
    return module, number.split('.')[0].split(' ')[0]


def section_of(label: str) -> str:
    """'COBS 4.5A' for 'COBS 4.5A.4', read from the printed label the reviewer also sees.
    A Turkish regulation is one section: 'Kanun 5549' for 'Kanun 5549 md. 4'."""
    turkish = TR_LABEL.match(label.strip())
    if turkish:
        return f'{turkish.group(1)} {turkish.group(2)}'
    module, _, number = label.strip().partition(' ')
    if ' Annex ' in f' {number} ':
        return f'{module} {number.split(" ")[0]} Annex'
    return f'{module} ' + '.'.join(number.split('.')[:2])


def application_rows(sections, module, chapter):
    """The scope evidence for applicability, in document order.

    FCA: the chapter's own application section (X.1 by convention), rules first. Module-wide
    application (for example COBS 1) and Glossary definitions are not read; the packet says
    so, and an applicability proposal stays a suggestion for review.
    Türkiye: the regulation's 'Kapsam' and 'Yükümlüler' articles, then 'Tanımlar'.
    """
    if is_turkish(module):
        rows = [s for s in sections if chapter_of(s['printed_label']) == (module.strip().upper(), str(chapter).strip())
                and scope_rank(s) and 'DELETED_PROVISION' not in s['quality_flags']]
        rows.sort(key=lambda s: (-scope_rank(s), s['ordinal']))
        budget = TR_SCOPE_CHARS
    else:
        rows = [s for s in sections if section_of(s['printed_label']) == f'{module} {chapter}.1'
                and 'DELETED_PROVISION' not in s['quality_flags']]
        rows.sort(key=lambda s: (s['legal_type'] != 'RULE', s['ordinal']))
        budget = SCOPE_CHARS
    chosen, used = [], 0
    for row in rows:
        if chosen and used + len(row['text']) > budget:
            continue
        chosen.append(row)
        used += len(row['text'])
    return sorted(chosen, key=lambda s: s['ordinal'])


def select_targets(sections, module, chapter, mode='rules', only_sections=(), labels=()):
    """Printed labels to analyse, in document order.

    The application section is scope evidence, not a list of duties, so it is not a target
    unless named explicitly. Deleted provisions carry no operative text.
    """
    module, chapter = module.strip().upper(), chapter.strip().upper()
    inside = [s for s in sections if chapter_of(s['printed_label']) == (module, chapter)]
    if not inside:
        words = {code: word for code, word in KINDS.values()}
        pretty = lambda mod, num: f'{words.get(mod, mod)} {num}'
        held = sorted({pretty(*chapter_of(s['printed_label'])) for s in sections})
        wanted = pretty(module, chapter)
        where = 'mevzuat.gov.tr' if is_turkish(module) else 'FCA'
        raise ValueError(f'Saklanan kaynakta {wanted} yok (kaynakta: {", ".join(held[:3]) or "boş"}). '
                         f'04. adımda "{where}\'dan güncel kaynağı indir" seç.')
    if mode == 'labels':
        known = {s['printed_label'] for s in inside}
        missing = [label for label in labels if label not in known]
        if missing or not labels:
            raise ValueError('Kaynakta bulunmayan provision: '+', '.join(missing[:8]) if missing else 'En az bir provision yaz.')
        return [s['printed_label'] for s in inside if s['printed_label'] in set(labels)]
    if is_turkish(module):
        # "Kısımlar" name articles ("3, 4, 8"); without them every operative article is read:
        # not the 'Amaç', 'Kapsam', 'Dayanak', 'Tanımlar', 'Yürürlük', 'Yürütme' ones.
        wanted = {part.strip().upper() for part in only_sections if part.strip()}
        numbers = {s['paragraph_number'].upper() for s in inside}
        unknown = wanted - numbers
        if unknown:
            raise ValueError('Bu düzenlemede olmayan madde: '+', '.join(sorted(unknown)))
        return [s['printed_label'] for s in inside if 'DELETED_PROVISION' not in s['quality_flags']
                and (s['paragraph_number'].upper() in wanted if wanted else not preliminary(s))]
    wanted = {f'{module} {chapter}.{part.strip().upper().split(".")[-1]}' for part in only_sections if part.strip()}
    unknown = wanted - {section_of(s['printed_label']) for s in inside}
    if unknown:
        raise ValueError('Bu bölümde olmayan kısım: '+', '.join(sorted(unknown)))
    return [s['printed_label'] for s in inside
            if 'DELETED_PROVISION' not in s['quality_flags']
            and (section_of(s['printed_label']) in wanted if wanted
                 else section_of(s['printed_label']) != f'{module} {chapter}.1')
            and (mode == 'all' or s['legal_type'] == 'RULE')]


def parse_source(source: Download, recorded_version=None):
    """The regulator's parser for a retained download, chosen by its host.

    A retained mevzuat snapshot names the parser version that produced it; it is read with
    that version's rules, so the evidence keeps reproducing its own hash while a fresh
    download gets the current rules.
    """
    if is_mevzuat(source):
        version = recorded_version if recorded_version in MEVZUAT_VERSIONS else MEVZUAT_PARSER_VERSION
        return parse_mevzuat(source, version), version
    return parse_handbook(source), PARSER_VERSION


def save_sources(directory: Path, downloads=None, module=None, chapter=None, mevzuat=None):
    """Fetch and retain one FCA chapter, or one Turkish regulation given as (kind, number, tertip)."""
    directory = extended(directory)
    directory.mkdir(parents=True, exist_ok=False)
    if downloads is None:
        if mevzuat:
            downloads = [fetch(text_url(*mevzuat))]
        else:
            downloads = [fetch(chapter_url(module, chapter))] if module else [fetch(url) for url in URLS]
    sources = []
    for source in downloads:
        document, parser_version = parse_source(source)
        # The FCA answers some withdrawn locators with a neighbouring chapter. Retaining
        # that would file one chapter's rules under another chapter's name.
        if module and not is_turkish(module) and any(chapter_of(p.printed_label) != (module.strip().upper(), chapter.strip().upper())
                                                     for p in document.paragraphs):
            raise ValueError(f'FCA, {module} {chapter} yerine başka bir bölüm döndürdü; bölüm kaldırılmış veya yeniden numaralanmış olabilir.')
        if mevzuat and identity(source.final_url)[:2] != (str(mevzuat[0]), str(mevzuat[1])):
            raise ValueError('mevzuat.gov.tr istenen düzenleme yerine başka bir metin döndürdü.')
        name = source.raw_hash + '.html'
        (directory/name).write_bytes(source.body)
        sources.append({'url': source.final_url, 'requested_url': source.requested_url,
                        'fetched_at': source.fetched_at.isoformat(), 'media_type': source.media_type,
                        'raw_hash': source.raw_hash, 'raw_file': name, 'parsed_hash': document.content_hash,
                        **({'parser_version': parser_version, 'title': document.title} if parser_version != PARSER_VERSION else {})})
    value = {'format': 'regchain-fca-snapshot-v1', 'parser_version': PARSER_VERSION, 'sources': sources}
    (directory/'snapshot.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    return value


def load_sources(directory: Path):
    # Same namespace for root and member, or the containment check below is void.
    root = extended(directory)
    bundle = json.loads((root/'snapshot.json').read_text(encoding='utf-8'))
    if bundle['format'] != 'regchain-fca-snapshot-v1' or bundle['parser_version'] != PARSER_VERSION:
        raise ValueError('Source snapshot format/parser mismatch')
    sections = []
    labels = set()
    for source in bundle['sources']:
        path = (root/source['raw_file']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Source file must stay inside the snapshot directory')
        validate_url(source['url'])
        item = Download(source['requested_url'], source['url'], path.read_bytes(), source['media_type'],
                        datetime.fromisoformat(source['fetched_at']))
        if item.raw_hash != source['raw_hash']:
            raise ValueError('Retained source bytes do not match source hash')
        document, parser_version = parse_source(item, source.get('parser_version'))
        # The retained snapshot is evidence: its parsed hash must be reproduced. A mevzuat
        # snapshot is read with the parser version it records; an FCA snapshot with today's
        # parser, which is admitted when it reads the document identically (the hash proves it).
        if document.content_hash != source['parsed_hash']:
            raise ValueError('Parsed source snapshot changed; explicitly re-ingest with the new parser')
        if source.get('parser_version', PARSER_VERSION) != parser_version and not is_mevzuat(item):
            raise ValueError('Source snapshot format/parser mismatch')
        version = digest({'url': source['url'], 'raw_hash': item.raw_hash})
        if is_mevzuat(item):
            kind, number, tertip = identity(source['url'])
            external = f'MEVZUAT-{kind}.{tertip}.{number}'          # the catalogue's own file naming
        else:
            external = 'FCA-HB-'+source['url'].rsplit('/', 1)[-1]
        for ordinal, p in enumerate(document.paragraphs):
            if p.printed_label in labels:
                raise ValueError('Ambiguous duplicate provision label in pilot snapshot')
            labels.add(p.printed_label)
            sections.append({'id': digest([version, p.section, p.number]), 'version_id': version,
                'text': p.text, 'content_hash': digest(p.text), 'source_kind': p.source_kind,
                'section_number': p.section, 'paragraph_number': p.number, 'printed_label': p.printed_label,
                'legal_type': p.legal_type, 'effective_from': p.effective_from.isoformat() if p.effective_from else None,
                'quality_flags': list(p.quality_flags), 'heading_path': list(p.heading_path),
                'defined_terms': list(p.defined_terms), 'locator_kind': p.locator_kind, 'ordinal': ordinal,
                'external_key': external, 'version_hash': document.content_hash,
                'source_url': source['url'], 'fetched_at': source['fetched_at']})
    return bundle, sections


def change_for(section, previous):
    if previous is None:
        return {'status': 'NO_BASELINE', 'old': None}
    matches = [p for p in previous if p['printed_label'] == section['printed_label']]
    if len(matches) > 1:
        raise ValueError('Ambiguous previous provision')
    if not matches:
        return {'status': 'NEW_IN_SNAPSHOT', 'old': None}
    old = matches[0]
    status = 'TEXT_UNCHANGED' if old['text'] == section['text'] else 'TEXT_CHANGED_REVIEW_REQUIRED'
    return {'status': status, 'old': old}
