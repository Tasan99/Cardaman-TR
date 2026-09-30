"""Test helpers for the TR corpus: a temporary store seeded from the packaged texts, and synthetic amended versions.

A synthetic version is built by re-rendering the stored articles of a real text with a few edited lines. It is marked
synthetic in the store and exists only in a temporary directory: it is a fixture for the change logic, never law.
"""
import html
import re
from datetime import datetime, timezone
from pathlib import Path

from regchain.ingestion.mevzuat import WRAPPED_TEXT_VERSION, parse_mevzuat
from regchain.ingestion.models import Download
from regchain.tr.corpus import CORPUS, CorpusStore, source_url

PACKAGED = CorpusStore(CORPUS)


def seed(store: CorpusStore, registry, regulation_id: str):
    """Record the packaged head version of a regulation in a temporary store, bytes unchanged."""
    meta = registry.regulations[regulation_id]
    version = PACKAGED.head(regulation_id)
    raw = PACKAGED.raw_path(version).read_bytes()
    url = source_url(meta.source_ref)
    download = Download(url, url, raw, 'text/html', version.fetched_at)
    return store.record(regulation_id, meta.source_ref, download, parse_mevzuat(download, WRAPPED_TEXT_VERSION), version.catalogue)[1]


def render(sections, title: str, edits=None, additions=None) -> bytes:
    """The articles of a stored version as a plain HTML page the parser reads back line for line.

    edits      {printed label: [(old text, new text), ...]} applied to the lines of that article
    additions  {printed label: [line, ...]} appended to that article as new source paragraphs
    """
    edits, additions = edits or {}, additions or {}
    parts = [f'<p align=center><b>{html.escape(title)}</b></p>']
    for section in sections:
        if 'DELETED_PROVISION' in section['quality_flags']:
            continue
        heading = section['heading_path'][2] if len(section['heading_path']) >= 3 else ''
        lines = [section['text'][line['start']:line['end']] for line in section['lines']] or [section['text']]
        for old, new in edits.get(section['printed_label'], []):
            if not any(old in line for line in lines):
                raise AssertionError(f'{section["printed_label"]}: "{old}" is not in the article')
            lines = [line.replace(old, new) for line in lines]
        lines = [line for line in lines if line.strip()] + list(additions.get(section['printed_label'], []))
        number = section['paragraph_number']
        mark = (f'GEÇİCİ MADDE {number.split(" ", 1)[1]}' if number.startswith('Geçici ') else
                f'EK MADDE {number.split(" ", 1)[1]}' if number.startswith('Ek ') else f'MADDE {number}')
        if heading:
            parts.append(f'<p><b>{html.escape(heading)}</b></p>')
        parts.append(f'<p><b>{mark} –</b> {html.escape(lines[0])}</p>')
        parts += [f'<p>{html.escape(line)}</p>' for line in lines[1:]]
    return ('<html><body><div class=WordSection1>' + '\n'.join(parts) + '</div></body></html>').encode('utf-8')


def amend(store: CorpusStore, registry, regulation_id: str, edits=None, additions=None, when=datetime(2026, 10, 15, tzinfo=timezone.utc)):
    """Store a synthetic next version: the head version's articles with the edits applied."""
    meta = registry.regulations[regulation_id]
    head = store.head(regulation_id)
    body = render(store.sections(regulation_id), head.title, edits, additions)
    url = source_url(meta.source_ref)
    download = Download(url, url, body, 'text/html', when)
    return store.record(regulation_id, meta.source_ref, download, parse_mevzuat(download, WRAPPED_TEXT_VERSION), head.catalogue,
                        synthetic=True)[1]
