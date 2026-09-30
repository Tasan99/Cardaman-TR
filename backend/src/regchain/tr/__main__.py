"""Run: python -m regchain.tr --help

corpus refresh   fetch the official text of every catalogued regulation that names its source; store new versions
corpus status    the stored versions
corpus verify    what the catalogue or the scopes claim that the stored texts do not support
corpus apply     write the fields the stored texts support into the catalogue (regulations.json)
"""
import argparse
import json
import sys
from pathlib import Path

from .corpus import CORPUS, CorpusStore, refresh, verified_fields, verify_catalogue
from .packs import DATA, Registry, load_json


def _print(value):
    sys.stdout.write(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def apply_verified(store: CorpusStore, path: Path = DATA / 'regulations.json') -> list[dict]:
    """Rewrite the catalogue entries from the stored texts. An entry becomes VERIFIED only when its catalogued title is
    the printed title of the stored text; otherwise it keeps UNVERIFIED and the report names the printed title."""
    from .core import RegulationMeta
    catalogue = load_json(path)
    report = []
    for entry in catalogue['regulations']:
        meta = RegulationMeta.model_validate(entry)
        if meta.source_ref is None or not store.versions(meta.regulation_id):
            report.append({'regulation_id': meta.regulation_id, 'status': 'NO_STORED_TEXT'})
            continue
        fields = verified_fields(meta, store)
        entry['source_url'] = fields['source_url']
        entry['gazette_date'] = fields['gazette_date']
        entry['gazette_number'] = fields['gazette_number']
        entry['versions'] = fields['versions']
        entry['effective_date'] = fields['effective_date']
        entry['effective_status'] = 'IN_FORCE' if fields['effective_date'] else 'UNKNOWN'
        if fields['title_matches']:
            entry['metadata_status'] = 'VERIFIED'
            head = store.head(meta.regulation_id)
            entry['verification_note'] = (
                f'Title, Mevzuat Bilgi Sistemi identifiers, Resmî Gazete date and issue and the text hash were checked against '
                f'the text fetched from mevzuat.gov.tr on {head.fetched_at.date().isoformat()} ({head.articles} articles, '
                f'{head.repealed_articles} repealed). Annex tables and footnotes are not parsed; '
                + ('the effective date is the publication date the text itself names.' if fields['effective_date'] else
                   'the effective date is not derived (staged or non-standard entry into force).'))
            report.append({'regulation_id': meta.regulation_id, 'status': 'VERIFIED', 'title': fields['printed_title']})
        else:
            entry['metadata_status'] = 'UNVERIFIED'
            report.append({'regulation_id': meta.regulation_id, 'status': 'TITLE_MISMATCH', 'catalogued': meta.title,
                           'printed': fields['printed_title']})
        RegulationMeta.model_validate(entry)
    path.write_text(json.dumps(catalogue, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m regchain.tr', description='Cardaman TR: official-source corpus')
    commands = parser.add_subparsers(dest='command', required=True)
    corpus = commands.add_parser('corpus', help='the official-source corpus')
    corpus.add_argument('action', choices=['refresh', 'status', 'verify', 'apply'])
    corpus.add_argument('--root', default=str(CORPUS), help='corpus directory (default: the packaged corpus)')
    corpus.add_argument('--only', action='append', default=[], help='regulation id; repeat to name several')
    args = parser.parse_args(argv)
    store = CorpusStore(Path(args.root))
    if args.action == 'refresh':
        catalogue = Registry.load().regulations.values()
        report = refresh(catalogue, store, only=set(args.only) or None)
        _print(report)
        return 1 if any(row['status'] == 'FAILED' for row in report) else 0
    if args.action == 'status':
        _print({rid: [{'version_id': v.version_id, 'previous': v.previous_version_id, 'fetched_at': v.fetched_at,
                       'articles': v.articles, 'repealed': v.repealed_articles, 'title': v.title,
                       'gazette_date': v.publication_date} for v in store.versions(rid)] for rid in store.regulation_ids()})
        return 0
    if args.action == 'apply':
        _print(apply_verified(store))
        return 0
    registry = Registry.load()
    problems = verify_catalogue(registry.regulations.values(), store, registry.scopes.values())
    _print({'problems': problems})
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
