"""Real public policy documents (PDF) as policy registers: convert, assess with every sector engine, summarise.

    python scripts/tr_external_test.py <pdf folder> <output folder>

The documents are third-party publications used as test input only; nothing of them is copied into the repository
(evaluation/external is ignored by git). Each company's PDFs become one register (register.json + numbered statements)
for the pilot profile of its type - a company name maps to a profile here, in test tooling, never in production logic.
No model is asked: this is the rule comparer with candidate statements and verification; escalated rows say what a
model would be asked about. The output folder gets registers/, assess-<profile>.json and SUMMARY.txt.
"""
import json
import re
import sys
import time
from pathlib import Path

import logging

from pypdf import PdfReader

logging.getLogger('pypdf').setLevel(logging.ERROR)      # font-encoding notes are not errors

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from regchain.tr.compare import load_register                                  # noqa: E402
from regchain.tr.corpus import CorpusStore                                     # noqa: E402
from regchain.tr.engines import ExpertServices, assess_by_engine               # noqa: E402
from regchain.tr.packs import Registry                                         # noqa: E402
from regchain.tr.profile import load_pilot_profiles                            # noqa: E402

# which pilot profile a company's documents are read as (test tooling only)
COMPANIES = {
    'efes': ('tr-bev-pilot-alcohol-integrated', 'Anadolu Efes'),
    'tuborg': ('tr-bev-pilot-alcohol-integrated', 'Türk Tuborg'),
    'turkish_22': ('tr-bev-pilot-non-alcohol-bottler', 'Coca-Cola İçecek'),
    'cci': ('tr-bev-pilot-non-alcohol-bottler', 'Coca-Cola İçecek'),
    'coca': ('tr-bev-pilot-non-alcohol-bottler', 'Coca-Cola İçecek'),
}
SENTENCE_END = re.compile(r'(?<=[.;!?])\s+(?=[A-ZÇĞİÖŞÜ(“"0-9])')
BULLET = re.compile(r'^\s*(?:[-•●▪■◦*]|\d{1,2}[.)]|[a-zçğıöşü][.)])\s+')
MIN_CHARS, MAX_STATEMENTS = 40, 400


def company_of(path: Path):
    name = path.stem.lower()
    for key, value in COMPANIES.items():
        if key in name:
            return key, value
    return None, None


def pdf_text(path: Path) -> str:
    pages = []
    for page in PdfReader(str(path)).pages:
        text = page.extract_text() or ''
        pages.append(text)
    text = '\n'.join(pages)
    text = re.sub(r'-\n(?=[a-zçğıöşü])', '', text)              # hyphenated line breaks
    text = re.sub(r'[ \t]+', ' ', text)
    return text


def statements_of(text: str) -> list[str]:
    """Normative-looking sentences and bullet items of a document, each at least MIN_CHARS long."""
    out = []
    for block in re.split(r'\n\s*\n|\n(?=\s*(?:[-•●▪■◦*]|\d{1,2}[.)]|[a-zçğıöşü][.)])\s)', text):
        block = ' '.join(line.strip() for line in block.splitlines() if line.strip())
        block = BULLET.sub('', block).strip()
        if not block:
            continue
        for sentence in SENTENCE_END.split(block):
            sentence = sentence.strip()
            if len(sentence) >= MIN_CHARS and not sentence.isupper() and sentence[-1] in '.;!?':
                out.append(sentence)
    return out[:MAX_STATEMENTS]


def write_register(root: Path, profile_id: str, company: str, documents: list[tuple[str, Path, list[str]]]) -> Path:
    directory = root / profile_id
    (directory / 'documents').mkdir(parents=True, exist_ok=True)
    records = []
    for index, (document_id, source, statements) in enumerate(documents):
        file = f'documents/{document_id}.md'
        body = [f'# {source.stem}', '', f'Kaynak: {source.name} (kamuya açık belge, test girdisi).', '']
        body += [f'{i + 1}. {s}\n' for i, s in enumerate(statements)]
        (directory / file).write_text('\n'.join(body), encoding='utf-8')
        records.append({'document_id': document_id, 'type': 'POLICY', 'title': source.stem, 'owner_department': 'COMPLIANCE', 'version': 'public',
                        'file': file, 'scope': {'policy_id': document_id, 'owner_level': 'GROUP'}})
    register = {'format': 'cardaman-tr-policy-register/1', 'profile_id': profile_id, 'synthetic': False,
                'disclaimer': f'{company}: kamuya açık politika belgeleri, test girdisi olarak numaralı cümlelere ayrıldı. Şirketin iç prosedürü, kontrolü veya kanıtı değildir.',
                'period': f'{time.localtime().tm_year}-Q{(time.localtime().tm_mon - 1) // 3 + 1}',
                'documents': records, 'controls': [], 'evidence': []}
    (directory / 'register.json').write_text(json.dumps(register, ensure_ascii=False, indent=2), encoding='utf-8')
    return directory


def main(pdf_folder: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list] = {}
    for path in sorted(pdf_folder.glob('*.pdf')):
        key, mapping = company_of(path)
        if mapping is None:
            print(f'skipped (no company mapping): {path.name}')
            continue
        profile_id, company = mapping
        statements = statements_of(pdf_text(path))
        document_id = 'EXT-' + re.sub(r'[^A-Za-z0-9]+', '-', path.stem).strip('-').upper()[:40]
        groups.setdefault(profile_id, []).append((document_id, path, statements, company))
        print(f'{path.name}: {len(statements)} statements -> {profile_id}')
    registry, store = Registry.load(), CorpusStore()
    profiles = {p.profile_id: p for p in load_pilot_profiles(registry.vocabulary).values()}
    services = ExpertServices(registry, store)
    summary = [f'Cardaman TR external-document test {time.strftime("%Y-%m-%d %H:%M")} (rules + candidates, no model)']
    for profile_id, documents in groups.items():
        company = ' / '.join(sorted({d[3] for d in documents}))
        directory = write_register(out / 'registers', profile_id, company, [(d[0], d[1], d[2]) for d in documents])
        register = load_register(profile_id, out / 'registers')
        result = assess_by_engine(profiles[profile_id], services, register)
        rows = []
        for pack, report in result['reports'].items():
            for row, assessment in zip(report.rows, result['assessments'][pack]):
                if row.document_coverage != 'NO_EVIDENCE' or assessment.escalation is not None:
                    rows.append({'pack': pack, 'ref': row.provision_ref, 'target': row.target_id, 'coverage': row.document_coverage,
                                 'status': row.mapping.status, 'basis': row.coverage_basis, 'reasons': row.coverage_reasons,
                                 'decision': assessment.decision, 'review_reasons': assessment.review_reasons,
                                 'escalation': assessment.escalation.reasons if assessment.escalation else [],
                                 'candidates': [c.passage_id for c in assessment.escalation.candidates] if assessment.escalation else [],
                                 'statements': [(r.passage_id, r.relation, r.quote[:160]) for r in row.readings if r.relation != 'UNRELATED'],
                                 'duty': row.quote[:200]})
        payload = {'profile_id': profile_id, 'company': company, 'register': str(directory),
                   'documents': {d[0]: {'file': d[1].name, 'statements': len(d[2])} for d in documents},
                   'engines': [e.model_dump(mode='json') for e in result['engines']], 'rows_of_interest': rows}
        (out / f'assess-{profile_id}.json').write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
        summary.append(f'== {company} -> {profile_id}: ' + ', '.join(f"{k} {v['statements']} cümle" for k, v in payload['documents'].items()))
        for engine in result['engines']:
            s = engine.statistics
            summary.append(f"   {engine.pack_id}: rows {s['rows']}, by_basis {s['by_basis']}, escalated {s['escalated_rows']} ({s['escalation_rate_rows']:.1%}), "
                           f"review {s['decision_review_rows']} {s['decision_review_reasons']}, not-verified {s['rows_not_verified']}")
            coverage = {}
            for row in result['reports'][engine.pack_id].rows:
                coverage[row.document_coverage] = coverage.get(row.document_coverage, 0) + 1
            summary.append(f'   coverage {coverage}')
        related = [r for r in rows if r['statements']]
        summary.append(f'   related rows {len(related)}; first ten:')
        for r in related[:10]:
            summary.append(f"     {r['coverage']} {r['ref']} @ {r['target']} <- {r['statements'][0][0]} {r['statements'][0][1]}: {r['statements'][0][2][:100]}")
    (out / 'SUMMARY.txt').write_text('\n'.join(summary) + '\n', encoding='utf-8')
    print('\n'.join(summary))
    print(f'\nfolder: {out}')


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else Path(sys.argv[1]).parent / time.strftime('run-%Y%m%d-%H%M%S'))
