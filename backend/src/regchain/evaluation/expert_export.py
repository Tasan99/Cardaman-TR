"""Expert review export (v0.19 Phase 13): every row of a run in one sheet a lawyer can fill in.

The labels of tr-aml-v1 were derived by the generator rules (build_dataset.py), so a score against
them measures agreement with those rules, not legal accuracy. The way out is a person who reads the
regulation, the profile and the policy and says what the label should be. This module writes, for
one run directory, a table with one row per expected obligation of the run's cases (paired with the
extracted duty exactly as the harness scores it: harness.pair_case over harness.prediction_rows)
and one row per extracted duty no expectation claimed (marked
UNEXPECTED_CANDIDATE): the clause text, the duty, the company profile, the policy passage the AI
quoted (or the passage ranked first), the AI's answers and reasons, the label and its provenance,
and three empty columns for the reviewer (reviewer_decision, reviewer_reason, reviewer_confidence).

Files: expert_review.csv (UTF-8 with a byte-order mark and ';' between fields, what Excel with a
Turkish list separator opens by double-click), expert_review.json (the same rows with the run and
dataset they come from), expert_review.xlsx (written with zipfile and a minimal SpreadsheetML: a
'review' sheet and a 'README' sheet, inline strings, no third-party dependency) and README.md
(Turkish instructions). Nothing in the run folder or the dataset is written.

A filled sheet comes back with ``labels import-review --sheet FILE --actor NAME``: ``sheet_changes``
turns each decision into proposed changes, which pending.py records as PENDING until a second person
approves them. The sheet never changes a label by itself.
"""
import csv
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from xml.etree import ElementTree
from xml.sax.saxutils import escape

from .identifiers import article_of, clause_id

EXPERT_FORMAT = 'cardaman-expert-review-v1'
# The columns of the Phase 13 specification, in this order; then the columns that tie a filled row back to the dataset.
COLUMNS = ('case', 'obligation_id', 'regulation', 'article/clause', 'regulation_text', 'obligation', 'company_profile', 'policy_passage',
           'ai_applicability', 'ai_coverage', 'ai_conflict', 'ai_reason', 'ai_confidence', 'expected_applicability', 'expected_coverage',
           'expected_conflict', 'label_source', 'reviewer_decision', 'reviewer_reason', 'reviewer_confidence')
EXTRA_COLUMNS = ('row_kind', 'ai_vs_label', 'row_id', 'action_keywords', 'expected_evidence', 'obligation_key', 'label_notes', 'match_basis',
                 'missing_reason')
ALL_COLUMNS = COLUMNS + EXTRA_COLUMNS
REVIEWER_COLUMNS = ('reviewer_decision', 'reviewer_reason', 'reviewer_confidence')
ROW_KINDS = ('EXPECTED', 'EXPECTED_NOT_EXTRACTED', 'EXPECTED_CASE_FAILED', 'UNEXPECTED_CANDIDATE')
DECISIONS = ('LABEL_CORRECT', 'AI_CORRECT', 'SET', 'UNSURE')
DECISION_ALIASES = {'ETIKET_DOGRU': 'LABEL_CORRECT', 'AI_DOGRU': 'AI_CORRECT', 'DUZELT': 'SET', 'EMIN_DEGILIM': 'UNSURE'}
CONFIDENCE = {'HIGH': 'HIGH', 'MEDIUM': 'MEDIUM', 'LOW': 'LOW', 'YUKSEK': 'HIGH', 'ORTA': 'MEDIUM', 'DUSUK': 'LOW'}
SETTABLE = ('applicability', 'coverage', 'conflict', 'entity_gate', 'required')
WIDTHS = {'case': 12, 'obligation_id': 16, 'regulation': 20, 'article/clause': 11, 'regulation_text': 60, 'obligation': 45,
          'company_profile': 40, 'policy_passage': 60, 'ai_reason': 60, 'reviewer_decision': 24, 'reviewer_reason': 45,
          'reviewer_confidence': 14, 'label_source': 20, 'row_id': 30, 'expected_evidence': 40, 'label_notes': 50, 'obligation_key': 30}
FOLD = str.maketrans({'ç': 'c', 'ğ': 'g', 'ı': 'i', 'İ': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u', 'Ç': 'c', 'Ğ': 'g', 'Ö': 'o', 'Ş': 's', 'Ü': 'u'})


def squash(text) -> str:
    return ' '.join(str(text or '').split())


def clip(text, size) -> str:
    text = squash(text)
    return text if len(text) <= size else text[:size - 1] + '…'


def code(text) -> str:
    """'Etiket doğru' -> 'ETIKET_DOGRU': a decision or confidence word as typed, folded to its code."""
    return re.sub(r'[\s-]+', '_', str(text or '').strip().translate(FOLD)).upper()


def flag(value) -> str:
    return '' if value is None else ('true' if value else 'false')


def boolish(text) -> str:
    """'true' / 'false' / '' of a sheet cell: Excel re-saves a true/false text as TRUE/FALSE, or DOĞRU/YANLIŞ in Turkish."""
    folded = code(text).lower()
    return 'true' if folded in ('true', 'dogru', 'evet') else 'false' if folded in ('false', 'yanlis', 'hayir') else ''


# ---------------------------------------------------------------------- one row

def company_text(profile: dict) -> str:
    part = lambda key: '; '.join(profile.get(key) or []) or '—'
    return (f'{profile.get("name")} ({profile.get("id")}) — faaliyet: {part("activities")} · lisans: {part("licences")} · '
            f'ürünler: {part("products")} · müşteriler: {part("customer_types")} · yetki alanı: {part("jurisdictions")} · '
            f'{profile.get("description") or ""}').strip()


def duty_text(candidate: dict) -> str:
    action = candidate.get('required_action') or candidate.get('prohibited_action') or ''
    text = f'{candidate.get("subject") or "—"} · {candidate.get("modality") or "—"}: {action}'
    for key, label in (('conditions', 'koşul'), ('exceptions', 'istisna')):
        if candidate.get(key):
            text += f' · {label}: ' + '; '.join(str(v) for v in candidate[key])
    if candidate.get('deadline'):
        text += f' · süre: {candidate["deadline"]}'
    return squash(text)


def clause_of(text: str, clause: str) -> str:
    """The numbered sub-paragraph '(n)' of a provision text, or the whole text when it has none."""
    number = clause_id(clause or '')
    if not number:
        return text
    starts = [(m.start(), m.group(1)) for m in re.finditer(r'(?m)^\s*\((\d+)\)', text)]
    for index, (start, found) in enumerate(starts):
        if found == number:
            end = starts[index + 1][0] if index + 1 < len(starts) else len(text)
            return text[start:end]
    return text


def provision_text(payload, article, clause) -> str:
    for case in payload.get('cases') or []:
        source = case.get('source') or {}
        if article_of(source.get('printed_label') or '') == str(article):
            return clause_of(source.get('text') or '', clause)
    return ''


def passage_text(passages, source_id, quote=None) -> str:
    chunk = passages.get(source_id) or {}
    where = f'[{chunk.get("filename") or "?"} #{chunk.get("number", "?")}] '
    return where + clip(quote or chunk.get('text') or '', 900)


def policy_text(payload, obligation, prediction, passages) -> str:
    """The passage the AI quoted as evidence (a contradiction first), else the passage ranked first."""
    proposal = obligation.get('proposal') or {}
    conflicts = [c for c in proposal.get('policy_checks') or [] if c.get('relation') == 'CONFLICTS' and c.get('quote')]
    quoted = [(c['source_id'], c['quote']) for c in conflicts] or [(q.get('source_id'), q.get('quote')) for q in proposal.get('policy_evidence') or []]
    if quoted:
        return '\n'.join(passage_text(passages, sid, quote) for sid, quote in quoted[:3])
    ranks = prediction.get('ranks') or {}
    if ranks:
        best = min(ranks, key=ranks.get)
        return '(alıntı yok; en üst sıradaki pasaj) ' + passage_text(passages, best)
    return '(politika pasajı okunmadı)'


def verifier_confidence(obligation) -> str:
    """The v0.19 verifier's confidence for this duty (a contradiction's first), '' when no verifier answered."""
    found = []
    for item in obligation.get('diagnostics') or []:
        for result in item.get('results') or []:
            verifier = result.get('verifier')
            if isinstance(verifier, dict) and verifier.get('confidence'):
                found.append((result.get('relation'), str(verifier['confidence'])))
    for check in (obligation.get('proposal') or {}).get('policy_checks') or []:
        if check.get('confidence'):
            found.append((check.get('relation'), str(check['confidence'])))
    conflict = [value for relation, value in found if relation == 'CONFLICTS']
    return (conflict or [value for _, value in found] or [''])[0]


def ai_reason(obligation) -> str:
    proposal = obligation.get('proposal') or {}
    text = (f'Uygulanabilirlik: {clip(proposal.get("applicability_reason"), 800)}\n'
            f'Kapsam: {clip(proposal.get("coverage_reason"), 800)}')
    if proposal.get('review_flags'):
        text += '\nİnceleme bayrakları: ' + ', '.join(map(str, proposal['review_flags']))
    return text


def label_source_text(dataset, expected) -> str:
    base = expected.label_source or dataset.default_label_source or 'RULE_DERIVED'
    per = {field: source for field, source in (expected.label_sources or {}).items() if source != base}
    return base + (' (' + ', '.join(f'{field}: {source}' for field, source in per.items()) + ')' if per else '')


def where_text(case, article, clause) -> str:
    number = clause_id(clause or '')
    return (f'md.{article}' if case.jurisdiction == 'TR' else str(article)) + (f'({number})' if number else '')


def row_id(case_id, expected) -> str:
    """'C08|6|1|kimlik tespitinde': what ties a filled row back to one expectation."""
    return f'{case_id}|{expected.article}|{clause_id(expected.clause or "")}|{";".join(expected.action_keywords)}'


def parse_row_id(text):
    """(case_id, article, clause, action_keywords) of an expectation row; None for an unexpected candidate or an unreadable id."""
    parts = str(text or '').split('|', 3)
    if len(parts) != 4 or parts[1] == '+':
        return None
    return parts[0], parts[1], parts[2], [k for k in parts[3].split(';') if k]


def comparison(expected, prediction) -> str:
    if prediction is None:
        return 'NOT_EXTRACTED'
    differs = []
    if expected.applicability != prediction['applicability']:
        differs.append('applicability')
    if expected.coverage != 'ANY' and expected.coverage != prediction['coverage']:
        differs.append('coverage')
    if expected.conflict is not None and expected.conflict != prediction['conflict']:
        differs.append('conflict')
    return 'DIFFERS: ' + ', '.join(differs) if differs else 'AGREE'


MISSING_REASON_TEXT = {
    'EXTRACTION_MISSED': 'Çıkarıcı bu fıkradan bu yükümlülüğü üretmedi (fıkra görev taşımayan türde sınıflandı, kip ifadesi yok ya da model '
                         'yükümlülük döndürmedi); sistem hakkında karar vermedi.',
    'GROUNDING_REJECTED': 'Model bu fıkradan bir yükümlülük önerdi ama kanıt kapısı (grounding) reddetti; sistem hakkında karar vermedi.',
    'MATCHING_FAILED': 'Bu fıkradan görev çıkarıldı ama etiketle eşleştirilemedi (ayrıntı: AMBIGUOUS = aynı cümleden birden çok görev; '
                       'LABEL_SENTENCE_NOT_EXTRACTED = görevler etiketin cümlesinden değil; CLAIMED_BY_ANOTHER_LABEL = başka etiket aldı).'}


def blank_row(case, dataset) -> dict:
    row = {column: '' for column in ALL_COLUMNS}
    row.update({'case': case.case_id, 'regulation': case.regulation_id, 'company_profile': company_text(case.company_profile)})
    return row


def expected_row(case, dataset, expected, kind) -> dict:
    row = blank_row(case, dataset)
    row.update({'article/clause': where_text(case, expected.article, expected.clause), 'expected_applicability': expected.applicability,
                'expected_coverage': expected.coverage, 'expected_conflict': flag(expected.conflict),
                'label_source': label_source_text(dataset, expected), 'row_kind': kind, 'row_id': row_id(case.case_id, expected),
                'action_keywords': ' | '.join(expected.action_keywords), 'expected_evidence': ' || '.join(expected.evidence),
                'label_notes': expected.notes})
    return row


def case_rows(case, dataset, payload, passages, error=None) -> list:
    """The rows of one case: its expectations (paired as the harness pairs them) and the unclaimed duties."""
    from .harness import pair_case, prediction_rows
    if payload is None:
        rows = []
        for expected in case.expected_obligations:
            row = expected_row(case, dataset, expected, 'EXPECTED_CASE_FAILED')
            row.update({key: 'CASE_FAILED' for key in ('ai_applicability', 'ai_coverage')})
            row['ai_reason'] = f'Vaka koşuda tamamlanmadı: {error or "paket yok"}'
            row['ai_vs_label'] = 'CASE_FAILED'
            rows.append(row)
        return rows
    predictions = prediction_rows(payload, case.regulation_id)
    matches, extras = pair_case(case.expected_obligations, predictions, payload)
    obligations = {o['id']: o for o in payload.get('obligations') or []}
    rows = []
    for match in matches:
        expected, prediction = match['expected'], match['prediction']
        if prediction is None:
            row = expected_row(case, dataset, expected, 'EXPECTED_NOT_EXTRACTED')
            # v0.19 (problem 1 F): why the row has no extracted duty (identifiers.MISSING_REASONS), in the reviewer's words.
            reason = MISSING_REASON_TEXT.get(match['missing_reason'], MISSING_REASON_TEXT['EXTRACTION_MISSED'])
            reason += f' ({match["missing_detail"]})' if match.get('missing_detail') else ''
            row.update({'regulation_text': squash(provision_text(payload, expected.article, expected.clause)),
                        'obligation': '(çıkarılmadı) beklenen eylem sözcükleri: ' + (', '.join(expected.action_keywords) or '—'),
                        'ai_applicability': 'NOT_EXTRACTED', 'ai_coverage': 'NOT_EXTRACTED', 'ai_vs_label': 'NOT_EXTRACTED',
                        'ai_reason': reason, 'match_basis': match['basis'],
                        'missing_reason': f'{match["missing_reason"]}/{match["missing_detail"]}' if match.get('missing_reason') else ''})
            rows.append(row)
            continue
        row = expected_row(case, dataset, expected, 'EXPECTED')
        row['match_basis'] = match['basis']
        rows.append(prediction_row(row, payload, prediction, obligations, passages, comparison(expected, prediction)))
    for prediction in extras:
        row = blank_row(case, dataset)
        row.update({'article/clause': where_text(case, prediction['article'], prediction['clause']), 'row_kind': 'UNEXPECTED_CANDIDATE',
                    'row_id': f'{case.case_id}|+|{prediction["key"]}', 'label_source': '',
                    'label_notes': 'Veri setinde bu yükümlülük için etiket yok (beklenmeyen aday).'})
        rows.append(prediction_row(row, payload, prediction, obligations, passages, 'NO_LABEL'))
    return rows


def prediction_row(row, payload, prediction, obligations, passages, compared) -> dict:
    from .coverage_errors import clause_text
    obligation = obligations.get(prediction['obligation_id']) or {}
    row.update({'obligation_id': prediction['obligation_id'], 'regulation_text': squash(clause_text(payload, obligation)),
                'obligation': duty_text(obligation.get('candidate') or {}), 'policy_passage': policy_text(payload, obligation, prediction, passages),
                'ai_applicability': prediction['applicability'], 'ai_coverage': prediction['coverage'], 'ai_conflict': flag(prediction['conflict']),
                'ai_reason': ai_reason(obligation), 'ai_confidence': verifier_confidence(obligation), 'obligation_key': prediction['key'],
                'ai_vs_label': compared})
    return row


def find_dataset(run_dir, manifest, dataset_path=None) -> Path:
    if dataset_path:
        return Path(dataset_path)
    recorded = manifest.get('dataset_path')             # where the v0.19 manifest keeps it (outside the hash)
    if recorded and Path(recorded).is_file():
        return Path(recorded)
    from .coverage_errors import resolve_dataset
    return resolve_dataset(run_dir, manifest)[0]


REPO_ROOT = Path(__file__).resolve().parents[4]


def portable(path) -> str:
    """A path as it may leave this machine: relative to the repository (posix), else the bare file name."""
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def export_rows(run_dir, dataset_path=None):
    """(rows, meta) of a run directory; nothing is written."""
    from .coverage_errors import passage_index
    from .harness import load_dataset
    from .taxonomy import load_results
    run_dir = Path(run_dir)
    manifest_path = run_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}
    dataset_file = find_dataset(run_dir, manifest, dataset_path)
    dataset = load_dataset(dataset_file)
    results = {}
    if (run_dir / 'results.json').is_file():
        results = {r['case_id']: r for r in load_results(run_dir)[0]}
    selected = list((manifest.get('mode') or {}).get('cases') or results)
    rows = []
    for case in dataset.cases:
        if case.case_id not in selected:
            continue
        packet_path = run_dir / 'packets' / f'{case.case_id}.json'
        payload = json.loads(packet_path.read_text(encoding='utf-8'))['events'][0]['payload'] if packet_path.is_file() else None
        error = (results.get(case.case_id) or {}).get('error') or ('the case was not run' if case.case_id not in results else None)
        rows.extend(case_rows(case, dataset, payload, passage_index(payload) if payload else {}, error))
    kinds = {kind: sum(1 for r in rows if r['row_kind'] == kind) for kind in ROW_KINDS}
    meta = {'format': EXPERT_FORMAT, 'created_at': datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            'run': run_dir.name, 'run_name': run_dir.name, 'manifest_sha256': manifest.get('manifest_sha256'),
            'models': manifest.get('models'), 'dataset': {'id': dataset.dataset_id, 'version': dataset.version, 'path': portable(dataset_file),
                                                          'sha256': sha256(dataset_file.read_bytes()).hexdigest(),
                                                          'canonical_sha256_at_run': (manifest.get('dataset') or {}).get('sha256')},
            'cases': [c for c in selected if c in {case.case_id for case in dataset.cases}], 'counts': {'rows': len(rows), **kinds},
            'columns': list(ALL_COLUMNS), 'reviewer_columns': list(REVIEWER_COLUMNS),
            'decisions': {'LABEL_CORRECT': 'the label stands (ETIKET_DOGRU)', 'AI_CORRECT': 'the AI answer is right where it differs (AI_DOGRU)',
                          'SET field=value; field=value': 'the label should be this (DUZELT)', 'UNSURE': 'no decision (EMIN_DEGILIM)'},
            'confidence': ['HIGH', 'MEDIUM', 'LOW']}
    return rows, meta


# ---------------------------------------------------------------------- files

def write_csv(path: Path, rows, columns=ALL_COLUMNS, delimiter=';'):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), delimiter=delimiter, lineterminator='\r\n', extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    Path(path).write_text(buffer.getvalue(), encoding='utf-8-sig', newline='')


ILLEGAL_XML = re.compile('[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]')
MAIN_NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PKG_REL_NS = 'http://schemas.openxmlformats.org/package/2006/relationships'


def column_letter(index: int) -> str:
    """0 -> 'A', 25 -> 'Z', 26 -> 'AA'."""
    letters, index = '', index + 1
    while index:
        index, rest = divmod(index - 1, 26)
        letters = chr(65 + rest) + letters
    return letters


def xml_text(value) -> str:
    # An Excel cell holds at most 32767 characters; control characters are not allowed in XML 1.0.
    return escape(ILLEGAL_XML.sub('', str(value))[:32767])


def cell(ref, value, style) -> str:
    if value in (None, ''):
        return f'<c r="{ref}" s="{style}"/>' if style else ''
    return f'<c r="{ref}" t="inlineStr" s="{style}"><is><t xml:space="preserve">{xml_text(value)}</t></is></c>'


def sheet_xml(table, widths, styles, selected=False, frozen=True, validations=()) -> str:
    """One worksheet: ``table`` is a list of rows (lists of values), the first the header; ``styles`` a style id per column."""
    last = f'{column_letter(max(len(table[0]), 1) - 1)}{len(table)}'
    rows = []
    for number, values in enumerate(table, 1):
        cells = ''.join(cell(f'{column_letter(i)}{number}', value, 1 if number == 1 else styles[i]) for i, value in enumerate(values))
        rows.append(f'<row r="{number}">{cells}</row>')
    view = ('<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/><selection pane="bottomLeft" activeCell="A2" sqref="A2"/>'
            if frozen else '')
    cols = ''.join(f'<col min="{i + 1}" max="{i + 1}" width="{w}" customWidth="1"/>' for i, w in enumerate(widths))
    tab = ' tabSelected="1"' if selected else ''
    checks = ''
    if validations:
        checks = f'<dataValidations count="{len(validations)}">' + ''.join(
            f'<dataValidation type="list" allowBlank="1" showInputMessage="1" showErrorMessage="{1 if strict else 0}" '
            f'errorStyle="{"stop" if strict else "information"}" sqref="{sqref}"><formula1>"{xml_text(",".join(values))}"</formula1></dataValidation>'
            for sqref, values, strict in validations) + '</dataValidations>'
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            f'<worksheet xmlns="{MAIN_NS}" xmlns:r="{REL_NS}"><dimension ref="A1:{last}"/>'
            f'<sheetViews><sheetView workbookViewId="0"{tab}>{view}</sheetView></sheetViews>'
            f'<sheetFormatPr defaultRowHeight="15"/><cols>{cols}</cols><sheetData>{"".join(rows)}</sheetData>'
            + (f'<autoFilter ref="A1:{last}"/>' if frozen else '') + checks +
            '<pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" header="0.3" footer="0.3"/></worksheet>')


STYLES = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
          f'<styleSheet xmlns="{MAIN_NS}">'
          '<fonts count="2"><font><sz val="11"/><name val="Calibri"/><family val="2"/></font>'
          '<font><b/><sz val="11"/><name val="Calibri"/><family val="2"/></font></fonts>'
          '<fills count="4"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill>'
          '<fill><patternFill patternType="solid"><fgColor rgb="FFDDEBF7"/><bgColor indexed="64"/></patternFill></fill>'
          '<fill><patternFill patternType="solid"><fgColor rgb="FFFFF2CC"/><bgColor indexed="64"/></patternFill></fill></fills>'
          '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
          '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
          '<cellXfs count="4"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
          '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1">'
          '<alignment vertical="top" wrapText="1"/></xf>'
          '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>'
          '<xf numFmtId="0" fontId="0" fillId="3" borderId="0" xfId="0" applyFill="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>'
          '</cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>')


def write_xlsx(path: Path, rows, columns=ALL_COLUMNS, readme_lines=(), title='Cardaman expert review'):
    """A two-sheet workbook ('review', 'README') from the rows; stdlib only, inline strings, openable in Excel."""
    columns = list(columns)
    table = [columns, *[[row.get(column, '') for column in columns] for row in rows]]
    styles = [3 if column in REVIEWER_COLUMNS else 2 for column in columns]
    widths = [WIDTHS.get(column, 16) for column in columns]
    count = len(table)
    validations = []
    for column, values, strict in (('reviewer_decision', ['LABEL_CORRECT', 'AI_CORRECT', 'SET coverage=PARTIAL', 'UNSURE'], False),
                                   ('reviewer_confidence', ['HIGH', 'MEDIUM', 'LOW'], True)):
        if column in columns and count > 1:
            letter = column_letter(columns.index(column))
            validations.append((f'{letter}2:{letter}{count}', values, strict))
    readme = [['README'], *[[line] for line in readme_lines]]
    created = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    last = f'{column_letter(len(columns) - 1)}{count}'
    parts = {
        '[Content_Types].xml': (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
            '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
            '</Types>'),
        '_rels/.rels': (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            f'<Relationships xmlns="{PKG_REL_NS}">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
            '</Relationships>'),
        'docProps/core.xml': (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            f'<dc:title>{xml_text(title)}</dc:title><dc:creator>Cardaman</dc:creator>'
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created></cp:coreProperties>'),
        'docProps/app.xml': (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
            'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Cardaman</Application></Properties>'),
        'xl/workbook.xml': (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            f'<workbook xmlns="{MAIN_NS}" xmlns:r="{REL_NS}"><bookViews><workbookView activeTab="0"/></bookViews>'
            '<sheets><sheet name="review" sheetId="1" r:id="rId1"/><sheet name="README" sheetId="2" r:id="rId2"/></sheets>'
            f'<definedNames><definedName name="_xlnm._FilterDatabase" localSheetId="0" hidden="1">review!$A$1:${last.rstrip("0123456789")}${count}</definedName>'
            '</definedNames></workbook>'),
        'xl/_rels/workbook.xml.rels': (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            f'<Relationships xmlns="{PKG_REL_NS}">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>'
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            '</Relationships>'),
        'xl/styles.xml': STYLES,
        'xl/worksheets/sheet1.xml': sheet_xml(table, widths, styles, selected=True, validations=validations),
        'xl/worksheets/sheet2.xml': sheet_xml(readme, [130], [2], frozen=False),
    }
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, text in parts.items():                 # [Content_Types].xml first, as Office writes it
            archive.writestr(name, text.encode('utf-8'))


def read_xlsx(path: Path, sheet='review') -> list:
    """The rows of one sheet of an .xlsx as dicts keyed by the header row (inline or shared strings, as Excel saves them)."""
    ns = {'m': MAIN_NS}
    with zipfile.ZipFile(path) as archive:
        workbook = ElementTree.fromstring(archive.read('xl/workbook.xml'))
        rels = {r.get('Id'): r.get('Target') for r in ElementTree.fromstring(archive.read('xl/_rels/workbook.xml.rels'))}
        sheets = [(s.get('name'), s.get(f'{{{REL_NS}}}id')) for s in workbook.find('m:sheets', ns)]
        rid = next((r for name, r in sheets if name == sheet), sheets[0][1])
        target = rels[rid]
        part = target.lstrip('/') if target.startswith('/') else 'xl/' + target
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            for item in ElementTree.fromstring(archive.read('xl/sharedStrings.xml')).findall('m:si', ns):
                parts = []                               # plain <t>, or rich-text runs <r><t>; phonetic <rPh> is not the value
                for child in item:
                    if child.tag == f'{{{MAIN_NS}}}t':
                        parts.append(child.text or '')
                    elif child.tag == f'{{{MAIN_NS}}}r':
                        parts.extend(t.text or '' for t in child.findall('m:t', ns))
                shared.append(''.join(parts))
        grid = []
        for row in ElementTree.fromstring(archive.read(part)).iter(f'{{{MAIN_NS}}}row'):
            values, position = {}, 0
            for c in row.findall('m:c', ns):
                letters = re.match(r'[A-Z]+', c.get('r') or '')
                if letters:
                    position = 0
                    for char in letters.group(0):
                        position = position * 26 + ord(char) - 64
                    position -= 1
                kind, raw = c.get('t'), c.find('m:v', ns)
                if kind == 's' and raw is not None:
                    value = shared[int(raw.text)]
                elif kind == 'inlineStr':
                    value = ''.join(t.text or '' for t in c.iter(f'{{{MAIN_NS}}}t'))
                else:
                    value = raw.text if raw is not None and raw.text is not None else ''
                values[position] = value
                position += 1
            grid.append(values)
    if not grid:
        return []
    header = {index: name for index, name in grid[0].items() if name}
    return [{name: row.get(index, '') for index, name in header.items()} for row in grid[1:] if any(str(v).strip() for v in row.values())]


def read_sheet(path: Path) -> list:
    """The rows of a filled expert-review file: .csv (';' or ','), .json (the bundle or a list) or .xlsx."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == '.xlsx':
        return read_xlsx(path)
    if suffix == '.json':
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        return list(data['rows'] if isinstance(data, dict) else data)
    try:
        text = path.read_text(encoding='utf-8-sig')
    except UnicodeDecodeError:
        # Excel's plain "CSV" save on a Turkish Windows writes the ANSI code page, not UTF-8.
        text = path.read_bytes().decode('cp1254')
    header = text.split('\n', 1)[0]
    return list(csv.DictReader(io.StringIO(text, newline=''), delimiter=';' if header.count(';') >= header.count(',') else ','))


def parse_decision(text):
    """(kind, {field: raw value}) of a reviewer_decision cell; ValueError when it cannot be read."""
    raw = str(text or '').strip()
    whole = DECISION_ALIASES.get(code(raw), code(raw))       # 'Etiket doğru', 'Emin değilim': several words, one decision
    if whole in ('LABEL_CORRECT', 'AI_CORRECT', 'UNSURE'):
        return whole, {}
    head, rest = (raw.split(None, 1) + [''])[:2]
    kind = DECISION_ALIASES.get(code(head), code(head))
    if '=' in head:                                          # 'coverage=PARTIAL' without the SET word
        kind, rest = 'SET', raw
    if kind not in DECISIONS:
        raise ValueError(f'unknown decision {head!r} (LABEL_CORRECT, AI_CORRECT, SET field=value, UNSURE)')
    assignments = {}
    if kind == 'SET':
        for part in re.split(r'[;,]', rest):
            if not part.strip():
                continue
            field, sep, value = part.partition('=')
            field = field.strip().lower()
            if not sep or field not in SETTABLE:
                raise ValueError(f'cannot read {part.strip()!r} (field=value; fields: {", ".join(SETTABLE)})')
            assignments[field] = value.strip()
        if not assignments:
            raise ValueError('SET needs at least one field=value')
    return kind, assignments


def sheet_changes(row: dict):
    """(proposed changes, note) of one filled sheet row; the note says why a decision proposes nothing."""
    decision = str(row.get('reviewer_decision') or '').strip()
    if not decision:
        return [], None
    try:
        kind, assignments = parse_decision(decision)
    except ValueError as exc:
        return [], f'reviewer_decision not understood: {exc}'
    if kind in ('LABEL_CORRECT', 'UNSURE'):
        return [], kind
    target = parse_row_id(row.get('row_id'))
    if target is None:
        return [], 'no expectation to change (an unexpected candidate or an unreadable row_id); add a label to a new dataset version instead'
    reason = str(row.get('reviewer_reason') or '').strip()
    if not reason:
        return [], 'reviewer_reason is empty (a change needs the reviewer\'s reason)'
    current = {'applicability': row.get('expected_applicability') or '', 'coverage': row.get('expected_coverage') or '',
               'conflict': boolish(row.get('expected_conflict'))}
    if kind == 'AI_CORRECT':
        ai = {'applicability': row.get('ai_applicability') or '', 'coverage': row.get('ai_coverage') or '', 'conflict': boolish(row.get('ai_conflict'))}
        assignments = {}
        if ai['applicability'] in ('APPLIES', 'DOES_NOT_APPLY', 'UNKNOWN') and current['applicability'] and ai['applicability'] != current['applicability']:
            assignments['applicability'] = ai['applicability']
        if current['coverage'] not in ('', 'ANY') and ai['coverage'] not in ('', 'NOT_EXTRACTED', 'CASE_FAILED') and ai['coverage'] != current['coverage']:
            assignments['coverage'] = ai['coverage']
        if current['conflict'] in ('true', 'false') and ai['conflict'] in ('true', 'false') and ai['conflict'] != current['conflict']:
            assignments['conflict'] = ai['conflict']
        if not assignments:
            return [], 'AI_CORRECT, but the AI answer already equals every asserted label'
    case_id, article, clause, keywords = target
    confidence = CONFIDENCE.get(code(row.get('reviewer_confidence')), None)
    evidence = {'row_id': row.get('row_id'), 'decision': decision, 'reviewer_confidence': confidence,
                'obligation_key': row.get('obligation_key') or None,
                'ai_output': {'applicability': row.get('ai_applicability'), 'coverage': row.get('ai_coverage'), 'conflict': row.get('ai_conflict')}}
    changes = []
    for field, value in assignments.items():
        change = {'case_id': case_id, 'article': article, 'clause': clause, 'action_keywords': keywords or None, 'field': field, 'to': value,
                  'reason': f'Expert review ({kind}): {reason}', 'evidence': dict(evidence)}
        if field in current:
            change['from'] = current[field] if field != 'conflict' else (current['conflict'] or None)
        changes.append(change)
    return changes, None


def readme_markdown(meta, dataset_file=None) -> str:
    """Turkish instructions for the reviewer (README.md and the README sheet).

    The package goes to an outside reviewer: it names the run folder and the repository-relative dataset
    path, never a local absolute path; commands are written as run from backend/.
    """
    counts, dataset = meta['counts'], meta['dataset']
    dataset_arg = '../' + dataset['path'] if not Path(dataset['path']).is_absolute() and '/' in dataset['path'] else dataset['path']
    pending_line = ''
    ledger = (Path(dataset_file) if dataset_file else REPO_ROOT / dataset['path']).resolve().parent.parent / 'reviewed_labels' / f'{dataset["id"]}.pending.json'
    if ledger.is_file():
        try:
            waiting = sum(1 for e in json.loads(ledger.read_text(encoding='utf-8-sig')).get('entries') or [] if e.get('status') == 'PENDING')
            pending_line = (f'Dışa aktarım anında `evaluation/reviewed_labels/{ledger.name}` dosyasında {waiting} bekleyen öneri vardı '
                            f'(`labels pending` ile görün); bunlar bir kişi onaylayana kadar hiçbir skoru değiştirmez.\n\n')
        except (ValueError, OSError):
            pending_line = ''
    return f'''# Uzman inceleme paketi — `{meta['run_name']}`

Bu paket, bir değerlendirme koşusunun her yükümlülük satırını bir uzmanın (hukukçu / uyum uzmanı) incelemesi için tek tabloda toplar. Amaç: veri setindeki "beklenen" etiketlerin hukuken doğru olup olmadığına bir insanın karar vermesi. tr-aml-v1 etiketleri üretici kurallarla (build_dataset.py) türetildi; bu yüzden onlara karşı alınan skor hukuki doğruluğu değil, kurala uyumu ölçer.

## Dosyalar

- `expert_review.xlsx` — Excel ile açın. `review` sayfası satırları, `README` sayfası bu açıklamayı içerir. Doldurulacak üç sütun sarıdır.
- `expert_review.csv` — aynı tablo; UTF-8 (BOM'lu), alanlar noktalı virgülle (;) ayrılır (Türkçe Excel çift tıklamayla doğru açar).
- `expert_review.json` — aynı satırlar ile koşu ve veri seti künyesi (makine okur).

Koşu: `{meta['run']}` (manifest sha256 `{str(meta.get('manifest_sha256') or '—')[:16]}…`). Veri seti: `{dataset_arg}` ({dataset['id']} v{dataset['version']}, dosya sha256 `{dataset['sha256'][:16]}…`). Satırlar: {counts['rows']} (etiketli ve eşleşmiş {counts['EXPECTED']}, çıkarılmamış {counts['EXPECTED_NOT_EXTRACTED']}, vakası tamamlanmamış {counts['EXPECTED_CASE_FAILED']}, etiketsiz beklenmeyen aday {counts['UNEXPECTED_CANDIDATE']}).

## Sütunlar

| Sütun | İçerik |
|---|---|
| case | Vaka kimliği |
| obligation_id | Sistemin çıkardığı yükümlülüğün kimliği (çıkarılmadıysa boş) |
| regulation, article/clause | Mevzuat kimliği ve madde/fıkra |
| regulation_text | Yükümlülüğün alındığı fıkranın metni |
| obligation | Sistemin çıkardığı yükümlülük (özne · kiplik: eylem · koşul/istisna/süre) |
| company_profile | Şirket profili (faaliyet, lisans, ürünler, müşteriler) |
| policy_passage | Sistemin alıntıladığı politika pasajı (çelişki varsa önce o); alıntı yoksa en üst sıradaki pasaj |
| ai_applicability, ai_coverage, ai_conflict | Sistemin cevabı (NOT_EXTRACTED: yükümlülük çıkarılmadı) |
| ai_reason | Sistemin gerekçesi ve inceleme bayrakları |
| ai_confidence | v0.19 doğrulayıcısının güveni (HIGH/MEDIUM/LOW); doğrulayıcı çalışmadıysa boş |
| expected_applicability, expected_coverage, expected_conflict | Veri setindeki etiket (ANY / boş = etiket bu alanda bir şey iddia etmiyor) |
| label_source | Etiketin kaynağı: RULE_DERIVED (kuraldan türetildi), MANUAL_LEGAL_READING, SYNTHETIC_CONTROLLED, HUMAN_REVIEWED |
| reviewer_decision, reviewer_reason, reviewer_confidence | **Sizin dolduracağınız sütunlar** |
| row_kind | EXPECTED, EXPECTED_NOT_EXTRACTED, EXPECTED_CASE_FAILED, UNEXPECTED_CANDIDATE |
| ai_vs_label | AGREE / DIFFERS: … — sistemle etiketin ayrıştığı alanlar (önceliklendirmek için filtreleyin) |
| row_id | Satırı veri setindeki etikete bağlayan anahtar — **değiştirmeyin** |
| action_keywords, expected_evidence, obligation_key, label_notes | Etiketin eylem sözcükleri, beklenen kanıt metinleri, sistem anahtarı ve etiket notu |
| match_basis, missing_reason | Etiketin görevle nasıl eşleştiği (CLAUSE_UNIQUE, FINGERPRINT, SOURCE_SPAN …) ve eşleşmediyse nedeni (EXTRACTION_MISSED, GROUNDING_REJECTED, MATCHING_FAILED / ayrıntı) |

## İnceleyici ne doldurur

Yalnızca üç sütunu doldurun; diğer sütunları (özellikle `row_id`) değiştirmeyin. Satır silebilir veya sıralayabilirsiniz.

**reviewer_decision** — şu değerlerden biri (Türkçe karşılıkları da kabul edilir):

| Değer | Türkçe | Anlamı |
|---|---|---|
| `LABEL_CORRECT` | `ETIKET_DOGRU` | Beklenen etiket doğru. Öneri oluşmaz. |
| `AI_CORRECT` | `AI_DOGRU` | Sistemin cevabı doğru: etiketin sistemle ayrıştığı her etiketli alan için değişiklik önerilir. |
| `SET alan=değer; alan=değer` | `DUZELT alan=değer` | Etiket şu olmalı. Örnek: `SET coverage=PARTIAL` veya `SET applicability=DOES_NOT_APPLY; coverage=NOT_ASSESSED`. |
| `UNSURE` | `EMIN_DEGILIM` | Karar verilemedi. Öneri oluşmaz. |
| (boş) | | İncelenmedi. |

`SET` ile verilebilecek alanlar ve değerler: `applicability` = APPLIES / DOES_NOT_APPLY / UNKNOWN; `coverage` = COVERS_TEXT / PARTIAL / CONFLICT / NO_EVIDENCE / UNKNOWN / NOT_ASSESSED / ANY; `conflict` = true / false / null; `entity_gate` = MATCH / MISMATCH / UNDETERMINED / NOT_RESTRICTED / ANY; `required` = true / false.

**reviewer_reason** — bir değişiklik öneriyorsanız zorunlu. Hangi mevzuat ifadesine ve hangi politika cümlesine dayandığınızı yazın (ör. "md. 6(1) telefon numarasını da ister; politika §2 bunu saymıyor, bu yüzden PARTIAL").

**reviewer_confidence** — `HIGH` / `MEDIUM` / `LOW` (`YÜKSEK` / `ORTA` / `DÜŞÜK` da olur).

`UNEXPECTED_CANDIDATE` satırlarında veri setinde etiket yoktur; bu satırlardaki bir karar öneri üretmez (yükümlülüğün etiketlenmesi gerekiyorsa yeni bir veri seti sürümünde elle eklenmelidir).

## Kararlar nasıl geri gelir (PENDING)

Doldurulmuş dosya (xlsx, csv veya json) kararları **bekleyen öneri (PENDING)** olarak kaydeder; hiçbir etiket ve hiçbir skor kendiliğinden değişmez. `backend` klasöründen:

```
set PYTHONUTF8=1
..\\.venv\\Scripts\\python.exe -m regchain.evaluation labels import-review --dataset {dataset_arg} --sheet <doldurulmuş expert_review.xlsx> --actor "Ad Soyad" --role "AML uzmanı"
```

Her değişiklik `evaluation/reviewed_labels/{dataset['id']}.pending.json` dosyasına PENDING olarak yazılır ve `{dataset['id']}.audit.jsonl` denetim günlüğüne (hash zinciri) eklenir. Tek bir değişiklik elle de önerilebilir:

```
..\\.venv\\Scripts\\python.exe -m regchain.evaluation labels propose --dataset {dataset_arg} --case C08 --article 6 --clause 1 --field coverage --to PARTIAL --reason "..." --actor "Ad Soyad" --role "AML uzmanı" --origin expert-review
```

Pilot çalışma alanındaki bir incelemenin OVERRIDE kararları da aynı yoldan gelir: `labels import-review --dataset … --run <koşu klasörü> --review <review.json>`.

Sonra:

- `labels pending --dataset …` bekleyenleri listeler.
- `labels approve --dataset … --ids <entry_id> --approver "Başka Bir Kişi"` — **öneren dışında bir kişi** onaylar (dört göz ilkesi). Onaylanan öneri `{dataset['id']}.reviewed.json` dosyasına ACCEPTED olarak taşınır; etiketin kaynağı HUMAN_REVIEWED olur.
- `labels reject --dataset … --ids <entry_id> --approver "…" --reason "…"` reddeder (gerekçesiyle saklanır).
- `labels verify --dataset …` denetim zincirini ve üç dosyanın tutarlılığını kontrol eder.
- Onaylı etiketlerle skor: `report --run <koşu klasörü> --rescore --reviewed-labels ../evaluation/reviewed_labels/{dataset['id']}.reviewed.json` (denetim günlüğü varken yalnızca onaylanmış, sonradan değiştirilmemiş kayıtlar uygulanır). Dikkat: bu komut koşu klasöründeki metrics.json ve report.md dosyalarını yeniden yazar; raporlarda atıf yapılan bir koşuda önce klasörün bir kopyasını alın.

{pending_line}Veri seti dosyası hiçbir adımda değişmez; değişiklik geçmişi korunur.
'''


def write_expert_review(run_dir, out_dir, dataset_path=None) -> dict:
    """Write expert_review.csv/.json/.xlsx and README.md for one run directory into ``out_dir``."""
    rows, meta = export_rows(Path(run_dir), dataset_path)
    manifest_path = Path(run_dir) / 'manifest.json'
    dataset_file = find_dataset(Path(run_dir), json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}, dataset_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    readme = readme_markdown(meta, dataset_file)
    paths = {'csv': out_dir / 'expert_review.csv', 'json': out_dir / 'expert_review.json', 'xlsx': out_dir / 'expert_review.xlsx',
             'readme': out_dir / 'README.md'}
    write_csv(paths['csv'], rows)
    paths['json'].write_text(json.dumps({**meta, 'readme': 'README.md', 'rows': rows}, ensure_ascii=False, indent=1), encoding='utf-8')
    write_xlsx(paths['xlsx'], rows, readme_lines=[line for line in readme.splitlines()], title=f'Cardaman expert review — {meta["run_name"]}')
    paths['readme'].write_text(readme, encoding='utf-8')
    # The operator's console still gets the local file; the package itself does not.
    return {'files': {key: str(path) for key, path in paths.items()}, **meta['counts'], 'dataset': str(dataset_file)}


def add_command(commands):
    export = commands.add_parser('expert-export', help='Expert review package of a run: expert_review.csv/.json/.xlsx and README.md '
                                                       '(nothing in the run or the dataset is written)')
    export.add_argument('--run', required=True)
    export.add_argument('--dataset', default=None, help='Dataset file (default: the one the manifest records, else evaluation/datasets/<id>.json)')
    export.add_argument('--out', required=True, help='Output directory')


def run_command(args) -> int:
    summary = write_expert_review(Path(args.run), Path(args.out), args.dataset)
    print(json.dumps(summary, ensure_ascii=False))
    return 0
