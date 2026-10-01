"""Label audit and reviewed labels (v0.18): a golden label that looks wrong goes to a person.

Nothing here writes the dataset. ``audit`` lists the suspicious rows of one or more runs for a
reviewer: a label left in dispute by the 23 September review (taxonomy.DISPUTED_LABELS), a row
every given run answers the same way and the label another (only with at least two runs: one
run disagreeing with a label is an error, not evidence against the label), and a row the error
taxonomy classified LABEL_AMBIGUITY. Each item carries the regulation text the label rests on
(the clause as the packet quotes it) and the reason it is listed.

A reviewer's decision goes to a separate reviewed-labels file
(evaluation/reviewed_labels/<dataset>.reviewed.json), one entry per changed field with the
reviewer, the date, the reason and a status. Only ACCEPTED entries are laid over the labels, in
memory, by ``report --rescore --reviewed-labels FILE``; the dataset file and evaluation/labels
stay as they were, so the dataset's history is kept and every re-scored number names the
reviewed file (and its sha256) it used.

v0.19: a change now reaches the reviewed file only through the approval flow of pending.py
(``labels propose`` -> PENDING -> ``labels approve`` by a second person), which records every step
in a hash-chained audit log beside the reviewed file. In strict mode ``apply_reviewed`` lays over
only the ACCEPTED entries that log approved, unedited; strict is on by default as soon as that log
exists, so a status typed by hand no longer changes a score once the flow is in use. An applied
change marks the field HUMAN_REVIEWED (label_sources), so metrics split by label source see it.
"""
import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from .identifiers import clause_id

AUDIT_FORMAT = 'cardaman-label-audit-v1'
REVIEWED_FORMAT = 'cardaman-reviewed-labels-v1'
# The label fields a reviewed entry may change, and the keyed view of the case that mirrors each.
EDITABLE = {'applicability': 'expected_applicability', 'entity_gate': 'expected_entity_gate', 'coverage': 'expected_policy_coverage',
            'conflict': 'expected_conflicts', 'evidence': 'expected_evidence', 'required': None}
ENTRY_FIELDS = ('case_id', 'article', 'clause', 'field', 'from', 'to', 'reviewer', 'reviewed_at', 'reason', 'status')
REVIEWED_NOTE = ('Reviewed label changes for this dataset. Nothing is changed automatically: the label audit '
                 '(python -m regchain.evaluation label-audit) only lists suspicious labels. A change a reviewer decides is one '
                 'entry {case_id, article, clause, action_keywords (optional, to single out one duty of a clause), field '
                 '(applicability | entity_gate | coverage | conflict | evidence | required), from (the label as it is now), to, '
                 'reviewer, reviewed_at (ISO date), reason, status: ACCEPTED | REJECTED}. Only ACCEPTED entries are laid over the '
                 'labels, in memory, by `report --rescore --reviewed-labels <this file>`; an entry whose "from" no longer matches '
                 'the dataset is skipped. The dataset file and evaluation/labels are never edited, so the dataset history is kept. '
                 'dataset_sha256 is the sha256 of the dataset file the entries were reviewed against.')


def file_sha256(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def reviewed_template(dataset_path: Path) -> dict:
    """An empty reviewed-labels file for a dataset (no entries: nothing is changed)."""
    from .harness import load_dataset
    dataset = load_dataset(Path(dataset_path))
    return {'format': REVIEWED_FORMAT, 'dataset_id': dataset.dataset_id, 'dataset_version': dataset.version,
            'dataset_path': Path(dataset_path).as_posix(), 'dataset_sha256': file_sha256(dataset_path), 'note': REVIEWED_NOTE, 'entries': []}


def row_id(case_id, expected):
    return (case_id, str(expected['article']), clause_id(expected.get('clause') or ''), tuple(expected.get('action_keywords') or []))


def find_targets(case, article, clause='', action_keywords=None) -> list:
    """The expectations (dicts of a dumped case) an entry names: same article and clause, and the
    same action keywords when the entry gives them. A review entry applies only when exactly one matches."""
    return [e for e in (case or {}).get('expected_obligations', []) if e['article'] == str(article)
            and clause_id(e['clause']) == clause_id(clause or '')
            and (not action_keywords or e['action_keywords'] == list(action_keywords))]


def resolved_source(target: dict, field: str, dataset_default=None) -> str:
    """The provenance of one label field of a dumped expectation (per-field, expectation, dataset, RULE_DERIVED)."""
    return (target.get('label_sources') or {}).get(field) or target.get('label_source') or dataset_default or 'RULE_DERIVED'


def dataset_id_of(run_dir, dataset_path=None) -> str:
    """The id of the dataset a run's rows are labelled by: the given dataset's, else the one the manifest records.

    Disputed labels belong to their dataset (taxonomy.disputed, B5): an independent dataset's C12 md.5(2) is not
    tr-aml-v1's disputed row, so the audit must look disputes up under the run's own dataset id.
    """
    from .taxonomy import LEGACY_DATASET
    if dataset_path:
        from .harness import load_dataset
        return load_dataset(Path(dataset_path)).dataset_id
    manifest_path = Path(run_dir) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}
    return (manifest.get('dataset') or {}).get('id') or LEGACY_DATASET


def run_outputs(run_dir, dataset_path=None):
    """{row id: what this run said about the labelled row} and the run's taxonomy records by row id."""
    from .taxonomy import classify_results, run_results
    results, _, packets, scored_against = run_results(Path(run_dir), dataset_path)
    report = classify_results(results, packets, dataset_id_of(run_dir, dataset_path))
    taxonomy = {}
    for kind in ('applicability_errors', 'coverage_errors'):
        for record in report[kind]:
            taxonomy.setdefault((record['case_id'], record['expectation']['article'], clause_id(record['expectation']['clause'] or ''),
                                 tuple(record['expectation']['action_keywords'] or [])), []).append(record)
    rows = {}
    for result in results:
        payload = packets.get(result['case_id']) or {}
        obligations = {o.get('id'): o for o in payload.get('obligations') or []}
        by_key = {}
        for prediction in result.get('predictions') or []:
            by_key.setdefault(prediction['key'], prediction)
        for item in result.get('scored') or []:
            prediction = by_key.get(item.get('prediction')) if item.get('matched') else None
            obligation = obligations.get((prediction or {}).get('obligation_id')) or {}
            rows[row_id(result['case_id'], item['expected'])] = {
                'expected': item['expected'], 'matched': item.get('matched'),
                'applicability': item['applicability'][1] if item.get('matched') else 'NOT_EXTRACTED',
                'coverage': item['coverage'][1] if 'coverage' in item else None,
                'conflict': item['conflict'][1] if 'conflict' in item else None,
                'obligation_key': (prediction or {}).get('key'),
                'regulation_text': (obligation.get('candidate') or {}).get('source_quote') or
                                   ((obligation.get('proposal') or {}).get('applicability_scope') or {}).get('child_excerpt'),
                'provision': obligation.get('source_label')}
    return rows, taxonomy, scored_against


def audit(run_dirs, dataset_path=None, min_agreeing_runs=2):
    """The suspicious-label review list over one or more runs (nothing is written)."""
    from .taxonomy import disputed
    runs = []
    for run_dir in run_dirs:
        rows, taxonomy, scored_against = run_outputs(run_dir, dataset_path)
        manifest_path = Path(run_dir) / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}
        runs.append({'name': Path(run_dir).name, 'run': str(run_dir), 'manifest_sha256': manifest.get('manifest_sha256'),
                     'scored_against': scored_against, 'rows': rows, 'taxonomy': taxonomy, 'dataset_id': dataset_id_of(run_dir, dataset_path)})
    ids = list(dict.fromkeys(key for run in runs for key in run['rows']))
    items = []
    for key in ids:
        case_id, article, clause, _ = key
        present = [run for run in runs if key in run['rows']]
        expected = present[0]['rows'][key]['expected']
        reasons = []
        dispute = disputed(case_id, article, clause, present[0]['dataset_id'])
        if dispute:
            reasons.append(f'DISPUTED ({dispute[0]}): {dispute[1]}')
        if len(runs) >= min_agreeing_runs and len(present) == len(runs):
            for dimension, label in (('applicability', expected['applicability']), ('coverage', expected.get('coverage')),
                                     ('conflict', expected.get('conflict'))):
                said = [run['rows'][key][dimension] for run in runs]
                if dimension != 'applicability' and any(value is None for value in said):
                    continue                                  # not asserted by the label, or not assessed in some run
                if 'NOT_EXTRACTED' in said or len(set(map(str, said))) != 1 or said[0] == label:
                    continue
                reasons.append(f'ALL_RUNS_DISAGREE ({dimension}): every run ({len(runs)}) says {said[0]}, the label says {label}.')
        for run in present:
            for record in run['taxonomy'].get(key, []):
                if record['taxonomy'] == 'LABEL_AMBIGUITY' and not (dispute and record['rationale'].startswith('Disputed label')):
                    reasons.append(f'LABEL_AMBIGUITY in {run["name"]}: {record["rationale"]}')
        if not reasons:
            continue
        text = next((run['rows'][key]['regulation_text'] for run in present if run['rows'][key]['regulation_text']), None)
        items.append({'case_id': case_id, 'where': f'{case_id} md.{article}' + (f'({clause})' if clause else ''),
                      'obligation_key': next((run['rows'][key]['obligation_key'] for run in present if run['rows'][key]['obligation_key']), None),
                      'provision': next((run['rows'][key]['provision'] for run in present if run['rows'][key]['provision']), None),
                      'expected': {k: expected.get(k) for k in ('applicability', 'entity_gate', 'coverage', 'conflict', 'evidence', 'notes')},
                      'system_output': {run['name']: {k: run['rows'][key][k] for k in ('applicability', 'coverage', 'conflict')}
                                        if key in run['rows'] else None for run in runs},
                      'regulation_text': text, 'reasons': list(dict.fromkeys(reasons))})
    record = {'format': AUDIT_FORMAT, 'created_at': datetime.now(timezone.utc).isoformat(),
              'runs': [{k: run[k] for k in ('name', 'run', 'manifest_sha256', 'scored_against')} for run in runs],
              'min_agreeing_runs': min_agreeing_runs,
              'note': 'A review list, not a decision: no label is changed. A change a reviewer accepts goes to the reviewed-labels file '
                      '(evaluation/reviewed_labels/), never to the dataset.', 'items': items}
    if dataset_path:
        record['dataset'] = {'path': str(dataset_path), 'sha256': file_sha256(dataset_path)}
    return record


def audit_markdown(record):
    lines = ['# Label audit (review list)', '', record['note'], '',
             'Runs: ' + ', '.join(f'`{run["name"]}` (scored against {run["scored_against"]})' for run in record['runs']) + '.', '',
             f'{len(record["items"])} rows listed. A row is listed when its label is disputed, when every run (at least '
             f'{record["min_agreeing_runs"]}) gives the same answer and the label another, or when the error taxonomy calls it LABEL_AMBIGUITY.', '',
             '| Row | Expected | ' + ' | '.join(run['name'] for run in record['runs']) + ' | Reasons |',
             '|---|---|' + '---|' * len(record['runs']) + '---|']
    show = lambda out: '—' if out is None else ' / '.join(str(out[k]) for k in ('applicability', 'coverage', 'conflict') if out[k] is not None)
    for item in record['items']:
        e = item['expected']
        lines.append(f'| {item["where"]} | {e["applicability"]} / {e["coverage"]} / {e["conflict"]} | '
                     + ' | '.join(show(item['system_output'][run['name']]) for run in record['runs'])
                     + ' | ' + '<br>'.join(r.replace('|', '/') for r in item['reasons']) + ' |')
    for item in record['items']:
        lines += ['', f'## {item["where"]}', '', f'- Provision: {item["provision"] or "—"}; obligation key `{item["obligation_key"] or "—"}`',
                  f'- Label note: {item["expected"]["notes"] or "—"}', f'- Expected evidence: {item["expected"]["evidence"] or "—"}',
                  '- Reasons: ' + ' '.join(item['reasons']), '', '> ' + ' '.join(str(item['regulation_text'] or '(no clause text in the packets)').split())]
    return '\n'.join(lines) + '\n'


def write_audit(run_dirs, out_stem=None, dataset_path=None, min_agreeing_runs=2):
    record = audit(run_dirs, dataset_path, min_agreeing_runs)
    stem = Path(out_stem) if out_stem else Path(run_dirs[0]) / 'label-audit'
    stem.parent.mkdir(parents=True, exist_ok=True)
    json_path, md_path = stem.with_name(stem.name + '.json'), stem.with_name(stem.name + '.md')
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding='utf-8')
    md_path.write_text(audit_markdown(record), encoding='utf-8')
    return record, json_path, md_path


def apply_reviewed(dataset, reviewed_path: Path, dataset_path=None, strict=None, audit_path=None):
    """The dataset with the ACCEPTED entries of a reviewed-labels file laid over it (a new object; no file is written).

    Returns (dataset, overlay record). An entry is skipped, and the record says why, when it is not
    ACCEPTED, lacks a field, names no single expectation, or its "from" is not the label as it is now.

    ``strict`` (v0.19) also skips an ACCEPTED entry that the audit log (``audit_path``, default
    <dataset_id>.audit.jsonl beside the reviewed file) does not approve: no APPROVED event for its
    entry_id, an entry edited after its approval, or a log whose hash chain does not verify. None
    (the default) means strict exactly when that log exists: a v0.18 reviewed file without a log is
    read as before, and once the approval flow is in use a hand-typed ACCEPTED changes nothing.
    """
    from typing import get_args
    from .schema import Dataset, LabelField
    reviewed_path = Path(reviewed_path)
    # A reviewer edits this file by hand; Windows editors (and PowerShell 5.1's Set-Content -Encoding utf8)
    # write a byte-order mark that json.loads refuses.
    reviewed = json.loads(reviewed_path.read_text(encoding='utf-8-sig'))
    if reviewed.get('format') != REVIEWED_FORMAT:
        raise ValueError(f'{reviewed_path} is not a {REVIEWED_FORMAT} file')
    if reviewed.get('dataset_id') != dataset.dataset_id:
        raise ValueError(f'{reviewed_path} reviews {reviewed.get("dataset_id")}, not {dataset.dataset_id}')
    audit_path = Path(audit_path) if audit_path else reviewed_path.with_name(f'{dataset.dataset_id}.audit.jsonl')
    strict = audit_path.is_file() if strict is None else bool(strict)
    approvals, chain_problems = {}, []
    if strict:
        from .pending import approvals as approved_events
        approvals, chain_problems = approved_events(audit_path)
    data = json.loads(dataset.model_dump_json())
    cases = {case['case_id']: case for case in data['cases']}
    applied, skipped = [], []
    for index, entry in enumerate(reviewed.get('entries') or []):
        name = f'#{index} {entry.get("case_id")} md.{entry.get("article")}{entry.get("clause") or ""} {entry.get("field")}'
        missing = [key for key in ENTRY_FIELDS if key not in entry or (entry[key] in (None, '') and key not in ('clause', 'from', 'to'))]
        if entry.get('status') != 'ACCEPTED':
            skipped.append({'entry': name, 'reason': f'status {entry.get("status")}'})
            continue
        if strict:
            from .pending import entry_digest
            event = approvals.get(entry.get('entry_id'))
            reason = (f'audit log fails verification ({chain_problems[0]})' if chain_problems else
                      'no approval record' if event is None else
                      'the entry differs from the one approved (edited after approval)' if event['entry_sha256'] != entry_digest(entry) else None)
            if reason:
                skipped.append({'entry': name, 'reason': reason})
                continue
        if missing:
            skipped.append({'entry': name, 'reason': 'missing ' + ', '.join(missing)})
            continue
        if entry['field'] not in EDITABLE:
            skipped.append({'entry': name, 'reason': f'field {entry["field"]} cannot be reviewed'})
            continue
        case = cases.get(entry['case_id'])
        targets = find_targets(case, entry['article'], entry.get('clause'), entry.get('action_keywords'))
        if len(targets) != 1:
            skipped.append({'entry': name, 'reason': f'{len(targets)} expectations match (one is needed)'})
            continue
        target = targets[0]
        if target[entry['field']] != entry['from']:
            # v0.19: which audited entry, and the label it met, so labels verify can tell a change a later dataset version
            # already contains from an approved decision that silently no longer applies.
            skipped.append({'entry': name, 'reason': f'the label is now {target[entry["field"]]!r}, the review changes {entry["from"]!r}',
                            **({'entry_id': entry['entry_id'], 'label_now': target[entry['field']]} if entry.get('entry_id') else {})})
            continue
        before = resolved_source(target, entry['field'], data.get('default_label_source'))
        target[entry['field']] = entry['to']
        after = None
        if entry['field'] in get_args(LabelField):
            # The field now rests on a person's decision, whatever the generator rule said.
            after = entry.get('label_source_after') or 'HUMAN_REVIEWED'
            target['label_sources'] = {**(target.get('label_sources') or {}), entry['field']: after}
        view = EDITABLE[entry['field']]
        if view:
            key = f'{case["regulation_id"]}/md.{target["article"]}/{clause_id(target["clause"]) or "*"}'
            if entry['to'] in (None, []) and view in ('expected_conflicts', 'expected_evidence'):
                case[view].pop(key, None)
            else:
                case[view][key] = entry['to']
        applied.append({'entry': name, 'from': entry['from'], 'to': entry['to'], 'reviewer': entry['reviewer'],
                        'reviewed_at': entry['reviewed_at'], 'reason': entry['reason'], 'entry_id': entry.get('entry_id'),
                        'approved_by': entry.get('approved_by'), 'label_source_before': before, 'label_source_after': after})
    overlaid = Dataset.model_validate(data)            # the reviewed labels must still satisfy the schema
    actual = file_sha256(dataset_path) if dataset_path and Path(dataset_path).is_file() else None
    return overlaid, {'path': str(reviewed_path), 'sha256': file_sha256(reviewed_path), 'dataset_sha256_reviewed': reviewed.get('dataset_sha256'),
                      'dataset_sha256': actual, 'same_dataset': actual is not None and actual == reviewed.get('dataset_sha256'),
                      'strict': strict, 'audit': {'path': str(audit_path), 'problems': chain_problems} if strict else None,
                      'applied': applied, 'skipped': skipped}
