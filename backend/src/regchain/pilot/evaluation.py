"""Expert labelling worklist and exact-span extraction evaluation; no invented gold labels."""
from collections import Counter


def label_template(packet):
    data = packet['events'][0]['payload']
    return {'format': 'conc-pilot-labels-v1', 'analysis_head': packet['events'][0]['event_hash'],
        'reviewer': '', 'reviewer_role': '', 'independent_test_set': False,
        'cases': [{'label': c['source']['printed_label'], 'source_hash': c['source']['content_hash'],
                   'status': 'UNREVIEWED', 'rationale': '', 'expected_obligations': None}
                  for c in data['cases']],
        'decisions': [{'obligation_id': o['id'], 'status': 'UNREVIEWED',
                       'expected_applicability': None, 'expected_coverage': None, 'rationale': ''}
                      for o in data['obligations']]}


def evaluate(packet, labels):
    data = packet['events'][0]['payload']
    if labels.get('format') != 'conc-pilot-labels-v1' or labels['analysis_head'] != packet['events'][0]['event_hash']:
        raise ValueError('Expert labels belong to a different analysis')
    rows = {c['source']['printed_label']: c for c in data['cases']}
    seen = set()
    tp = fp = fn = reviewed = 0
    for case in labels['cases']:
        name = case['label']
        if name in seen or name not in rows or case['source_hash'] != rows[name]['source']['content_hash']:
            raise ValueError('Duplicate, unknown or stale labelled source')
        seen.add(name)
        if case['status'] not in ('EXPERT_REVIEWED', 'UNREVIEWED'):
            raise ValueError('Unknown case review status')
        if case['status'] != 'EXPERT_REVIEWED':
            continue
        if not labels['reviewer'].strip() or not labels['reviewer_role'].strip() or not case['rationale'].strip():
            raise ValueError('Reviewed labels require reviewer, role and rationale')
        if not isinstance(case['expected_obligations'], list):
            raise ValueError('Expert must supply an expected obligations list, including [] for a negative example')
        actual = Counter((o['subject'], o['modality'], o['required_action'] or o['prohibited_action'])
                         for o in rows[name]['output']['obligations'])
        expected = Counter()
        for o in case['expected_obligations']:
            if o['modality'] not in ('MUST', 'MUST_NOT', 'SHOULD', 'SHOULD_NOT', 'MAY', 'MAY_NOT'):
                raise ValueError('Unknown expert modality')
            if not o['subject'] or not o['action'] or o['subject'] not in rows[name]['source']['text'] or o['action'] not in rows[name]['source']['text']:
                raise ValueError('Gold subject/action must be exact source spans')
            expected[(o['subject'], o['modality'], o['action'])] += 1
        tp += sum((actual & expected).values())
        fp += sum((actual-expected).values())
        fn += sum((expected-actual).values())
        reviewed += 1
    if seen != set(rows):
        raise ValueError('Label worklist must account for all analyzed provisions')
    by_id = {o['id']: o for o in data['obligations']}
    decision_ids = set()
    measured = {'applicability': [0, 0], 'coverage': [0, 0]}
    for item in labels['decisions']:
        oid = item['obligation_id']
        if oid not in by_id or oid in decision_ids:
            raise ValueError('Unknown or duplicate labelled candidate')
        decision_ids.add(oid)
        if item['status'] not in ('EXPERT_REVIEWED', 'UNREVIEWED'):
            raise ValueError('Unknown decision review status')
        if item['status'] != 'EXPERT_REVIEWED':
            continue
        if not labels['reviewer'].strip() or not labels['reviewer_role'].strip() or not item['rationale'].strip():
            raise ValueError('Reviewed decision requires reviewer, role and rationale')
        # An expert may label a duty 'POSSIBLY_APPLIES' (a substantive match with facts still
        # missing); the proposal is scored correct only on the exact state, never on a near miss.
        for field, choices in [('applicability', ('APPLIES', 'POSSIBLY_APPLIES', 'DOES_NOT_APPLY', 'UNKNOWN')),
                               ('coverage', ('COVERS_TEXT', 'PARTIAL', 'CONFLICT', 'NO_EVIDENCE', 'UNKNOWN'))]:
            expected = item['expected_'+field]
            if expected not in choices:
                raise ValueError('Expert decision label is missing or invalid')
            measured[field][1] += 1
            measured[field][0] += by_id[oid]['proposal'][field] == expected
    if decision_ids != set(by_id):
        raise ValueError('Decision worklist must account for every candidate')
    reviewed_decisions = measured['coverage'][1]
    complete = reviewed == len(rows) and reviewed_decisions == len(by_id)
    return {'status': 'EXPERT_REVIEW_PENDING' if not reviewed and not reviewed_decisions else 'LABELLED' if complete else 'PARTIAL',
        'reviewed_provisions': reviewed, 'total_provisions': len(rows),
        'reviewed_decisions': reviewed_decisions, 'total_decisions': len(by_id),
        'extraction_exact_span': {'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp/(tp+fp) if tp+fp else None, 'recall': tp/(tp+fn) if tp+fn else None},
        'decisions': {k: {'correct': v[0], 'labelled': v[1], 'accuracy': v[0]/v[1] if v[1] else None} for k, v in measured.items()},
        'independent_test_set': labels.get('independent_test_set', False),
        'limitations': 'Exact source-span metric; equivalent wording may differ. Labels and expert identity are operator assertions. No legal accuracy claim without suitable independent expert evaluation.'}
