"""One-line summaries of the self-test outputs (scripts/Test-TR.ps1): argv = kind, file."""
import json
import sys
from pathlib import Path

kind, path = sys.argv[1], Path(sys.argv[2])
raw = path.read_bytes()
text = raw.decode('utf-16') if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else raw.decode('utf-8', errors='replace')
try:
    data = json.loads(text)
except ValueError:
    print('not json: ' + ' | '.join(text.strip().splitlines()[-3:]))
    sys.exit(0)
if kind == 'assess':
    print(' ; '.join(f"{e['pack_id']}: rows {e['statistics']['rows']} escalated {e['statistics']['escalated_rows']} "
                     f"review {e['statistics']['decision_review_rows']} by-scope {e['statistics']['rows_applying_by_regulation_scope']} "
                     f"not-verified {e['statistics']['rows_not_verified']}" for e in data['engines']))
elif kind == 'evaluate':
    c, p = data['clauses'], data.get('pipeline', {})
    print('kind DEV', c['DEV']['tasks']['OBLIGATION_EXTRACTION/rule']['accuracy'], 'HOLDOUT', c['HOLDOUT']['tasks']['OBLIGATION_EXTRACTION/rule']['accuracy'],
          '| coverage rules DEV', data['coverage']['tasks']['POLICY_COVERAGE/rule']['accuracy'], 'HOLDOUT', data['coverage_holdout']['tasks']['POLICY_COVERAGE/rule']['accuracy'],
          '| pipeline', {s: {k: p[s]['metrics'][k] for k in ('auto_correct', 'auto_wrong', 'review_required', 'strict_accuracy', 'auto_false_covered', 'auto_false_contradicted')} for s in p})
elif kind == 'qdms':
    s = data['summary']
    print({k: s[k] for k in ('rows', 'by_policy_status', 'by_decision', 'actions', 'actions_by_state', 'ready_for_qdms')})
elif kind == 'live':
    s = data.get('selective', {})
    print({k: s.get(k) for k in ('rows', 'escalated_rows', 'adjudications', 'calls_failed', 'calls_without_thinking', 'decision_review_rows', 'model_ms_per_call', 'model_ms_per_obligation')})
