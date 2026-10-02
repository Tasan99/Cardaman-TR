"""Dump every assessed row and every routing decision of the three pilots, with stable keys, for one code version.

    PYTHONPATH=<worktree>/backend/src python backend/scripts/tr_rows_dump.py <out.json>

Same inputs as scripts/Test-TR.ps1 step 3: the packaged corpus, the pilot profiles and synthetic registers, the recorded
similarity table and adjudications of 20261001 replayed (no model). Keys: row = (pack, obligation_id, target_id);
decision = (pack, obligation_id, target_id) for every target of the scope's level, whatever the status.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import regchain
from regchain.tr.compare import COMPARE_RULES_VERSION, load_register
from regchain.tr.engines import ExpertServices, engines_for
from regchain.tr.extraction import route
from regchain.tr.packs import Registry
from regchain.tr.corpus import CorpusStore
from regchain.tr.profile import load_pilot_profiles

src = Path(regchain.__file__).resolve().parents[1]
data = src / 'regchain' / 'tr' / 'data'
rec = data / 'evaluation' / 'recorded' / '20261001'
commit = subprocess.run(['git', '-C', str(src), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
registry, store = Registry.load(), CorpusStore()
services = ExpertServices.recorded(rec / 'similarities.json', [rec / f'adjudicate-{s}.jsonl' for s in ('dev', 'holdout', 'validation')],
                                   registry, store)
out = {'commit': commit, 'src': str(src), 'compare_rules': COMPARE_RULES_VERSION, 'started': time.strftime('%Y-%m-%d %H:%M:%S'),
       'inputs': {'similarities': str(rec / 'similarities.json'), 'adjudications': 'adjudicate-{dev,holdout,validation}.jsonl', 'model': 'none (recorded replay)'},
       'versions': {rid: (store.head(rid).version_id if store.head(rid) else None) for rid in sorted(registry.regulations)},
       'profiles': {}}
for profile in load_pilot_profiles(registry.vocabulary).values():
    register = load_register(profile.profile_id)
    rows, decisions = [], []
    for engine, selection in engines_for(profile, services):
        report, assessments, run = engine.assess(profile, register, selection)
        for row, a in zip(report.rows, assessments):
            rows.append({'pack': engine.pack_id, 'obligation_id': row.obligation_id, 'target_id': row.target_id, 'entity_id': row.entity_id,
                         'level': row.level, 'regulation_id': row.regulation_id, 'ref': row.provision_ref, 'applicability': row.applicability,
                         'mapping': row.mapping.status, 'coverage': row.document_coverage, 'decision': a.decision,
                         'review_reasons': a.review_reasons, 'basis': a.applicability_basis})
        for obligation in engine.obligations():
            for d in route(obligation, profile, registry, store)[0]:
                decisions.append({'pack': engine.pack_id, 'obligation_id': obligation.obligation_id, 'target_id': d.target_id,
                                  'entity_id': d.entity_id, 'level': d.level, 'regulation_id': obligation.regulation_id,
                                  'ref': obligation.provision_ref, 'status': d.status, 'reasons': d.reason_codes,
                                  'gates': [{k: v for k, v in g.items() if k in ('gate', 'status', 'required', 'stated', 'complete')} for g in d.gates],
                                  'scope': {k: v for k, v in obligation.scope.model_dump().items()
                                            if k in ('level', 'sales_channels', 'product_attributes', 'product_classes', 'activity_classes',
                                                     'entity_classes', 'alcohol_scope', 'scope_status') and v},
                                  'flags': obligation.flags, 'text': obligation.text[:400]})
    out['profiles'][profile.profile_id] = {'rows': rows, 'decisions': decisions}
    print(profile.profile_id, len(rows), 'rows', len(decisions), 'decisions', flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
print('written', sys.argv[1])
