"""HOLDOUT coverage, case by case, with and without the recorded similarity table, for one code version.

    PYTHONPATH=<worktree>/backend/src python backend/scripts/tr_holdout_ab.py <out.json>

Same 28 cases, same labels (beverage_tr_dev_v2.json of that tree), same registers, same recorded table. Three readings:
  score_coverage      devset.score_coverage path (compare_obligation, no table): the 'coverage rules' figure of ai evaluate
  rules_no_table      devset.score_pipeline with table=None, no adjudications: its 'rule' word
  rules_with_table    devset.score_pipeline with the recorded table, no adjudications: its 'rule' word (rules_only_correct)
A case is right when the word is one of its expected words. Denominator: every case of the split.
"""
import json
import subprocess
import sys
from pathlib import Path

import regchain
from regchain.tr import devset
from regchain.tr.profile import load_pilot_profiles
from regchain.tr.semantic import SimilarityTable

src = Path(regchain.__file__).resolve().parents[1]
rec = src / 'regchain' / 'tr' / 'data' / 'evaluation' / 'recorded' / '20261001'
commit = subprocess.run(['git', '-C', str(src), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
dataset = devset.load()
readers = devset.Readers(dataset, None, None)
profiles = {p.profile_id: p for p in load_pilot_profiles(readers.registry.vocabulary).values()}
table = SimilarityTable.load(rec / 'similarities.json')
out = {'commit': commit, 'dataset': f"{dataset['dataset_id']} {dataset['version']}", 'table_model': table.model_version, 'splits': {}}
for split in ('DEV', 'HOLDOUT', 'VALIDATION'):
    cases = [c for c in dataset['coverage'] if c.get('split', 'DEV') == split]
    words = {}
    for case in cases:
        profile = profiles[case['profile_id']]
        register = devset.case_register(case, {})
        words[case['case_id']] = {'ref': case['ref'], 'target': case['target_id'], 'expected': case['expected'],
                                  'score_coverage': devset._word(devset._gap_rows(readers, case['ref'], profile, register, case['target_id']))}
    for key, tab in (('rules_no_table', None), ('rules_with_table', table)):
        result = devset.score_pipeline(dataset, readers, profiles, tab, {}, split)
        for item in result['cases']:
            words[item['case_id']][key] = item['rule']
    totals = {k: sum(w[k] in w['expected'] for w in words.values()) for k in ('score_coverage', 'rules_no_table', 'rules_with_table')}
    out['splits'][split] = {'n': len(cases), 'correct': totals, 'cases': words}
    print(commit, split, len(cases), totals, flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
