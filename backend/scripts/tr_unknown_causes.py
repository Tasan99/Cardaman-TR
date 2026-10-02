"""Why applicability is UNKNOWN: the review records of one pilot, split by cause, with the evidence the text offers.

    PYTHONPATH=<tree>/backend/src python backend/scripts/tr_unknown_causes.py <profile_id> <out.json>

Reads, never decides: no addressee is assigned, nothing is routed differently. For a clause whose addressee the rules
leave UNCLEAR it records where an addressee might be found in the source - the sentence before in the same paragraph,
the lead-in of a list, the regulation's scope article (catalogue activities), an explicit cross-reference - and which
rule made it UNCLEAR (a statute's clause without a mapped actor, an actor word the vocabulary has no class for, a list
item mixing product classes). For a missing company fact it records the gate, what the duty requires and what the
profile states. Policy evidence plays no part here: these duties' coverage is not assessed.
"""
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

import regchain
from regchain.tr.adjudicate import assess_profile
from regchain.tr.compare import load_register
from regchain.tr.corpus import CorpusStore
from regchain.tr.engines import ExpertServices, engines_for
from regchain.tr.extraction import extract_regulation, regulation_scope
from regchain.tr.frames import fold
from regchain.tr.packs import Registry
from regchain.tr.profile import load_pilot_profiles

CROSS_REFERENCE = re.compile(r'(?:\d+\s*(?:inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü)\s+madde|bu (?:madde|fıkra|bent)|(?:birinci|ikinci|üçüncü|dördüncü|beşinci) fıkra|'
                             r'yukarıda|aşağıdaki|belirtilen|sayılan)')
src = Path(regchain.__file__).resolve().parents[1]
commit = subprocess.run(['git', '-C', str(src), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
profile_id, out_path = sys.argv[1], Path(sys.argv[2])
registry, store = Registry.load(), CorpusStore()
profile = next(p for p in load_pilot_profiles(registry.vocabulary).values() if p.profile_id == profile_id)
rec = src / 'regchain' / 'tr' / 'data' / 'evaluation' / 'recorded' / '20261001'
services = ExpertServices.recorded(rec / 'similarities.json', [rec / f'adjudicate-{s}.jsonl' for s in ('dev', 'holdout', 'validation')], registry, store)
register = load_register(profile_id)

records = []
frames_of_reg = {}
for engine, _ in engines_for(profile, services):
    report = assess_profile(profile, engine.obligations(), register, registry, store, services.table, services.adjudicator)[0]
    by_id = {o.obligation_id: o for o in engine.obligations()}
    for review in report.applicability_reviews:
        obligation = by_id[review.obligation_id]
        frame = obligation.frame
        if obligation.regulation_id not in frames_of_reg:
            frames = extract_regulation(obligation.regulation_id, registry, store)[0]
            meta = registry.regulations[obligation.regulation_id]
            frames_of_reg[obligation.regulation_id] = (frames, regulation_scope(frames, meta, registry.vocabulary))
        frames, scope = frames_of_reg[obligation.regulation_id]
        index = next((i for i, f in enumerate(frames) if f.ref == frame.ref), None)
        paragraph = frame.ref.split('/b.')[0].split('/c.')[0]
        before = [f for f in frames[:index] if f.ref.startswith(paragraph)] if index is not None else []
        prior_actors = sorted({a.id for f in before for a in f.actors if a.classes or a.entities})
        record = {'pack': engine.pack_id, 'review_id': review.review_id, 'obligation_id': review.obligation_id, 'provision_ref': review.provision_ref,
                  'regulation_id': review.regulation_id, 'statute': review.regulation_id.startswith('TR:KANUN:'), 'target_id': review.target_id,
                  'level': review.level, 'review_type': review.review_type, 'reason_codes': review.reason_codes,
                  'missing_facts': review.missing_facts, 'flags': obligation.flags, 'basis': obligation.basis, 'kind': frame.kind,
                  'marker': frame.marker, 'passive': frame.passive, 'list_item': frame.marker.endswith(('(chapeau)', '(closing)')),
                  'actor_words': [{'id': a.id, 'text': a.text, 'where': a.where, 'classes': a.classes, 'entities': a.entities} for a in frame.actors],
                  'authorities': [a.text for a in frame.authorities], 'chapeau': frame.chapeau[:200],
                  'prior_sentence_actors': prior_actors, 'scope_article_activities': list(scope.activity_classes),
                  'scope_article_basis': scope.basis, 'cross_reference': bool(CROSS_REFERENCE.search(fold(frame.text))),
                  'text': frame.text[:300]}
        records.append(record)


def cause(r):
    if r['review_type'] == 'COMPLETE_PROFILE_FACT':
        return 'COMPANY_FACT_MISSING'
    if r['review_type'] == 'RESOLVE_PROFILE_CONFLICT':
        return 'COMPANY_FACT_CONFLICT'
    if r['review_type'] == 'CHECK_SOURCE_GROUNDING':
        return 'SOURCE_NOT_GROUNDED'
    if r['review_type'] == 'CLARIFY_REGULATORY_SCOPE':
        if 'PRODUCT_SCOPE_MIXED' in r['flags']:
            return 'SOURCE_PRODUCT_SCOPE_MIXED'
        if 'ACTOR_UNMAPPED' in r['flags']:
            return 'SOURCE_ACTOR_WORD_WITHOUT_CLASS'
        return 'SOURCE_NO_ADDRESSEE_STATUTE' if r['statute'] else 'SOURCE_NO_ADDRESSEE_NO_SCOPE_ACTIVITIES'
    return 'OTHER'


def hint(r):
    """Where the source may name the addressee: evidence to read, never an assignment."""
    found = []
    if r['prior_sentence_actors']:
        found.append('PRIOR_SENTENCE_NAMES_A_PARTY')
    if r['chapeau'] and r['list_item']:
        found.append('LIST_LEAD_IN')
    if r['scope_article_activities']:
        found.append('SCOPE_ARTICLE_OR_CATALOGUE_ACTIVITIES')
    if r['cross_reference']:
        found.append('CROSS_REFERENCE')
    if r['authorities']:
        found.append('AUTHORITY_NAMED')
    if r['passive'] or not r['actor_words']:
        found.append('IMPERSONAL_OR_PASSIVE')
    return found or ['NONE_FOUND']


for r in records:
    r['cause'] = cause(r)
    r['source_hints'] = hint(r) if r['review_type'] == 'CLARIFY_REGULATORY_SCOPE' else []
summary = {'commit': commit, 'profile_id': profile_id, 'records': len(records),
           'by_cause': dict(collections.Counter(r['cause'] for r in records).most_common()),
           'by_pack': dict(collections.Counter(r['pack'] for r in records)), 'causes': {}}
for c in summary['by_cause']:
    group = [r for r in records if r['cause'] == c]
    summary['causes'][c] = {
        'records': len(group), 'unique_obligations': len({r['obligation_id'] for r in group}),
        'unique_provisions': len({r['provision_ref'] for r in group}), 'targets': dict(collections.Counter(r['target_id'] for r in group)),
        'regulations': dict(collections.Counter(r['regulation_id'] for r in group).most_common()),
        'missing_gates': dict(collections.Counter((f['gate'], tuple(f.get('required') or [])) and f['gate'] for r in group for f in r['missing_facts']).most_common()),
        'records_per_obligation': dict(collections.Counter(collections.Counter(r['obligation_id'] for r in group).values())),
        'source_hints_by_obligation': dict(collections.Counter(h for o in {r['obligation_id']: r for r in group}.values() for h in o['source_hints']).most_common()),
        'actor_words_unmapped': dict(collections.Counter(a['id'] for o in {r['obligation_id']: r for r in group}.values()
                                                         for a in o['actor_words'] if not a['classes'] and not a['entities']).most_common()),
    }
out_path.write_text(json.dumps({'summary': summary, 'records': records}, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=1))
