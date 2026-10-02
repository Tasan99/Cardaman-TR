"""A grouping PROPOSAL for applicability reviews: counts and an example, nothing is changed in the export.

    python backend/scripts/tr_review_grouping.py <unknown-causes.json> <out.json>

Rules of the proposal: a parent task gathers records that share one uncertainty, one responsible department and one
due date (none is stated today); it keeps every entity's own decision and evidence as a child. Causes are never merged:
a source question (whom does the clause bind) goes to Legal and is the same for every entity, so one parent per clause;
a company-fact question goes to whoever keeps the profile and is the same for every clause it blocks, so one parent per
entity and missing fact, listing the clauses it unblocks.
"""
import collections
import json
import sys

data = json.load(open(sys.argv[1], encoding='utf-8'))
records = data['records']
DEPARTMENT = {'CLARIFY_REGULATORY_SCOPE': 'LEGAL', 'CHECK_SOURCE_GROUNDING': 'LEGAL', 'COMPLETE_PROFILE_FACT': 'COMPLIANCE',
              'RESOLVE_PROFILE_CONFLICT': 'COMPLIANCE'}


def topic_key(r):
    """What one parent task is about."""
    if r['review_type'] in ('CLARIFY_REGULATORY_SCOPE', 'CHECK_SOURCE_GROUNDING'):
        return (r['review_type'], r['cause'], r['obligation_id'])
    gates = tuple(sorted(f['gate'] for f in r['missing_facts'] if f['gate'] not in ('EXCEPTIONS',))) or ('EXCEPTIONS_ONLY',)
    return (r['review_type'], r['cause'], r['target_id'], gates)


def task_key(r):
    """What one user-facing task is: source questions of one article together (same department, same text to read);
    company-fact questions of one entity and one missing fact together."""
    key = topic_key(r)
    if r['review_type'] in ('CLARIFY_REGULATORY_SCOPE', 'CHECK_SOURCE_GROUNDING'):
        return (r['review_type'], r['cause'], r['provision_ref'].split('/')[0])
    return key


topics = collections.defaultdict(list)
tasks = collections.defaultdict(list)
for r in records:
    topics[topic_key(r)].append(r)
    tasks[task_key(r)].append(r)
counts = {'raw_records': len(records), 'unique_review_topics': len(topics), 'user_tasks': len(tasks),
          'by_type': {t: {'raw_records': sum(r['review_type'] == t for r in records),
                          'unique_review_topics': sum(k[0] == t for k in topics), 'user_tasks': sum(k[0] == t for k in tasks)}
                      for t in sorted({r['review_type'] for r in records})},
          'by_cause': {c: {'raw_records': sum(r['cause'] == c for r in records), 'unique_review_topics': sum(k[1] == c for k in topics),
                           'user_tasks': sum(k[1] == c for k in tasks)} for c in sorted({r['cause'] for r in records})}}


def parent(key, group):
    first = group[0]
    clauses = sorted({r['provision_ref'] for r in group})
    return {'task_id': 'APPLICABILITY-TASK:' + ':'.join(str(k) if not isinstance(k, tuple) else '+'.join(k) for k in key),
            'review_type': first['review_type'], 'cause': first['cause'], 'department': DEPARTMENT[first['review_type']], 'due_date': None,
            'question': ('Whom does this text bind? Read the clauses below in their article; name the addressee only from the text.'
                         if first['review_type'] == 'CLARIFY_REGULATORY_SCOPE' else
                         f"State the missing company fact(s) of {first['target_id']}: " + ', '.join(sorted({f['gate'] for r in group for f in r['missing_facts']}))),
            'clauses': clauses, 'records': len(group), 'entities': sorted({r['target_id'] for r in group}),
            'children': [{'review_id': r['review_id'], 'target_id': r['target_id'], 'provision_ref': r['provision_ref'], 'status': 'UNKNOWN',
                          'reason_codes': r['reason_codes'], 'missing_facts': r['missing_facts'], 'source_hints': r['source_hints'],
                          'text': r['text'][:160]} for r in group],
            'approval': {'required': True, 'status': 'PENDING'}, 'state': 'DRAFT'}


examples = []
for wanted in ('SOURCE_NO_ADDRESSEE_STATUTE', 'COMPANY_FACT_MISSING', 'SOURCE_ACTOR_WORD_WITHOUT_CLASS'):
    key = next((k for k in tasks if k[1] == wanted), None)
    if key:
        example = parent(key, tasks[key])
        example['children'] = example['children'][:6] + ([{'...': f"{len(tasks[key]) - 6} more children"}] if len(tasks[key]) > 6 else [])
        examples.append(example)
largest = sorted(((len(v), k) for k, v in tasks.items()), reverse=True)[:8]
out = {'counts': counts, 'largest_tasks': [{'records': n, 'key': [str(x) for x in k]} for n, k in largest], 'examples': examples}
json.dump(out, open(sys.argv[2], 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps({'counts': counts, 'largest_tasks': out['largest_tasks']}, ensure_ascii=False, indent=1))
