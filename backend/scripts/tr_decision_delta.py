"""Routing decisions of two row dumps (tr_rows_dump.py), key by key. argv: old.json new.json out.json

Key: (pack, obligation_id, target_id). A decision that disappears belongs to a clause no longer read as a duty; one that
appears belongs to a clause newly read. Every key whose status moved from UNKNOWN to an automatic status (APPLIES,
PARTIAL, DOES_NOT_APPLY), and every new key with an automatic status, is listed with its clause text: those are the
decisions to check against the source.
"""
import collections
import json
import sys

old_d, new_d = (json.load(open(p, encoding='utf-8')) for p in sys.argv[1:3])
key = lambda d: (d['pack'], d['obligation_id'], d['target_id'])
out = {'old': old_d['commit'], 'new': new_d['commit'], 'profiles': {}}
for pid in new_d['profiles']:
    od = {key(d): d for d in old_d['profiles'][pid]['decisions']}
    nd = {key(d): d for d in new_d['profiles'][pid]['decisions']}
    moves = collections.Counter()
    to_check = []
    for k in set(od) | set(nd):
        a, b = od.get(k), nd.get(k)
        before = a['status'] if a else 'NOT_READ'
        after = b['status'] if b else 'NOT_READ'
        if before != after:
            moves[(before, after)] += 1
        if after in ('APPLIES', 'PARTIAL', 'DOES_NOT_APPLY') and before in ('UNKNOWN', 'NOT_READ'):
            to_check.append({'pack': k[0], 'obligation_id': k[1], 'target_id': k[2], 'ref': b['ref'], 'before': before, 'after': after,
                             'reasons': b['reasons'], 'scope': b['scope'], 'text': b['text']})
    unclear = lambda ds: sum(d['status'] == 'UNKNOWN' and 'REGULATORY_SCOPE_UNCLEAR' in d['reasons'] for d in ds.values())
    incomplete = lambda ds: sum(d['status'] == 'UNKNOWN' and 'REGULATORY_SCOPE_UNCLEAR' not in d['reasons'] for d in ds.values())
    out['profiles'][pid] = {
        'status_old': dict(collections.Counter(d['status'] for d in od.values())), 'status_new': dict(collections.Counter(d['status'] for d in nd.values())),
        'source_unclear_old': unclear(od), 'source_unclear_new': unclear(nd), 'other_unknown_old': incomplete(od), 'other_unknown_new': incomplete(nd),
        'moves': {f'{a} -> {b}': n for (a, b), n in sorted(moves.items(), key=lambda kv: -kv[1])},
        'new_automatic_decisions': sorted(to_check, key=lambda x: (x['ref'], x['target_id']))}
json.dump(out, open(sys.argv[3], 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for pid, p in out['profiles'].items():
    print('==', pid)
    print('   source-unclear UNKNOWN', p['source_unclear_old'], '->', p['source_unclear_new'], '| other UNKNOWN', p['other_unknown_old'], '->', p['other_unknown_new'])
    print('   status', p['status_old'], '->', p['status_new'])
    print('   moves', p['moves'])
    by_ref = collections.Counter((x['ref'], x['before'], x['after']) for x in p['new_automatic_decisions'])
    print('   new automatic decisions', len(p['new_automatic_decisions']), dict(by_ref))
