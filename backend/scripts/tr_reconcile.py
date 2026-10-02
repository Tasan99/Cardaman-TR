"""Reconcile two row dumps (tr_rows_dump.py) by stable keys. argv: old.json new.json out.json [label_old label_new]

Units kept apart:
  raw rows            one gap row = (pack, obligation_id, target_id); what the self-test 'rows' figure counts, per pack
  unique obligations  obligation_id with at least one row
  pair rows           (obligation_id, target_id) whatever the pack: a regulation two packs carry gives two raw rows, one pair
Every row of the old run missing from the new one is joined with the new routing decision of the same key.
"""
import collections
import json
import sys

old_d, new_d = (json.load(open(p, encoding='utf-8')) for p in sys.argv[1:3])
out = {'old': old_d['commit'], 'new': new_d['commit'], 'profiles': {}}
for pid in old_d['profiles']:
    o, n = old_d['profiles'][pid], new_d['profiles'][pid]
    key = lambda r: (r['pack'], r['obligation_id'], r['target_id'])
    orow, nrow = {key(r): r for r in o['rows']}, {key(r): r for r in n['rows']}
    assert len(orow) == len(o['rows']) and len(nrow) == len(n['rows']), 'duplicate row keys'
    ndec = {key(d): d for d in n['decisions']}
    odec = {key(d): d for d in o['decisions']}
    packs = sorted({k[0] for k in orow} | {k[0] for k in nrow})
    units = {}
    for label, rows in (('old', o['rows']), ('new', n['rows'])):
        units[label] = {'raw_rows_by_pack': {p: sum(r['pack'] == p for r in rows) for p in packs}, 'raw_rows': len(rows),
                        'unique_obligations': len({r['obligation_id'] for r in rows}),
                        'pairs': len({(r['obligation_id'], r['target_id']) for r in rows})}
    gone = sorted(set(orow) - set(nrow))
    added = sorted(set(nrow) - set(orow))
    by_pack = {p: {'old': units['old']['raw_rows_by_pack'][p], 'gone': sum(k[0] == p for k in gone), 'added': sum(k[0] == p for k in added),
                   'new': units['new']['raw_rows_by_pack'][p]} for p in packs}
    for p, v in by_pack.items():
        v['check'] = v['old'] - v['gone'] + v['added'] == v['new']
    detail = []
    for k in gone:
        d = ndec.get(k)
        detail.append({'pack': k[0], 'obligation_id': k[1], 'target_id': k[2], 'ref': orow[k]['ref'], 'regulation_id': orow[k]['regulation_id'],
                       'old_applicability': orow[k]['applicability'], 'new_status': d['status'] if d else 'NO_DECISION (obligation not read)',
                       'new_reasons': d['reasons'] if d else [], 'scope': d['scope'] if d else {},
                       'gates': [g for g in (d['gates'] if d else []) if g['gate'] in ('SALES_CHANNEL', 'PRODUCT_SCOPE', 'ACTIVITY_CLASS', 'ENTITY_CLASS')],
                       'flags': d['flags'] if d else [], 'text': d['text'] if d else ''})
    for k in added:
        d = odec.get(k)
        detail.append({'pack': k[0], 'obligation_id': k[1], 'target_id': k[2], 'ref': nrow[k]['ref'], 'regulation_id': nrow[k]['regulation_id'],
                       'added': True, 'old_status': d['status'] if d else 'NO_DECISION (obligation not read)',
                       'new_applicability': nrow[k]['applicability']})
    changed = [k for k in set(orow) & set(nrow) if (orow[k]['mapping'], orow[k]['decision']) != (nrow[k]['mapping'], nrow[k]['decision'])]
    reasons = collections.Counter((x['regulation_id'], x['new_status'], tuple(r for r in x['new_reasons'] if r.endswith(('MISMATCH', 'INCOMPLETE', 'UNCLEAR', 'CONFLICT')))) for x in detail if not x.get('added'))
    out['profiles'][pid] = {'units': units, 'by_pack': by_pack, 'gone': len(gone), 'added': len(added), 'same_key_changed_status': len(changed),
                            'gone_by_regulation_status_reason': [[*k, v] for k, v in reasons.most_common()], 'detail': detail}
    print('==', pid)
    print('   units old', units['old'], '\n   units new', units['new'])
    for p, v in by_pack.items():
        print('  ', p, v)
    print('   gone', len(gone), 'added', len(added), 'same key, other mapping/decision', len(changed))
    for k, v in reasons.most_common():
        print('     ', v, k)
json.dump(out, open(sys.argv[3], 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
