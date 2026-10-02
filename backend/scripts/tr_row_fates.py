"""The fate of every row of two runs, and of every new row in the QDMS export. argv: old_rows.json new_rows.json qdms_root out_dir

Key: (pack, obligation_id, target_id), one raw row of one pack. Fates of an old row: KEPT, KEPT_STATUS_CHANGED (mapping or
decision differs), MOVED_TO_APPLICABILITY_REVIEW (new routing UNKNOWN), OUT_OF_SCOPE (new routing DOES_NOT_APPLY, with
its reason codes), OBLIGATION_NOT_READ (the clause is no longer read as a duty). Fates of a new row that the old run did
not have: RE_ENTERED (old routing DOES_NOT_APPLY or UNKNOWN) or NEW_OBLIGATION. Each new row also carries its QDMS ledger
outcome (EXPORTED, MERGED_SAME_DUTY_AND_TARGET, EXCLUDED_COVERED_AUTO). Writes row-fates.csv and row-fates-summary.json.
"""
import collections
import csv
import json
import sys
from pathlib import Path

old_d, new_d = (json.load(open(p, encoding='utf-8')) for p in sys.argv[1:3])
qdms_root, out = Path(sys.argv[3]), Path(sys.argv[4])
key = lambda r: (r['pack'], r['obligation_id'], r['target_id'])
lines, summary = [], {'old_commit': old_d['commit'], 'new_commit': new_d['commit'], 'profiles': {}}
for pid in new_d['profiles']:
    o, n = old_d['profiles'][pid], new_d['profiles'][pid]
    orow, nrow = {key(r): r for r in o['rows']}, {key(r): r for r in n['rows']}
    odec, ndec = {key(d): d for d in o['decisions']}, {key(d): d for d in n['decisions']}
    ledger_path = qdms_root / f'qdms-{pid}' / 'export.json'
    ledger = {}
    if ledger_path.exists():
        for e in json.load(open(ledger_path, encoding='utf-8'))['ledger']:
            ledger[(e['pack'], e['row_id'])] = e['outcome']
    fates = collections.Counter()
    for k in sorted(set(orow) | set(nrow)):
        a, b = orow.get(k), nrow.get(k)
        if a and b:
            fate = 'KEPT' if (a['mapping'], a['decision']) == (b['mapping'], b['decision']) else 'KEPT_STATUS_CHANGED'
            detail = '' if fate == 'KEPT' else f"{a['mapping']}/{a['decision']} -> {b['mapping']}/{b['decision']}"
        elif a:
            d = ndec.get(k)
            fate = 'OBLIGATION_NOT_READ' if d is None else 'MOVED_TO_APPLICABILITY_REVIEW' if d['status'] == 'UNKNOWN' else \
                'OUT_OF_SCOPE' if d['status'] == 'DOES_NOT_APPLY' else 'OTHER'
            detail = '; '.join(d['reasons']) if d else ''
        else:
            d = odec.get(k)
            fate = 'NEW_OBLIGATION' if d is None else 'RE_ENTERED'
            detail = f"old routing {d['status']}: {'; '.join(d['reasons'])}" if d else ''
        row = b or a
        qdms = ledger.get((k[0], f'{k[1]}:{k[2]}'), '') if b else ''
        fates[fate] += 1
        lines.append([pid, k[0], k[1], k[2], row['ref'], row['regulation_id'], a['mapping'] if a else '', b['mapping'] if b else '', fate, detail, qdms])
    per_pack = {p: {'old': sum(k[0] == p for k in orow), 'new': sum(k[0] == p for k in nrow)} for p in sorted({k[0] for k in orow} | {k[0] for k in nrow})}
    summary['profiles'][pid] = {'fates': dict(fates), 'raw_rows_by_pack': per_pack,
                                'unique_duty_target_old': len({(k[1], k[2]) for k in orow}), 'unique_duty_target_new': len({(k[1], k[2]) for k in nrow}),
                                'qdms_outcomes_of_new_rows': dict(collections.Counter(l[10] for l in lines if l[0] == pid and l[7]))}
out.mkdir(parents=True, exist_ok=True)
with open(out / 'row-fates.csv', 'w', encoding='utf-8-sig', newline='') as handle:
    writer = csv.writer(handle, delimiter=';')
    writer.writerow(['profile', 'pack', 'obligation_id', 'target_id', 'provision_ref', 'regulation_id', 'old_mapping', 'new_mapping', 'fate', 'detail', 'qdms_outcome'])
    writer.writerows(lines)
total = collections.Counter()
for p in summary['profiles'].values():
    total.update(p['fates'])
summary['total_fates'] = dict(total)
summary['raw_rows_old'] = sum(v['old'] for p in summary['profiles'].values() for v in p['raw_rows_by_pack'].values())
summary['raw_rows_new'] = sum(v['new'] for p in summary['profiles'].values() for v in p['raw_rows_by_pack'].values())
(out / 'row-fates-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=1))
