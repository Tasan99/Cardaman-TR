"""Every row of the old run that the new run does not have, with its source provision, the company profile evidence of
the new decision and the reason. argv: old.json new.json out.csv"""
import csv
import json
import sys

old_d, new_d = (json.load(open(p, encoding='utf-8')) for p in sys.argv[1:3])
key = lambda r: (r['pack'], r['obligation_id'], r['target_id'])
with open(sys.argv[3], 'w', encoding='utf-8-sig', newline='') as handle:
    writer = csv.writer(handle, delimiter=';')
    writer.writerow(['profile', 'pack', 'obligation_id', 'target_id', 'provision_ref', 'regulation_id', 'old_applicability', 'new_status',
                     'new_reason_codes', 'scope_constraints', 'profile_evidence', 'provision_text'])
    for pid in old_d['profiles']:
        orow = {key(r): r for r in old_d['profiles'][pid]['rows']}
        nrow = {key(r): r for r in new_d['profiles'][pid]['rows']}
        ndec = {key(d): d for d in new_d['profiles'][pid]['decisions']}
        for k in sorted(set(orow) - set(nrow)):
            r, d = orow[k], ndec.get(k)
            evidence = ' | '.join(f"{g['gate']} {g['status']} required={g.get('required')} stated={g.get('stated')} complete={g.get('complete')}"
                                  for g in (d['gates'] if d else []) if g['status'] in ('MISMATCH', 'UNDETERMINED'))
            writer.writerow([pid, k[0], k[1], k[2], r['ref'], r['regulation_id'], r['applicability'],
                             d['status'] if d else 'NO_DECISION: the clause is no longer read as a duty',
                             '; '.join(d['reasons']) if d else '', json.dumps(d['scope'], ensure_ascii=False) if d else '', evidence,
                             (d['text'] if d else '')[:400]])
print('written', sys.argv[3])
