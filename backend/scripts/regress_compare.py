"""Compare two unittest outputs by failure identity AND reason (the last line of each traceback), not by count.
argv: baseline.txt now.txt"""
import re, sys
from collections import Counter


def failures(path):
    text = open(path, encoding='utf-8', errors='replace').read()
    out = {}
    for block in re.split(r'\n={70}\n', text)[1:]:
        head, _, body = block.partition('\n' + '-' * 70 + '\n')
        if not head.startswith(('ERROR:', 'FAIL:')):
            continue
        body = body.split('\n' + '-' * 70)[0].rstrip()
        lines = [l for l in body.splitlines() if l.strip()]
        reason = next((l for l in reversed(lines) if re.match(r'^[A-Za-z_.]+(Error|Exception|Failure|Exit|Interrupt)?\b.*:', l) and not l.startswith(' ')), lines[-1] if lines else '')
        # paths, addresses, temp names and durations differ between runs without the reason differing
        reason = re.sub(r'0x[0-9A-Fa-f]+', '0x…', reason)
        reason = re.sub(r'tmp[a-z0-9_]{6,}', 'tmp…', reason)
        out[head.strip()] = reason.strip()[:300]
    return out


base, now = failures(sys.argv[1]), failures(sys.argv[2])
tail = [l for l in open(sys.argv[2], encoding='utf-8', errors='replace').read().splitlines() if l.strip()][-3:]
print('\n'.join(tail))
print('baseline failures', len(base), '| now', len(now))
only_base, only_now = sorted(set(base) - set(now)), sorted(set(now) - set(base))
changed = sorted(k for k in set(base) & set(now) if base[k] != now[k])
print('fixed since baseline (in baseline only):', len(only_base))
for k in only_base[:30]:
    print('   -', k)
print('NEW failures (not in baseline):', len(only_now))
for k in only_now[:30]:
    print('   +', k, '|', now[k])
print('same test, different reason:', len(changed))
for k in changed[:30]:
    print('   ~', k, '\n       was:', base[k], '\n       now:', now[k])
kinds = Counter(re.split(r'[:(]', v)[0] for v in now.values())
print('reasons now by exception type:', dict(kinds.most_common(8)))
print('VERDICT:', 'IDENTICAL failure identities and reasons' if not (only_base or only_now or changed) else 'DIFFERS')
