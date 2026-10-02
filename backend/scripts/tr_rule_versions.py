"""Check or record the rule version labels (regchain.tr.rule_versions).

    python backend/scripts/tr_rule_versions.py check
    python backend/scripts/tr_rule_versions.py record --note "<what changed>"

`record` adds every current label that is not recorded yet, with the SHA-256 of its files and the git commit the
working tree is on. It refuses to change a recorded label: content that changed under an old label needs a new label.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from regchain.tr import rule_versions   # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['check', 'record'])
    parser.add_argument('--note', default='')
    args = parser.parse_args()
    status = rule_versions.status()
    for family, s in status.items():
        print(f"{family:13} {s['label']:24} {'recorded' if s['recorded'] else 'NOT RECORDED':13} {'ok' if s['matches'] else 'CHANGED' if s['recorded'] else '-'}")
    if args.action == 'check':
        return 0 if all(s['matches'] for s in status.values()) else 1
    changed = [f for f, s in status.items() if s['recorded'] and not s['matches']]
    if changed:
        print(f'refused: {changed} changed under a recorded label; give them a new label first')
        return 1
    record = rule_versions.load()
    commit = subprocess.run(['git', '-C', str(rule_versions.PACKAGE), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    for family, s in status.items():
        if not s['recorded'] and s['label'] not in record['labels']:
            record['labels'][s['label']] = {'family': family, 'files': s['files'], 'sha256': s['sha256'], 'recorded_on': commit,
                                            'recorded_at': time.strftime('%Y-%m-%d'), 'note': args.note}
            print('recorded', s['label'])
    rule_versions.RECORD.write_text(json.dumps(record, ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
