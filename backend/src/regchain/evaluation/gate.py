"""Regression gate: a run against a baseline run, judged by thresholds from a config file.

v0.18 adds applicability accuracy, DOES_NOT_APPLY F1 and R@3 drops, a WARN tier for the UNKNOWN
rate, LLM calls per obligation, parser snapshot compatibility (the regulation snapshot fingerprint
and the parser versions in the manifests) and the integrity of a frozen baseline. The v0.17 checks
keep their names; a thresholds file that predates a key gets the default below.
"""
import hashlib
import json
import os
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_THRESHOLDS = {
    'applicability_f1_max_drop': 0.05,          # points of F1 (0-1 scale) the APPLIES class may lose
    'coverage_macro_f1_max_drop': 0.03,
    'extraction_recall_max_drop': 0.05,
    'conflict_fpr_max_increase': 0.0,           # any rise in the conflict false positive rate fails
    'entity_gate_false_exclusion_max_increase': 0.0,
    'unknown_rate_max_increase': 0.10,          # FAIL above this rise ...
    'model_failure_rate_max': 0.05,             # absolute ceiling, baseline or not
    'runtime_warn_increase': 0.30,              # wall clock relative to the baseline
    'runtime_fail_increase': 0.50,
    'require_comparable_manifest': True,
    # v0.18
    'applicability_accuracy_max_drop': 0.05,
    'applicability_dna_f1_max_drop': 0.05,      # the DOES_NOT_APPLY class, mostly the gates' own output
    'recall_at_3_max_drop': 0.05,
    'unknown_rate_warn_increase': 0.02,         # ... and WARN on any rise above this one
    'llm_calls_warn_increase': 0.30,            # LLM calls per obligation, relative to the baseline
    'allow_parser_change': False}               # a parser version change fails unless a re-ingestion allows it
# The versions whose change makes a run read other regulation or policy text than its baseline did.
PARSER_KEYS = ('fca_parser', 'mevzuat_parser', 'policy_reader')
FROZEN = 'FROZEN.json'


def load_thresholds(path: Path | None) -> dict:
    values = dict(DEFAULT_THRESHOLDS)
    if path:
        values.update(json.loads(Path(path).read_text(encoding='utf-8')))
    return values


def pick(metrics, *path):
    value = metrics
    for key in path:
        value = (value or {}).get(key)
    return value


def evaluate_gate(run_metrics, baseline_metrics, thresholds, run_manifest=None, baseline_manifest=None, baseline_dir=None):
    """[{'check', 'status': PASS|WARN|FAIL|SKIP, 'value', 'baseline', 'threshold', 'detail'}], overall status."""
    checks = []
    limit = lambda key: thresholds.get(key, DEFAULT_THRESHOLDS[key])

    def drop(name, path, limit_key):
        value, base = pick(run_metrics, *path), pick(baseline_metrics, *path)
        if value is None or base is None:
            checks.append({'check': name, 'status': 'SKIP', 'value': value, 'baseline': base, 'threshold': limit(limit_key), 'detail': 'undefined in run or baseline'})
            return
        status = 'FAIL' if base - value > limit(limit_key) else 'PASS'
        checks.append({'check': name, 'status': status, 'value': value, 'baseline': base, 'threshold': limit(limit_key),
                       'detail': f'drop {round(base - value, 4)}'})

    def rise(name, path, limit_key, warn_key=None):
        value, base = pick(run_metrics, *path), pick(baseline_metrics, *path)
        if value is None or base is None:
            checks.append({'check': name, 'status': 'SKIP', 'value': value, 'baseline': base, 'threshold': limit(limit_key), 'detail': 'undefined in run or baseline'})
            return
        status = ('FAIL' if value - base > limit(limit_key) else
                  'WARN' if warn_key and value - base > limit(warn_key) else 'PASS')
        checks.append({'check': name, 'status': status, 'value': value, 'baseline': base,
                       'threshold': f'warn {limit(warn_key)}, fail {limit(limit_key)}' if warn_key else limit(limit_key),
                       'detail': f'rise {round(value - base, 4)}'})

    drop('applicability F1 (APPLIES)', ('applicability', 'f1'), 'applicability_f1_max_drop')
    drop('applicability accuracy', ('applicability', 'accuracy'), 'applicability_accuracy_max_drop')
    drop('applicability F1 (DOES_NOT_APPLY)', ('applicability', 'per_class', 'DOES_NOT_APPLY', 'f1'), 'applicability_dna_f1_max_drop')
    drop('coverage macro F1', ('coverage', 'macro', 'f1'), 'coverage_macro_f1_max_drop')
    drop('extraction recall', ('extraction', 'recall'), 'extraction_recall_max_drop')
    drop('retrieval R@3', ('retrieval', 'recall_at_3'), 'recall_at_3_max_drop')
    rise('conflict false positive rate', ('conflict', 'false_positive_rate'), 'conflict_fpr_max_increase')
    rise('entity gate false exclusion rate', ('entity_gate', 'false_exclusion_rate'), 'entity_gate_false_exclusion_max_increase')
    rise('UNKNOWN rate', ('reliability', 'unknown_rate'), 'unknown_rate_max_increase', 'unknown_rate_warn_increase')
    failure = pick(run_metrics, 'reliability', 'model_failure_rate')
    checks.append({'check': 'model failure rate', 'status': 'SKIP' if failure is None else 'FAIL' if failure > limit('model_failure_rate_max') else 'PASS',
                   'value': failure, 'baseline': pick(baseline_metrics, 'reliability', 'model_failure_rate'), 'threshold': limit('model_failure_rate_max'),
                   'detail': 'absolute ceiling'})
    wall, base_wall = pick(run_metrics, 'performance', 'wall_seconds'), pick(baseline_metrics, 'performance', 'wall_seconds')
    if wall is not None and base_wall:
        increase = (wall - base_wall) / base_wall
        status = 'FAIL' if increase > limit('runtime_fail_increase') else 'WARN' if increase > limit('runtime_warn_increase') else 'PASS'
        checks.append({'check': 'wall clock', 'status': status, 'value': wall, 'baseline': base_wall,
                       'threshold': f'warn {limit("runtime_warn_increase")}, fail {limit("runtime_fail_increase")}', 'detail': f'{round(100 * increase)}%'})
    checks.append(calls_check(run_metrics, baseline_metrics, limit('llm_calls_warn_increase')))
    if thresholds.get('require_comparable_manifest') and run_manifest and baseline_manifest:
        same = all(run_manifest.get(k) == baseline_manifest.get(k) for k in ('dataset', 'mode'))
        checks.append({'check': 'comparable configuration (dataset, mode)', 'status': 'PASS' if same else 'WARN', 'value': run_manifest.get('manifest_sha256', '')[:12],
                       'baseline': baseline_manifest.get('manifest_sha256', '')[:12], 'threshold': 'same dataset and mode',
                       'detail': 'models may differ (that is a benchmark); dataset or mode differing makes the gate meaningless'})
    if run_manifest and baseline_manifest:
        checks += parser_checks(run_manifest, baseline_manifest, bool(limit('allow_parser_change')))
    if baseline_dir is not None:
        checks.append(integrity_check(baseline_dir))
    overall = 'FAIL' if any(c['status'] == 'FAIL' for c in checks) else 'WARN' if any(c['status'] == 'WARN' for c in checks) else 'PASS'
    return checks, overall


def calls_per_obligation(metrics):
    """performance.calls_per_obligation, or llm_calls / obligations for a v0.17 metrics file."""
    value = pick(metrics, 'performance', 'calls_per_obligation')
    if value is not None:
        return value
    calls, obligations = pick(metrics, 'performance', 'llm_calls'), pick(metrics, 'performance', 'obligations')
    return round(calls / obligations, 4) if calls is not None and obligations else None


def calls_check(run_metrics, baseline_metrics, warn_increase):
    value, base = calls_per_obligation(run_metrics), calls_per_obligation(baseline_metrics)
    row = {'check': 'LLM calls per obligation', 'value': value, 'baseline': base, 'threshold': f'warn {warn_increase}'}
    if value is None or base is None:
        return {**row, 'status': 'SKIP', 'detail': 'undefined in run or baseline'}
    if not base:
        # Rules mode makes no call: against a model-free baseline any call is a new cost, not a ratio.
        return {**row, 'status': 'WARN' if value else 'PASS', 'detail': 'baseline makes no call'}
    increase = (value - base) / base
    return {**row, 'status': 'WARN' if increase > warn_increase else 'PASS', 'detail': f'{round(100 * increase)}%'}


def parser_checks(run_manifest, baseline_manifest, allow_change):
    """Parser snapshot compatibility: the regulation snapshot fingerprints and the parser versions.

    A fingerprint is compared for every fixture both runs read; a manifest without fingerprints
    (runs before v0.18) is SKIP. A parser version change is FAIL unless the thresholds allow it
    (a deliberate re-ingestion), when it is WARN.
    """
    checks = []
    mine, theirs = run_manifest.get('snapshots'), baseline_manifest.get('snapshots')
    if not isinstance(mine, dict) or not isinstance(theirs, dict):
        checks.append({'check': 'regulation snapshot fingerprint', 'status': 'SKIP', 'value': None if mine is None else len(mine),
                       'baseline': None if theirs is None else len(theirs), 'threshold': 'identical',
                       'detail': 'a manifest records no snapshot fingerprint'})
    else:
        shared = sorted(set(mine) & set(theirs))
        # A fixture the run could not load ('unreadable: <error>') is no evidence of compatibility, even
        # when the baseline failed on it the same way.
        changed = [fixture for fixture in shared if mine[fixture] != theirs[fixture] or str(mine[fixture]).startswith('unreadable')]
        checks.append({'check': 'regulation snapshot fingerprint', 'status': 'SKIP' if not shared else 'FAIL' if changed else 'PASS',
                       'value': ', '.join(str(mine[f])[:12] for f in shared) or None,
                       'baseline': ', '.join(str(theirs[f])[:12] for f in shared) or None, 'threshold': 'identical',
                       'detail': ('no fixture in common' if not shared else
                                  'changed: ' + ', '.join(changed) if changed else f'{len(shared)} fixture(s) identical')})
    run_versions, base_versions = run_manifest.get('versions') or {}, baseline_manifest.get('versions') or {}
    keys = [key for key in PARSER_KEYS if key in run_versions and key in base_versions]
    changed = [f'{key} {base_versions[key]} -> {run_versions[key]}' for key in keys if run_versions[key] != base_versions[key]]
    checks.append({'check': 'parser versions', 'status': 'SKIP' if not keys else ('WARN' if allow_change else 'FAIL') if changed else 'PASS',
                   'value': ', '.join(str(run_versions[k]) for k in keys) or None,
                   'baseline': ', '.join(str(base_versions[k]) for k in keys) or None,
                   'threshold': 'unchanged' + (' (change allowed)' if allow_change else ''),
                   'detail': 'not recorded' if not keys else '; '.join(changed) or 'unchanged'})
    return checks


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_frozen(directory: Path):
    """None for a directory without FROZEN.json; else the files that are missing or changed ([] = intact)."""
    marker = Path(directory) / FROZEN
    if not marker.is_file():
        return None
    try:
        record = json.loads(marker.read_text(encoding='utf-8'))
    except ValueError:
        return [f'{FROZEN} unreadable']              # an edited marker is a tampered baseline, not a crash of the gate
    problems = []
    for name, expected in sorted((record.get('files') or {}).items()):
        path = Path(directory) / name
        if not path.is_file():
            problems.append(f'{name} missing')
        elif file_sha256(path) != expected:
            problems.append(f'{name} changed')
    return problems


def integrity_check(baseline_dir):
    problems = verify_frozen(baseline_dir)
    row = {'check': 'baseline integrity', 'value': None, 'baseline': str(baseline_dir), 'threshold': 'every file in FROZEN.json unchanged'}
    if problems is None:
        return {**row, 'status': 'SKIP', 'detail': 'baseline is not frozen (no FROZEN.json)'}
    return {**row, 'status': 'FAIL' if problems else 'PASS', 'value': len(problems), 'detail': '; '.join(problems) or 'all files match'}


def freeze(run_dir: Path, out_dir: Path) -> dict:
    """Copy a scored run into an immutable baseline directory.

    manifest, metrics, report and results (and, when present, the re-scored results, the rescore
    note and the error taxonomy) are copied; baseline.json/.md hold the Phase A metric list; every
    file's sha256 goes to FROZEN.json and the files are made read-only. A directory that already
    has content is refused: a baseline is frozen once, and a new one gets a new name.
    """
    from .report import baseline_markdown, baseline_record
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f'{out_dir} exists and is not empty; a frozen baseline is never overwritten')
    for name in ('manifest.json', 'metrics.json', 'report.md', 'results.json'):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f'{run_dir / name} is missing; score the run (report) before freezing it')
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [name for name in ('manifest.json', 'metrics.json', 'report.md', 'results.json', 'results.rescored.json', 'rescore.json',
                               'errors.json', 'errors.md') if (run_dir / name).is_file()]
    for name in names:
        shutil.copyfile(run_dir / name, out_dir / name)
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    metrics = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))
    frozen_at = datetime.now(timezone.utc).isoformat()
    record = baseline_record(manifest, metrics, str(run_dir), frozen_at)
    (out_dir / 'baseline.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'baseline.md').write_text(baseline_markdown(record), encoding='utf-8')
    names += ['baseline.json', 'baseline.md']
    frozen = {'format': 'cardaman-frozen-baseline-v1', 'frozen_at': frozen_at, 'source_run': str(run_dir),
              'files': {name: file_sha256(out_dir / name) for name in names}}
    (out_dir / FROZEN).write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding='utf-8')
    # Read-only guards against an accidental edit, it is not a seal: the gate checks FROZEN.json's hashes,
    # and FROZEN.json's own hash (returned here, printed by the CLI) is what to write down elsewhere.
    for name in [*names, FROZEN]:
        os.chmod(out_dir / name, stat.S_IREAD)
    return {**frozen, 'frozen_sha256': file_sha256(out_dir / FROZEN)}


def gate_markdown(checks, overall, run_dir, baseline_dir):
    lines = [f'# Regression gate: {overall}', '', f'Run `{run_dir}` against baseline `{baseline_dir}`.', '',
             '| Check | Status | Run | Baseline | Threshold | Detail |', '|---|---|---|---|---|---|']
    for c in checks:
        lines.append(f'| {c["check"]} | {c["status"]} | {c["value"]} | {c["baseline"]} | {c["threshold"]} | {c["detail"]} |')
    return '\n'.join(lines) + '\n'
