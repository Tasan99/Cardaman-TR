"""python -m regchain.evaluation {validate|run|report|compare|gate|benchmark|errors|coverage-errors|label-audit|labels|expert-export|freeze|diff}"""
import argparse
import json
import sys
from pathlib import Path

from .gate import evaluate_gate, freeze, gate_markdown, load_thresholds
from .harness import load_dataset, run
from .report import before_after_markdown, compare_markdown, render_markdown


def load_run(directory: Path):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    metrics = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
    return manifest, metrics


def compared_runs(args):
    """[(name, manifest, metrics)] for compare: each run's metrics.json, or (v0.19) each run scored in memory on
    --cases or on the cases every run completed (--common-cases); v0.19 call costs are filled in from ai-calls.jsonl."""
    from .harness import completed_cases, score_subset, with_call_costs
    directories = [Path(d) for d in args.runs]
    cases = [c for c in (args.cases or '').split(',') if c]
    if args.common_cases:
        done = [set(completed_cases(d)) for d in directories]
        cases = sorted(set.intersection(*done))
        if not cases:
            raise SystemExit('compare --common-cases: the runs have no completed case in common')
    runs = []
    for directory in directories:
        manifest, metrics = score_subset(directory, cases, args.dataset) if cases else load_run(directory)
        runs.append((directory.name, manifest, with_call_costs(directory, metrics)))
    note = (f'Scored in memory on {len(cases)} case(s) {"every run completed" if args.common_cases else "chosen with --cases"}: '
            f'{", ".join(cases)}. No run folder was written.') if cases else None
    return runs, note


def main(argv=None):
    parser = argparse.ArgumentParser(description='Cardaman evaluation harness')
    commands = parser.add_subparsers(dest='command', required=True)
    validate = commands.add_parser('validate', help='Check a dataset file against the schema')
    validate.add_argument('--dataset', required=True)
    runner = commands.add_parser('run', help='Run a dataset and write manifest, results, metrics and report')
    runner.add_argument('--dataset', required=True)
    runner.add_argument('--out', default='evaluation/runs')
    runner.add_argument('--provider', choices=['rules', 'ollama'], default='rules')
    runner.add_argument('--retrieval', choices=['lexical', 'hybrid'], default='lexical')
    runner.add_argument('--reranker', action='store_true', help='Use the configured RERANK_MODEL (semantic retrieval only)')
    runner.add_argument('--cases', default='', help='Comma-separated case ids; default all')
    runner.add_argument('--label', default='')
    for key in ('extraction-model', 'judge-model', 'embed-model', 'rerank-model'):
        runner.add_argument('--' + key, default=None)
    runner.add_argument('--judge-thinking', choices=['on', 'off'], default=None)
    runner.add_argument('--applicability-clear-match', choices=['model', 'quick', 'rule'], default=None,
                        help='APPLICABILITY_CLEAR_MATCH for this run: who decides when every gate is a clear match')
    runner.add_argument('--relevance-screen', choices=['on', 'off'], default=None, help='RELEVANCE_SCREEN for this run')
    runner.add_argument('--coverage-pipeline', choices=['v18', 'v19'], default=None, help='COVERAGE_PIPELINE for this run')
    runner.add_argument('--ctx-mode', choices=['fixed', 'adaptive'], default=None, help='JUDGE_CTX_MODE for this run (judge context window)')
    runner.add_argument('--judge-num-ctx', type=int, default=None,
                        help='JUDGE_NUM_CTX for this run: the judge\'s base window (the product uses 8192; unset, the code default 16384 applies)')
    runner.add_argument('--judge-num-ctx-large', type=int, default=None,
                        help='JUDGE_NUM_CTX_LARGE for this run: the window adaptive mode falls back to (default 16384)')
    rerender = commands.add_parser('report', help='Re-render report.md of a run directory')
    rerender.add_argument('--rescore', action='store_true', help='Re-score the saved packets against the current dataset labels')
    rerender.add_argument('--dataset', default=None, help='Dataset file (defaults to the one the run manifest names)')
    rerender.add_argument('--reviewed-labels', default=None, help='With --rescore: lay the ACCEPTED entries of this reviewed-labels file '
                                                                  'over the labels (in memory; the dataset is not written)')
    rerender.add_argument('--run', required=True)
    compare = commands.add_parser('compare', help='Side-by-side table of several run directories')
    compare.add_argument('--runs', nargs='+', required=True)
    compare.add_argument('--out', default=None)
    compare.add_argument('--dataset', default=None, help='With --cases/--common-cases: the dataset to score against (default: the manifest\'s)')
    subset = compare.add_mutually_exclusive_group()
    subset.add_argument('--cases', default='', help='Score every run in memory on these comma-separated case ids (no run folder is written)')
    subset.add_argument('--common-cases', action='store_true', help='Score every run in memory on the cases all of them completed')
    compare.add_argument('--view', choices=['table', 'tradeoff', 'reranker', 'versions', 'ctx'], default='table',
                         help='table: every metric; tradeoff: the model benchmark trade-off; reranker: A/B of two runs (A without, B with); '
                              'versions: METRIC | v0.18 | v0.19 | DELTA of two runs (before, after); ctx: the judge window table of two '
                              'runs (fixed 8k first, then adaptive 8k/16k)')
    gate = commands.add_parser('gate', help='Regression gate of a run against a baseline run')
    gate.add_argument('--run', required=True)
    gate.add_argument('--baseline', required=True)
    gate.add_argument('--thresholds', default=None)
    gate.add_argument('--out', default=None)
    bench = commands.add_parser('benchmark', help='Run a matrix of configurations (JSON list of override objects) and compare them')
    bench.add_argument('--dataset', required=True)
    bench.add_argument('--matrix', required=True)
    bench.add_argument('--out', default='evaluation/runs')
    bench.add_argument('--report', default=None)
    bench.add_argument('--cases', default='')
    errors = commands.add_parser('errors', help='Classify every wrong decision of a run (error taxonomy): errors.json and errors.md')
    errors.add_argument('--run', required=True)
    errors.add_argument('--dataset', default=None, help='Re-pair the packets with this dataset in memory (default: the scored results)')
    errors.add_argument('--out', default=None, help='Directory for errors.json/.md (default: the run directory)')
    cov_errors = commands.add_parser('coverage-errors', help='Every coverage/conflict error of a run with its evidence chain and category: '
                                                             'coverage-errors.json and coverage-errors.md')
    cov_errors.add_argument('--run', required=True)
    cov_errors.add_argument('--dataset', default=None, help='Dataset file (default: the one the manifest records, else evaluation/datasets/<id>.json)')
    cov_errors.add_argument('--annotations', default=None, help='Manual annotations JSON keyed by "<case_id> md.<article>(<clause>)" '
                                                                '(default: evaluation/reports/coverage-error-annotations-<run name>.json if present)')
    cov_errors.add_argument('--label-review', default=None, help='Label-audit file whose DISPUTED reasons flag LABEL_AMBIGUITY '
                                                                 '(default: the newest evaluation/reports/label-review-*.json)')
    cov_errors.add_argument('--out-json', default=None, help='Default: <run>/coverage-errors.json')
    cov_errors.add_argument('--out-md', default=None, help='Default: <run>/coverage-errors.md')
    audit = commands.add_parser('label-audit', help='Review list of suspicious golden labels (never edits the dataset)')
    audit.add_argument('--runs', nargs='+', required=True)
    audit.add_argument('--dataset', default=None)
    audit.add_argument('--min-agreeing-runs', type=int, default=2)
    audit.add_argument('--out', default=None, help='File stem for the .json and .md (default: <first run>/label-audit)')
    freezer = commands.add_parser('freeze', help='Freeze a scored run as an immutable baseline directory')
    freezer.add_argument('--run', required=True)
    freezer.add_argument('--out', required=True)
    diff = commands.add_parser('diff', help='METRIC | BEFORE | AFTER | DELTA table of two runs or baselines')
    diff.add_argument('--before', required=True)
    diff.add_argument('--after', required=True)
    diff.add_argument('--out', default=None)
    diff.add_argument('--versions', nargs=2, default=None, metavar=('BEFORE_LABEL', 'AFTER_LABEL'),
                      help='The version table (e.g. --versions v0.18 v0.19): METRIC | v0.18 | v0.19 | DELTA with per-class coverage F1, '
                           'conflict P/R/FPR and thinking calls per obligation')
    # v0.19 human review loop: pending label changes (labels ...) and the expert review package (expert-export).
    from . import expert_export, pending
    pending.add_commands(commands)
    expert_export.add_command(commands)
    args = parser.parse_args(argv)
    if args.command == 'labels':
        return pending.run_command(args, parser)
    if args.command == 'expert-export':
        return expert_export.run_command(args)
    if args.command == 'validate':
        dataset = load_dataset(Path(args.dataset))
        print(f'{dataset.dataset_id} v{dataset.version}: {len(dataset.cases)} cases, '
              f'{sum(len(c.expected_obligations) for c in dataset.cases)} expected obligations; schema OK')
        return 0
    if args.command == 'run':
        overrides = {k: getattr(args, k) for k in ('extraction_model', 'judge_model', 'embed_model', 'judge_thinking',
                                                   'applicability_clear_match', 'relevance_screen', 'coverage_pipeline', 'ctx_mode',
                                                   'judge_num_ctx', 'judge_num_ctx_large') if getattr(args, k)}
        if args.rerank_model is not None:
            overrides['rerank_model'] = args.rerank_model
        run_dir, manifest, metrics = run(Path(args.dataset), Path(args.out), provider=args.provider, retrieval=args.retrieval, reranker=args.reranker,
                                         cases=[c for c in args.cases.split(',') if c], label=args.label, overrides=overrides)
        print(json.dumps({'run': str(run_dir), 'manifest': manifest['manifest_sha256'][:12],
                          'applicability_accuracy': metrics['applicability']['accuracy'], 'applicability_f1': metrics['applicability']['f1'],
                          'coverage_macro_f1': metrics['coverage']['macro']['f1'], 'conflict_fpr': metrics['conflict']['false_positive_rate'],
                          'llm_calls': metrics['performance']['llm_calls'], 'wall_seconds': metrics['performance']['wall_seconds'],
                          'errors': len(metrics['errors'])}, ensure_ascii=False))
        return 0
    if args.command == 'report':
        # Scores the run from its results (also a run that was stopped early) and re-renders the report;
        # --rescore re-pairs the saved packets with the dataset now on disk (labels corrected after the run).
        from .harness import rescore, score_partial
        if args.reviewed_labels and not args.rescore:
            parser.error('--reviewed-labels needs --rescore')
        directory, _, metrics = (rescore(Path(args.run), args.dataset, reviewed_labels=args.reviewed_labels) if args.rescore
                                 else score_partial(Path(args.run), dataset_path=args.dataset))
        print(json.dumps({'run': str(directory), 'completed_cases': metrics['completed_cases'],
                          'applicability_accuracy': metrics['applicability']['accuracy']}, ensure_ascii=False))
        return 0
    if args.command == 'compare':
        from .report import ctx_comparison_markdown, reranker_ab_markdown, tradeoff_markdown, version_diff_markdown
        if args.view in ('reranker', 'versions', 'ctx') and len(args.runs) != 2:
            parser.error(f'--view {args.view} compares exactly two runs (A/before/fixed 8k, then B/after/adaptive)')
        runs, note = compared_runs(args)
        text = (tradeoff_markdown(runs) if args.view == 'tradeoff' else reranker_ab_markdown(*runs) if args.view == 'reranker'
                else '# v0.18 / v0.19\n\n' + version_diff_markdown(runs[0][2], runs[1][2], runs[0][0], runs[1][0]) if args.view == 'versions'
                else ctx_comparison_markdown(*runs) if args.view == 'ctx'
                else compare_markdown(runs, note))
        if note and args.view != 'table':
            text = text.replace('\n', f'\n\n{note}\n', 1)
        if args.out:
            Path(args.out).write_text(text, encoding='utf-8')
        print(text)
        return 0
    if args.command == 'gate':
        run_manifest, run_metrics = load_run(Path(args.run))
        base_manifest, base_metrics = load_run(Path(args.baseline))
        checks, overall = evaluate_gate(run_metrics, base_metrics, load_thresholds(Path(args.thresholds) if args.thresholds else None), run_manifest, base_manifest,
                                        baseline_dir=Path(args.baseline))
        text = gate_markdown(checks, overall, args.run, args.baseline)
        if args.out:
            Path(args.out).write_text(text, encoding='utf-8')
        print(text)
        return 0 if overall != 'FAIL' else 1
    if args.command == 'benchmark':
        matrix = json.loads(Path(args.matrix).read_text(encoding='utf-8'))
        runs = []
        for leg in matrix:
            name = leg.pop('name')
            provider = leg.pop('provider', 'ollama')
            retrieval = leg.pop('retrieval', 'hybrid')
            reranker = bool(leg.pop('reranker', False)) or bool(leg.get('rerank_model'))
            # v0.19: a leg may name its cases (the matrix then fixes the subset); --cases still wins.
            leg_cases = [c for c in args.cases.split(',') if c] or list(leg.pop('cases', None) or [])
            leg.pop('cases', None)
            print(f'== benchmark leg {name}: {leg}', flush=True)
            run_dir, manifest, metrics = run(Path(args.dataset), Path(args.out), provider=provider, retrieval=retrieval, reranker=reranker,
                                             cases=leg_cases, label=name, overrides=leg)
            runs.append((name, manifest, metrics))
        from .report import ctx_comparison_markdown, judge_window, tradeoff_markdown
        text = compare_markdown(runs) + '\n' + tradeoff_markdown(runs)
        modes = [judge_window(manifest)['mode'] for _, manifest, _ in runs]
        if modes == ['fixed', 'adaptive']:
            # evaluation/ctx-compare-v019.json: fixed 8k, then adaptive 8k/16k (user directive items 8-10).
            text += '\n' + ctx_comparison_markdown(*runs)
        if args.report:
            Path(args.report).write_text(text, encoding='utf-8')
        print(text)
        return 0
    if args.command == 'errors':
        from .taxonomy import write_errors
        report, json_path, md_path = write_errors(Path(args.run), args.dataset, args.out)
        print(json.dumps({'errors': str(json_path), 'markdown': str(md_path), 'applicability': report['counts']['applicability'],
                          'coverage': report['counts']['coverage'], 'failed_cases': len(report['case_errors'])}, ensure_ascii=False))
        return 0
    if args.command == 'coverage-errors':
        from .coverage_errors import write_report
        report, json_path, md_path = write_report(Path(args.run), args.dataset, args.annotations, args.out_json, args.out_md, args.label_review)
        print(json.dumps({'coverage_errors': str(json_path), 'markdown': str(md_path), 'errors': report['counts']['errors'],
                          'auto': report['counts']['auto'], 'final': report['counts']['final'], 'annotations': report['annotations'],
                          'dataset_same_as_run': report['dataset']['same_as_run']}, ensure_ascii=False))
        return 0
    if args.command == 'label-audit':
        from .labels import write_audit
        record, json_path, md_path = write_audit([Path(d) for d in args.runs], args.out, args.dataset, args.min_agreeing_runs)
        print(json.dumps({'audit': str(json_path), 'markdown': str(md_path), 'items': len(record['items'])}, ensure_ascii=False))
        return 0
    if args.command == 'freeze':
        try:
            frozen = freeze(Path(args.run), Path(args.out))
        except (FileExistsError, FileNotFoundError) as exc:
            print(f'freeze refused: {exc}', file=sys.stderr)
            return 1
        print(json.dumps({'baseline': args.out, 'files': len(frozen['files']), 'frozen_sha256': frozen['frozen_sha256']}, ensure_ascii=False))
        return 0
    if args.command == 'diff':
        from .harness import with_call_costs
        from .report import version_diff_markdown
        _, before = load_run(Path(args.before))
        _, after = load_run(Path(args.after))
        if args.versions:
            # Thinking calls per obligation of a run scored before v0.19 are counted from its ai-calls.jsonl.
            before, after = with_call_costs(Path(args.before), before), with_call_costs(Path(args.after), after)
            text = f'# {args.versions[0]} / {args.versions[1]}\n\n' + version_diff_markdown(before, after, args.before, args.after, *args.versions)
        else:
            text = '# Before / after\n\n' + before_after_markdown(before, after, args.before, args.after)
        if args.out:
            Path(args.out).write_text(text, encoding='utf-8')
        print(text)
        return 0
    return 2


if __name__ == '__main__':
    sys.exit(main())
