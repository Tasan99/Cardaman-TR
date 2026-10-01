"""Markdown rendering of an evaluation run and of comparisons between runs."""
import json

from regchain.extraction import providers as _providers
from .metrics import COVERAGE, LABEL_SOURCES, MISSING_REASONS, WINDOW_ACTIONS

ASSESSED_COVERAGE = ('COVERS_TEXT', 'PARTIAL', 'CONFLICT', 'NO_EVIDENCE')
# The judge windows a run gets when JUDGE_NUM_CTX / JUDGE_NUM_CTX_LARGE are unset (providers.configured_judge): the v0.18
# code default, 16,384, at which qwen3:8b runs 19.5% on the CPU of the 8 GB target laptop. The product and the v0.19
# matrices set 8,192; a report has to say which of the two a run had, or an 8k leg and a 16k leg read the same.
CODE_NUM_CTX = getattr(_providers, 'NUM_CTX', 16384)
CODE_NUM_CTX_LARGE = getattr(_providers, 'JUDGE_NUM_CTX_LARGE', CODE_NUM_CTX)
RULE_DERIVED_BANNER = ('RULE_DERIVED rows measure agreement with the generator rule (evaluation/build_dataset.py computed those '
                       'labels with rule functions), not legal accuracy; only HUMAN_REVIEWED, MANUAL_LEGAL_READING and '
                       'SYNTHETIC_CONTROLLED rows are evidence of how often the analysis is right.')
NO_WINNER = ('No winner is chosen here: the table shows measurements on the same cases; which trade between quality and cost '
             'is acceptable is a decision this table does not make.')


def pct(value):
    return '—' if value is None else f'{round(100 * value, 1)}%'


def num(value):
    return '—' if value is None else str(value)


def table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join('---' for _ in headers) + '|']
    lines += ['| ' + ' | '.join(str(cell) for cell in row) + ' |' for row in rows]
    return '\n'.join(lines)


def confusion_table(block, labels):
    return table(['expected \\ predicted', *labels, 'OTHER'],
                 [[label, *(block['confusion'][label][other] for other in (*labels, 'OTHER'))] for label in labels])


def label_source_rows(metrics):
    """The per-label-source slices of a metrics dict as table rows (none for a metrics file written before v0.19)."""
    rows = []
    for source, block in (metrics.get('by_label_source') or {}).items():
        a, c, k, r = block['applicability'], block['coverage'], block['conflict'], block['retrieval']
        f1 = lambda label: pct((a['per_class'].get(label) or {}).get('f1'))
        rows.append([source, block['expectations'], f'{a["n"]} / {pct(a["accuracy"])}', f1('APPLIES'), f1('DOES_NOT_APPLY'),
                     f'{c["n"]} / {pct(c["macro"]["f1"])}', f'{k["n"]} / {pct(k["precision"])} / {pct(k["recall"])} / {pct(k["false_positive_rate"])}',
                     pct(r['recall_at_3']), pct(block['extraction']['recall'])])
    return rows


def label_source_banner(metrics):
    """The RULE_DERIVED warning when any scored row resolves to that source; [] otherwise."""
    rule = (metrics.get('by_label_source') or {}).get('RULE_DERIVED')
    if not rule:
        return []
    rows = ', '.join(f'{rule[family]["n"]} of {metrics[family]["n"]} {family.replace("_", " ")}'
                     for family in ('applicability', 'coverage', 'conflict') if rule[family]['n'])
    return [f'> **Label provenance:** scored rows resolve to RULE_DERIVED ({rows or "expectations only"}; no label source stated, '
            'or stated as such). ' + RULE_DERIVED_BANNER, '']


def stage_rows(perf):
    stages = perf.get('obligation_stages') or {}
    rows = [[name, s['mean_s'], s['median_s'], s['max_s'], s['obligations']] for name, s in (stages.get('stages') or {}).items()]
    if stages.get('total'):
        t = stages['total']
        rows.append(['**total per obligation**', t['mean_s'], t['median_s'], t['max_s'], t['obligations']])
    return rows


def class_f1(block):
    """A class's F1 as the tables show it: 0 when the class was never right but had support or predictions.

    metrics.prf leaves F1 undefined (None) when the class was never predicted (precision 0/0), even when it
    had support; the macro F1 of the same table counts that class as 0. A dash there hid a total class failure
    (smoke run: PARTIAL support 3, recall 0, shown as '—' beside a macro F1 of 0.0%). A dash stays for a class
    with neither support nor predictions, where there is nothing to measure.
    """
    block = block or {}
    if block.get('f1') is not None:
        return block['f1']
    return 0.0 if not block.get('tp') and (block.get('fn') or block.get('fp')) else None


def judge_window(manifest) -> dict:
    """The judge window settings a run's manifest recorded, with the code default named when a setting was unset.

    'base' / 'large' are the windows the run had (int); '*_text' says where they came from. A manifest written
    before v0.19's addendum has no JUDGE_NUM_CTX_LARGE key at all (not recorded) and every older one may have
    JUDGE_NUM_CTX empty (the smoke run of 24 September 2026: all 26 judge calls at 16,384).
    """
    env = manifest.get('environment') or {}
    number = lambda value: int(value) if str(value or '').strip().isdigit() else None

    def window(key, default):
        value = number(env.get(key))
        if value is not None:
            return value, str(value)
        return default, f'{default} ({key} unset: code default)' if key in env else f'{default} (not recorded: code default)'
    base, base_text = window('JUDGE_NUM_CTX', CODE_NUM_CTX)
    large, large_text = window('JUDGE_NUM_CTX_LARGE', CODE_NUM_CTX_LARGE)
    mode = str(env.get('JUDGE_CTX_MODE') or '').strip().lower()
    return {'mode': mode or 'fixed', 'mode_text': mode or 'fixed (default)', 'base': base, 'base_text': base_text, 'large': large,
            'large_text': large_text, 'fast_model': env.get('FAST_MODEL') or None}


def runtime_lines(perf):
    """The v0.19 runtime section of a run report (none for a metrics file without call records per model)."""
    rt = perf.get('runtime') or {}
    if not rt.get('by_model_ctx') or set(rt['by_model_ctx']) == {'unknown @ ?'}:
        return []
    gpu = rt.get('gpu') or {}
    rows = [[key, e['live_calls'], e['model_seconds'], pct((gpu.get(key) or {}).get('full_gpu_share')),
             ', '.join(f'{p} {n}' for p, n in sorted(((gpu.get(key) or {}).get('processors') or {}).items(), key=lambda kv: -kv[1])) or '—']
            for key, e in rt['by_model_ctx'].items()]
    speed = lambda field: ', '.join(f'tokens/s {model} {s["mean"]} / {s["median"]}' for model, s in ((rt.get(field) or {}).get('by_model') or {}).items()) or '—'
    return ['## Runtime by model and window', '',
            'Live (non-cache) calls by model and context window; "100% GPU share" is the share of those calls whose model was wholly '
            'on the GPU when GET /api/ps was asked (a dash: no observation, e.g. a run before v0.19).', '',
            table(['Model @ num_ctx', 'Live calls', 'Model seconds', '100% GPU share', 'Processor (live calls)'], rows), '',
            f'Output throughput (mean / median of live calls): {speed("tokens_per_second")}.',
            f'Prompt throughput: {speed("prompt_tokens_per_second").replace("tokens/s ", "")}.',
            f'Judge window: window actions: {", ".join(f"{a} {n}" for a, n in (rt.get("window_actions") or {}).items()) or "—"}; '
            f'fallbacks to the large window {num(rt.get("fallbacks"))}.',
            f'Model switches (live calls in time order): {num(rt.get("model_switches"))}; window changes of one model {num(rt.get("window_switches"))}; '
            f'model loads of at least 0.5 s: {num(rt.get("model_loads"))} ({num(rt.get("load_seconds"))} s)'
            + ('' if rt.get('time_ordered', True) else ' (some records carry no start time: counted in log order)') + '.', '']


def render_markdown(manifest, metrics, results):
    mode = manifest['mode']
    a, c, k, g, r, p, rel, perf, ex = (metrics[key] for key in ('applicability', 'coverage', 'conflict', 'entity_gate', 'retrieval',
                                                                  'proposals', 'reliability', 'performance', 'extraction'))
    env = manifest.get('environment') or {}
    window = judge_window(manifest)
    cov_f1 = lambda label: pct(class_f1(c['per_class'].get(label)))
    parts = [f'# Evaluation run — {manifest["dataset"]["id"]} v{manifest["dataset"]["version"]}',
             '',
             f'Created {manifest["created_at"]} · Cardaman {manifest["cardaman_version"]} · source tree `{manifest["source_tree_sha256"][:12]}` · '
             f'manifest `{manifest["manifest_sha256"][:12]}`',
             '',
             f'**Mode:** provider `{mode["provider"]}`, retrieval `{mode["retrieval"]}`, reranker `{mode.get("reranker_status", {}).get("status", "off")}`; '
             f'models: extraction `{manifest["models"]["extraction"] or "—"}`, judge `{manifest["models"]["judge"] or "—"}` '
             f'(thinking `{manifest["models"]["judge_thinking"] or "default"}`), embedding `{manifest["models"]["embedding"] or "—"}`, '
             f'reranker `{manifest["models"]["reranker"]}`.',
             '',
             f'**Pipeline settings:** coverage pipeline `{env.get("COVERAGE_PIPELINE") or "v18 (default)"}`, judge window '
             f'`{window["mode_text"]}` at num_ctx `{window["base_text"]}`, large `{window["large_text"]}`, fast model '
             f'`{window["fast_model"] or "the extraction model"}`, applicability clear match `{env.get("APPLICABILITY_CLEAR_MATCH") or "default"}`, '
             f'relevance screen `{env.get("RELEVANCE_SCREEN") or "default"}`.',
             '']
    parts += label_source_banner(metrics)
    if mode['provider'] == 'rules':
        parts += ['> **Rules mode.** No language model ran: extraction is the deterministic rules baseline, every applicability '
                  'and coverage judgement that needs a model is UNKNOWN, and only the entity gate, the retrieval order and the '
                  'extraction rules are measured. These numbers are a harness check, not the product\'s accuracy.', '']
    if metrics.get('reviewed_labels'):
        overlay = metrics['reviewed_labels']
        strict = overlay.get('strict')
        audit = overlay.get('audit') or {}
        parts += [f'> **Reviewed labels overlaid** from `{overlay.get("path")}` (sha256 `{str(overlay.get("sha256"))[:12]}`): '
                  f'{len(overlay.get("applied") or [])} ACCEPTED entries applied, {len(overlay.get("skipped") or [])} skipped'
                  + ('' if strict is None else
                     f' ({"strict: only entries the audit log approved" if strict else "not strict: every ACCEPTED entry, audited or not"}'
                     + (f'; audit log problems: {"; ".join(audit.get("problems") or [])}' if audit.get('problems') else '') + ')')
                  + '. The dataset file itself is unchanged.', '']
    parts += ['## Headline', '',
              table(['Metric', 'Value'], [
                  ['Cases / obligations scored', f'{perf["cases"]} / {a["n"]}'],
                  ['Applicability accuracy', pct(a['accuracy'])],
                  ['Applicability P / R / F1 (APPLIES)', f'{pct(a["precision"])} / {pct(a["recall"])} / {pct(a["f1"])}'],
                  ['Applicability macro F1', pct(a['macro']['f1'])],
                  ['Coverage macro P / R / F1', f'{pct(c["macro"]["precision"])} / {pct(c["macro"]["recall"])} / {pct(c["macro"]["f1"])}'],
                  ['Coverage F1 COVERS_TEXT / PARTIAL / CONFLICT / NO_EVIDENCE', ' / '.join(cov_f1(label) for label in ASSESSED_COVERAGE)],
                  ['Conflict P / R / FPR', f'{pct(k["precision"])} / {pct(k["recall"])} / {pct(k["false_positive_rate"])}'],
                  ['Entity gate false exclusion / inclusion', f'{pct(g["false_exclusion_rate"])} / {pct(g["false_inclusion_rate"])}'],
                  ['Retrieval R@3 / R@5 / R@10 / MRR', f'{pct(r["recall_at_3"])} / {pct(r["recall_at_5"])} / {pct(r["recall_at_10"])} / {num(r["mrr"])}'],
                  ['Evidence hit rate', pct(r['evidence_hit_rate'])],
                  ['Extraction recall / precision', f'{pct(ex["recall"])} / {pct(ex["precision"])}'],
                  # v0.19 (problem 1 F): every label row, and why the unscored ones are missing (a metrics file written before shows a dash)
                  ['Label rows: required / scored / missing (required missing)', label_rows_text(metrics)],
                  ['Missing rows: extraction missed / grounding rejected / matching failed', missing_text(metrics)],
                  ['Proposal success / draft schema validity', f'{pct(p["success_rate"])} / {pct(p["schema_validity_rate"])}'],
                  ['UNKNOWN rate / model failure / timeout / malformed', f'{pct(rel["unknown_rate"])} / {pct(rel["model_failure_rate"])} / '
                                                                          f'{pct(rel["timeout_rate"])} / {pct(rel["malformed_response_rate"])}'],
                  ['Wall clock / model seconds / LLM calls', f'{perf["wall_seconds"]} s / {perf["model_seconds"]} s / {perf["llm_calls"]}'],
                  ['Tokens (prompt / output) · cache hit ratio', f'{perf["prompt_tokens"]} / {perf["output_tokens"]} · {pct(perf["cache_hit_ratio"])}'],
                  ['Seconds per obligation', num(perf['seconds_per_obligation'])],
                  # v0.18 rows (a metrics file written before v0.18 shows a dash)
                  ['Applicability rows left UNKNOWN', pct(a.get('unknown_rate_applicability'))],
                  ['LLM calls / model seconds per obligation', f'{num(perf.get("calls_per_obligation"))} / {num(perf.get("model_seconds_per_obligation"))} s'],
                  ['Live (non-cache) calls / retries', f'{num(perf.get("live_calls"))} / {num(rel.get("retries"))}'],
                  ['Context overflows / open circuit', f'{num(rel.get("context_overflows"))} / {num(rel.get("circuit_open"))}'],
                  # v0.19 rows
                  ['Thinking calls (all / live) · per obligation', f'{num(perf.get("thinking_calls"))} / {num(perf.get("thinking_calls_live"))} · '
                                                                   f'{num(perf.get("thinking_calls_per_obligation"))}'],
                  ['Strong-model (thinking judge) / live calls per obligation', f'{num(perf.get("strong_model_calls_per_obligation"))} / '
                                                                                f'{num(perf.get("live_calls_per_obligation"))}'],
                  ['Model failures by code', ', '.join(f'{code} {n}' for code, n in sorted((rel.get('failures_by_code') or {}).items())) or '—']]),
              '',
              *(['## By label source', '',
                 table(['Source', 'Expectations', 'Applicability n / accuracy', 'APPLIES F1', 'DOES_NOT_APPLY F1', 'Coverage n / macro F1',
                        'Conflict n / P / R / FPR', 'R@3', 'Extraction recall'], label_source_rows(metrics)), '',
                 'Each asserted field counts under its own source (per-field source, else the expectation\'s, the case default, the dataset '
                 'default, RULE_DERIVED), so the slices of a family add up to its total.', '']
                if metrics.get('by_label_source') else []),
              '## Applicability', '', confusion_table(a, ('APPLIES', 'DOES_NOT_APPLY', 'UNKNOWN')), '',
              table(['Class', 'Precision', 'Recall', 'F1', 'Support'],
                    [[label, pct(v['precision']), pct(v['recall']), pct(v['f1']), v['support']] for label, v in a['per_class'].items()]),
              '',
              'Decided by (every obligation of the run): ' + (', '.join(f'{k} {v}' for k, v in sorted((a.get('decided_by') or {}).items(),
                                                                                                 key=lambda kv: -kv[1])) or '—')
              + ('; review flags: ' + ', '.join(f'{k} {v}' for k, v in sorted(a['review_flags'].items())) if a.get('review_flags') else ''),
              '',
              '## Policy coverage', '', confusion_table(c, ('COVERS_TEXT', 'PARTIAL', 'CONFLICT', 'NO_EVIDENCE', 'UNKNOWN')), '',
              table(['Class', 'Precision', 'Recall', 'F1', 'Support'],
                    [[label, pct(v['precision']), pct(v['recall']), pct(v['f1']), v['support']] for label, v in c['per_class'].items()]),
              f'\nNot-assessed expected / correct: {c["not_assessed_expected"]} / {c["not_assessed_correct"]}', '',
              '## Conflict', '', table(['TP', 'FP', 'FN', 'TN', 'Precision', 'Recall', 'FPR'],
                                       [[k['tp'], k['fp'], k['fn'], k['tn'], pct(k['precision']), pct(k['recall']), pct(k['false_positive_rate'])]]), '',
              '## Entity gate', '', table(['Rows', 'Expected excluded', 'False exclusions', 'False inclusions'],
                                          [[g['n'], g['expected_excluded'], g['false_exclusions'], g['false_inclusions']]]), '',
              '## Retrieval', '', table(['n', 'R@3', 'R@5', 'R@10', 'MRR', 'Evidence hit'],
                                        [[r['n'], pct(r['recall_at_3']), pct(r['recall_at_5']), pct(r['recall_at_10']), num(r['mrr']), pct(r['evidence_hit_rate'])]]), '',
              '## Proposals and reliability', '',
              table(['Needed', 'Produced', 'Drafts attempted', 'Schema valid', 'UNKNOWN', 'Failures', 'Timeouts', 'Malformed'],
                    [[p['needed'], p['produced'], p['drafts_attempted'], pct(p['schema_validity_rate']), rel['unknown'], rel['failures'], rel['timeouts'],
                      rel['malformed_responses']]]), '',
              '## Performance', '', table(['Task', 'Calls', 'Model seconds'], [[task, calls, num((perf.get('model_seconds_by_task') or {}).get(task))]
                                                                              for task, calls in sorted(perf['calls_by_task'].items(), key=lambda kv: -kv[1])]),
              '\nSlowest cases: ' + ', '.join(f'{s["case_id"]} ({s["wall_seconds"]} s)' for s in perf['slowest_cases']), '',
              'Stage seconds (summed over cases; parse = snapshot and policy reading): '
              + (', '.join(f'{stage} {seconds}' for stage, seconds in (perf.get('stage_seconds') or {}).items()) or '—'), '',
              *(['Seconds per assessed obligation, by stage (packet rows\' timings; a stage an obligation did not run counts 0 s; '
                 f'{(perf.get("obligation_stages") or {}).get("assessed_obligations")} assessed obligations):', '',
                 table(['Stage', 'Mean s', 'Median s', 'Max s', 'Obligations with the stage'], stage_rows(perf)), '']
                if stage_rows(perf) else []),
              *runtime_lines(perf),
              '## Cases', '']
    rows = []
    for result in results:
        for item in result['scored']:
            e = item['expected']
            where = f'md.{e["article"]}{e["clause"] or ""}'
            if not item['matched']:
                missing = (f'{item["missing_reason"]} ({item.get("missing_detail") or "—"})' if item.get('missing_reason') else 'not extracted')
                rows.append([result['case_id'], where, e['applicability'], f'**missing: {missing}**' if e['required'] else f'missing: {missing} (optional)',
                             '', '', ''])
                continue
            app = item['applicability']
            cov = item.get('coverage', ('—', '—'))
            con = item.get('conflict')
            mark = lambda pair: ('✓' if pair[0] == pair[1] else '✗') if pair and pair[0] != '—' else ''
            rows.append([result['case_id'], where, f'{app[0]} → {app[1]} {mark(app)}', f'{cov[0]} → {cov[1]} {mark(cov)}' if cov[0] != '—' else '—',
                         ('' if con is None else f'{con[0]} → {con[1]} {mark(con)}'),
                         (item.get('retrieval') or {}).get('rank', ''), round(result['wall_seconds'])])
        if result['error']:
            rows.append([result['case_id'], 'ERROR', result['error'], '', '', '', round(result['wall_seconds'])])
    parts += [table(['Case', 'Obligation', 'Applicability (exp → pred)', 'Coverage (exp → pred)', 'Conflict', 'Evidence rank', 's'], rows), '',
              *missing_section(metrics, results),
              '## Known false positives and false negatives', '']
    kinds = {}
    for item in metrics['mismatches']:
        kinds.setdefault(item['kind'], []).append(item)
    for kind, items in sorted(kinds.items()):
        parts.append(f'- **{kind}** ({len(items)}): ' + '; '.join(
            f'{i["where"]}' + (f' [{i.get("expected")} → {i.get("predicted")}]' if i.get('expected') is not None and i.get('predicted') is not None else '')
            for i in items[:20]) + (' …' if len(items) > 20 else ''))
    if not kinds:
        parts.append('None.')
    if metrics['errors']:
        parts += ['', '## Errors', ''] + [f'- {e["case_id"]}: {e["error"]}' for e in metrics['errors']]
    parts += ['', '---', 'Unit tests prove the code does what its author meant; these numbers measure how often the analysis agrees with the '
              'expert labels of this dataset, in this mode, with these models. Neither is the product\'s accuracy in the wild.']
    return '\n'.join(parts) + '\n'


def label_rows_text(metrics) -> str:
    match = metrics.get('matching') or {}
    if 'required_rows' not in match:
        return '—'
    return (f'{match["required_rows"]} / {match["scored_rows"]} / {match["missing_rows"]} ({match["missing_required"]}) '
            f'of {match["expected_rows"]} expected')


def missing_text(metrics) -> str:
    match = metrics.get('matching') or {}
    if 'missing_by_reason' not in match:
        return '—'
    by, required = match['missing_by_reason'], match.get('missing_required_by_reason') or {}
    text = ' / '.join(f'{by.get(reason, 0)} ({required.get(reason, 0)})' for reason in MISSING_REASONS)
    return text + (f' · not recorded {by["NOT_RECORDED"]}' if by.get('NOT_RECORDED') else '') + ' (required in brackets)'


def missing_section(metrics, results) -> list:
    """'## Missing rows': every label row no extracted duty was paired with, and why (identifiers.MISSING_REASONS)."""
    match = metrics.get('matching') or {}
    rows = [[result['case_id'], f'md.{item["expected"]["article"]}{item["expected"]["clause"] or ""}',
             ', '.join(item['expected'].get('action_keywords') or []) or '—', 'yes' if item['expected']['required'] else 'no',
             item.get('missing_reason') or 'NOT_RECORDED', item.get('missing_detail') or '—', item.get('match_basis') or '—']
            for result in results for item in result['scored'] if not item['matched']]
    if not rows:
        return []
    details = '; '.join(f'{reason}: ' + ', '.join(f'{code} {n}' for code, n in codes.items())
                        for reason, codes in (match.get('missing_details') or {}).items())
    return ['## Missing rows', '',
            'EXTRACTION_MISSED = no extraction attempt produced a candidate from the clause (classified as carrying no duty, no modal '
            'wording, or the model returned none); GROUNDING_REJECTED = the unit of the clause ended GROUNDING_REJECTED / '
            'SECOND_PASS_UNCERTAIN / INSUFFICIENT_EVIDENCE; MATCHING_FAILED = a candidate from the article/clause exists but was not '
            'paired (the detail says why). ' + (f'Details: {details}.' if details else ''), '',
            table(['Case', 'Obligation', 'Label words', 'Required', 'Missing reason', 'Detail', 'Match basis'], rows), '']


def reranker_of(manifest) -> str:
    """What reranked the run: the status the harness recorded (mode.reranker_status), with the model when it ran.

    The env's RERANK_MODEL (models.reranker) names a model even for a run whose reranker was off or failed to load.
    """
    status = ((manifest.get('mode') or {}).get('reranker_status') or {}).get('status') or ('on' if (manifest.get('mode') or {}).get('reranker') else 'off')
    model = (manifest.get('models') or {}).get('reranker')
    return status if status == 'off' or not model or model == 'off' else f'{status} ({model})'


def failed_cases(metrics) -> set:
    """The case ids that ended with an error: a result without predictions, which scores nothing."""
    return {e.get('case_id') for e in metrics.get('errors') or [] if isinstance(e, dict)}


def cases_of(manifest, metrics) -> str:
    """'8 run, 8 scored' (B3: the dataset size is not the run's size): the cases the run was asked for, and those scored.

    A case that ended with an error is in results.json (completed_cases counts it) but scored nothing, so it
    is not counted as scored: a leg with two failed cases of eight reads '8 run, 6 scored'.
    """
    asked = len((manifest.get('mode') or {}).get('cases') or []) or (manifest.get('dataset') or {}).get('cases')
    failed = failed_cases(metrics)
    subset = (metrics.get('subset') or {}).get('cases')
    scored = (len([c for c in subset if c not in failed]) if subset else
              None if metrics.get('completed_cases') is None else metrics['completed_cases'] - len(failed))
    return f'{asked} run, {scored if scored is not None else "?"} scored'


def scored_cases(manifest, metrics) -> tuple:
    """The case ids behind a metrics dict: the subset it was scored on, else the cases the run was asked for; without
    the cases that ended with an error, so two legs whose cases failed differently are seen to differ."""
    failed = failed_cases(metrics)
    return tuple(c for c in (metrics.get('subset') or {}).get('cases') or (manifest.get('mode') or {}).get('cases') or [] if c not in failed)


def pipeline_of(manifest) -> str:
    """'coverage `v19`, window `fixed` at num_ctx `8192`, large `16384`': the judge window as the run had it (an unset
    window is named as the code default, so a 16k leg never reads like an 8k one)."""
    env = manifest.get('environment') or {}
    window = judge_window(manifest)
    return (f'coverage `{env.get("COVERAGE_PIPELINE") or "v18"}`, window `{window["mode"]}` at num_ctx `{window["base_text"]}`, '
            f'large `{window["large_text"]}`')


def configuration_lines(runs):
    lines = []
    for name, manifest, metrics in runs:
        models, mode, dataset = manifest.get('models') or {}, manifest.get('mode') or {}, manifest.get('dataset') or {}
        lines.append(f'- **{name}**: provider `{mode.get("provider")}`, retrieval `{mode.get("retrieval")}`, '
                     f'reranker `{reranker_of(manifest)}`, extraction `{models.get("extraction") or "—"}`, judge `{models.get("judge") or "—"}` '
                     f'(thinking `{models.get("judge_thinking") or "default"}`), {pipeline_of(manifest)}, dataset `{dataset.get("id")} v{dataset.get("version")}` '
                     f'({cases_of(manifest, metrics)}), source `{str(manifest.get("source_tree_sha256"))[:12]}`')
    return lines


def compare_markdown(runs, note=None):
    """runs: [(name, manifest, metrics)] -> one table per metric family, runs as columns.

    ``note`` says how the runs were scored (e.g. on their common cases); the v0.19 rows add per-class
    coverage F1, per-obligation cost and one applicability / coverage row per label source present.
    """
    names = [name for name, _, _ in runs]
    lines = ['# Run comparison', '', *([note, ''] if note else []), 'Configurations:', '', *configuration_lines(runs)]
    sources = [s for s in (*LABEL_SOURCES, *sorted({s for _, _, x in runs for s in x.get('by_label_source') or {}} - set(LABEL_SOURCES)))
               if any(s in (x.get('by_label_source') or {}) for _, _, x in runs)]
    by_source = lambda x, source, *path: pick((x.get('by_label_source') or {}).get(source) or {}, path)
    picks = [('Applicability accuracy', lambda x: pct(x['applicability']['accuracy'])),
             ('Applicability F1 (APPLIES)', lambda x: pct(x['applicability']['f1'])),
             ('Applicability macro F1', lambda x: pct(x['applicability']['macro']['f1'])),
             ('Coverage macro F1', lambda x: pct(x['coverage']['macro']['f1'])),
             ('Conflict precision', lambda x: pct(x['conflict']['precision'])),
             ('Conflict recall', lambda x: pct(x['conflict']['recall'])),
             ('Conflict FPR', lambda x: pct(x['conflict']['false_positive_rate'])),
             ('Entity gate false exclusion', lambda x: pct(x['entity_gate']['false_exclusion_rate'])),
             ('Entity gate false inclusion', lambda x: pct(x['entity_gate']['false_inclusion_rate'])),
             ('Retrieval R@5', lambda x: pct(x['retrieval']['recall_at_5'])),
             ('Retrieval MRR', lambda x: num(x['retrieval']['mrr'])),
             ('Evidence hit rate', lambda x: pct(x['retrieval']['evidence_hit_rate'])),
             ('Extraction recall', lambda x: pct(x['extraction']['recall'])),
             ('UNKNOWN rate', lambda x: pct(x['reliability']['unknown_rate'])),
             ('Model failure rate', lambda x: pct(x['reliability']['model_failure_rate'])),
             ('LLM calls', lambda x: x['performance']['llm_calls']),
             ('Wall clock (s)', lambda x: x['performance']['wall_seconds']),
             ('Model seconds', lambda x: x['performance']['model_seconds']),
             ('Seconds per obligation', lambda x: num(x['performance']['seconds_per_obligation'])),
             ('Cache hit ratio', lambda x: pct(x['performance']['cache_hit_ratio'])),
             # v0.19 rows (a metrics file written before v0.19 shows a dash where the value cannot be derived)
             *[(f'Coverage F1 {label}', lambda x, label=label: pct(derived(x, f'coverage_{label.lower()}_f1'))) for label in ASSESSED_COVERAGE],
             ('Retrieval R@3', lambda x: pct(x['retrieval'].get('recall_at_3'))),
             ('LLM calls / obligation', lambda x: num(derived(x, 'calls_per_obligation'))),
             ('Thinking calls / obligation', lambda x: num(derived(x, 'thinking_calls_per_obligation'))),
             *[(f'Applicability accuracy [{s}]', lambda x, s=s: pct(by_source(x, s, 'applicability', 'accuracy'))) for s in sources],
             *[(f'Coverage macro F1 [{s}]', lambda x, s=s: pct(by_source(x, s, 'coverage', 'macro', 'f1'))) for s in sources]]
    lines += ['', table(['Metric', *names], [[label, *(fn(metrics) for _, _, metrics in runs)] for label, fn in picks]), '',
              'No winner is chosen here: the table shows measurements on the same dataset; the reader decides what matters.']
    return '\n'.join(lines) + '\n'


def cell_with_delta(value, reference, kind, is_reference):
    """A value, and for a leg other than the reference its change against the reference leg."""
    text = show(value, kind)
    return text if is_reference or value is None or reference is None else f'{text} ({delta(reference, value, kind)})'


# The Phase 19 trade-off rows: quality first, then cost (label, metric name, kind as in BEFORE_AFTER).
TRADEOFF_ROWS = [('Coverage macro F1', 'coverage_macro_f1', 'pct'), ('Conflict precision', 'conflict_precision', 'pct'),
                 ('Conflict recall', 'conflict_recall', 'pct'), ('Applicability accuracy', 'applicability_accuracy', 'pct'),
                 ('UNKNOWN rate', 'unknown_rate', 'pct'), ('Runtime / obligation (s)', 'seconds_per_obligation', 'rel'),
                 ('LLM calls / obligation', 'calls_per_obligation', 'rel'), ('Thinking calls / obligation', 'thinking_calls_per_obligation', 'rel'),
                 ('Total tokens', 'total_tokens', 'rel'), ('Model errors', 'model_failures', 'num')]


def tradeoff_markdown(runs, reference=0):
    """The model benchmark trade-off table: legs as columns, each leg's change against the reference leg in brackets.

    runs: [(name, manifest, metrics)], scored on the same cases (compare --common-cases / --cases). Thinking
    calls per obligation of a metrics file written before v0.19 come from its ai-calls.jsonl
    (harness.with_call_costs); without them the cell is a dash. No leg is declared better: a larger judge
    that is right more often and three times slower is a trade someone has to choose.
    """
    names = [name for name, _, _ in runs]
    ref_metrics = runs[reference][2]
    rows = [['Models (extraction / judge)', *(f'{(m.get("models") or {}).get("extraction") or "—"} / {(m.get("models") or {}).get("judge") or "—"}'
                                              for _, m, _ in runs)],
            ['Cases', *(cases_of(m, x) for _, m, x in runs)]]
    for label, name, kind in TRADEOFF_ROWS:
        ref = derived(ref_metrics, name)
        rows.append([label, *(cell_with_delta(derived(x, name), ref, kind, i == reference) for i, (_, _, x) in enumerate(runs))])
    scored = {scored_cases(m, x) for _, m, x in runs}
    warning = ([] if len(scored) == 1 else ['> **The legs were scored on different case sets**; compare them with `compare --common-cases` '
                                           'or `--cases` before reading the differences.', ''])
    return '\n'.join(['# Model benchmark trade-off', '', *warning, *configuration_lines(runs), '',
                      f'In brackets: the change against `{names[reference]}` (shares in percentage points, costs also relative).', '',
                      table(['Metric', *names], rows), '', NO_WINNER]) + '\n'


# The reranker A/B rows (label, metric name, kind).
RERANKER_ROWS = [('R@3', 'recall_at_3', 'pct'), ('R@5', 'recall_at_5', 'pct'), ('MRR', 'mrr', 'num'), ('Evidence hit rate', 'evidence_hit_rate', 'pct'),
                 ('Coverage macro F1', 'coverage_macro_f1', 'pct'), ('Conflict recall', 'conflict_recall', 'pct'),
                 ('Runtime / obligation (s)', 'seconds_per_obligation', 'rel'), ('LLM calls / obligation', 'calls_per_obligation', 'rel')]


def reranker_ab_markdown(a, b):
    """The reranker A/B table (A without, B with the cross-encoder) and the rule that decides it, with the measured deltas.

    a, b: (name, manifest, metrics), scored on the same cases. The rule is v0.18's: the reranker stays OFF
    unless it helps retrieval (R@3 or MRR up) without costing coverage macro F1 or conflict recall. The
    table states which of those conditions the measurement meets; switching the setting stays a decision.
    """
    (a_name, _, a_metrics), (b_name, _, b_metrics) = a, b
    rows, deltas = [], {}
    for label, name, kind in RERANKER_ROWS:
        before, after = derived(a_metrics, name), derived(b_metrics, name)
        deltas[name] = None if before is None or after is None else after - before
        rows.append([label, show(before, kind), show(after, kind), delta(before, after, kind)])
    helps = [deltas[n] for n in ('recall_at_3', 'mrr') if deltas[n] is not None]
    keeps = [deltas[n] for n in ('coverage_macro_f1', 'conflict_recall') if deltas[n] is not None]
    answer = lambda values, test: '—' if not values else 'yes' if test(values) else 'no'
    same_cases = scored_cases(a[1], a_metrics) == scored_cases(b[1], b_metrics)
    lines = ['# Reranker A/B', '', *configuration_lines([a, b]), '',
             *([] if same_cases else ['> **A and B were scored on different case sets**; use `compare --common-cases` so the deltas compare like '
                                      'for like.', '']),
             table(['METRIC', f'A `{a_name}`', f'B `{b_name}`', 'DELTA (B − A)'], rows), '',
             '**Rule:** the reranker stays OFF unless it helps: it is switched on only when B raises R@3 or MRR and lowers neither coverage '
             'macro F1 nor conflict recall, at a runtime cost that is accepted.',
             f'Measured deltas (B − A): R@3 {delta(derived(a_metrics, "recall_at_3"), derived(b_metrics, "recall_at_3"), "pct")}, '
             f'MRR {delta(derived(a_metrics, "mrr"), derived(b_metrics, "mrr"), "num")}, '
             f'coverage macro F1 {delta(derived(a_metrics, "coverage_macro_f1"), derived(b_metrics, "coverage_macro_f1"), "pct")}, '
             f'conflict recall {delta(derived(a_metrics, "conflict_recall"), derived(b_metrics, "conflict_recall"), "pct")}, '
             f'runtime / obligation {delta(derived(a_metrics, "seconds_per_obligation"), derived(b_metrics, "seconds_per_obligation"), "rel")}.',
             f'Retrieval improved (R@3 or MRR up): {answer(helps, lambda v: any(x > 0 for x in v))} · coverage F1 and conflict recall held: '
             f'{answer(keeps, lambda v: all(x >= 0 for x in v))}.', '', NO_WINNER]
    return '\n'.join(lines) + '\n'


# The judge windows of the user directive of 24 September 2026 (8k base, 16k only as the adaptive fallback).
CTX_BASE, CTX_LARGE = 8192, 16384
CTX_RULE = ('**Rule (user directive of 24 September 2026, items 8-10):** adaptive 8k/16k becomes the default only if its quality is not '
            'meaningfully worse than fixed 8k on the same cases, and the full 30-case run starts only after this comparison. The table gives '
            'the measured deltas; whether a difference is meaningful is decided by a person, not by this table.')
CTX_QUALITY = [('Applicability accuracy', 'applicability_accuracy', 'pct'), ('Coverage macro F1', 'coverage_macro_f1', 'pct'),
               *[(f'Coverage F1 {label}', f'coverage_{label.lower()}_f1', 'pct') for label in ASSESSED_COVERAGE],
               ('Conflict precision', 'conflict_precision', 'pct'), ('Conflict recall', 'conflict_recall', 'pct')]
CTX_COST = [('Runtime / obligation (s)', 'seconds_per_obligation', 'rel'), ('Model seconds / obligation', 'model_seconds_per_obligation', 'rel'),
            ('Live calls / obligation', 'live_calls_per_obligation', 'rel')]
CTX_LOAD = [('Window fallbacks to the large window', 'window_fallbacks', 'num'), ('Model switches', 'model_switches', 'num'),
            ('Model loads (at least 0.5 s)', 'model_loads', 'num')]


def window_warnings(fixed, adaptive) -> list:
    """What makes either leg something else than the directive's fixed 8k or adaptive 8k/16k window (from the manifests)."""
    (f_name, f_manifest, _), (a_name, a_manifest, _) = fixed, adaptive
    fw, aw = judge_window(f_manifest), judge_window(a_manifest)
    lines = []
    if not (fw['mode'] == 'fixed' and fw['base'] == CTX_BASE):
        lines.append(f'> **`{f_name}` is not a fixed {CTX_BASE} window** (JUDGE_CTX_MODE `{fw["mode_text"]}`, judge num_ctx `{fw["base_text"]}`): '
                     'its column does not measure fixed 8k.')
    if not (aw['mode'] == 'adaptive' and aw['base'] == CTX_BASE and aw['large'] == CTX_LARGE):
        lines.append(f'> **`{a_name}` is not an adaptive {CTX_BASE}/{CTX_LARGE} window** (JUDGE_CTX_MODE `{aw["mode_text"]}`, judge num_ctx '
                     f'`{aw["base_text"]}`, large `{aw["large_text"]}`): its column does not measure adaptive 8k/16k.')
    return [*lines, ''] if lines else []


def ctx_comparison_markdown(fixed, adaptive):
    """The fixed-8k against adaptive-8k/16k table the user directive asks for before the full run (items 8-10).

    fixed, adaptive: (name, manifest, metrics), scored on the same cases. Quality first (applicability accuracy,
    coverage macro and per-class F1, conflict precision and recall), then cost (runtime, model seconds and live calls
    per obligation), throughput by model (metrics.runtime), the judge window's actions, the share of live calls wholly
    on the GPU by model and window, the live calls per window and the model switches. It states the rule with the
    measured deltas and how much one row moves each quality number on these cases; it declares no winner.
    """
    (f_name, f_manifest, f_metrics), (a_name, a_manifest, a_metrics) = fixed, adaptive
    runtime = lambda x: (x.get('performance') or {}).get('runtime') or {}
    fr, ar = runtime(f_metrics), runtime(a_metrics)
    rows = []

    def add(label, before, after, kind):
        rows.append([label, show(before, kind), show(after, kind), delta(before, after, kind)])
    for label, name, kind in CTX_QUALITY + CTX_COST:
        add(label, derived(f_metrics, name), derived(a_metrics, name), kind)
    speed = lambda s: '—' if not s else f'{s["mean"]} / {s["median"]}'
    for field, label in (('tokens_per_second', 'Tokens/s'), ('prompt_tokens_per_second', 'Prompt tokens/s')):
        by = lambda r: (r.get(field) or {}).get('by_model') or {}
        for model in sorted(set(by(fr)) | set(by(ar))):
            before, after = by(fr).get(model), by(ar).get(model)
            rows.append([f'{label} {model} (mean / median)', speed(before), speed(after),
                         delta((before or {}).get('mean'), (after or {}).get('mean'), 'rel')])
    for label, name, kind in CTX_LOAD[:1]:
        add(label, derived(f_metrics, name), derived(a_metrics, name), kind)
    actions = lambda r: r.get('window_actions') or {}
    rows.append(['Window compressed / trimmed', *(f'{actions(r).get("compressed", 0)} / {actions(r).get("trimmed", 0)}' if r else '—' for r in (fr, ar)), '—'])
    gpu_order = lambda key: (key.rsplit(' @ ', 1)[0], int(key.rsplit(' @ ', 1)[-1]) if key.rsplit(' @ ', 1)[-1].isdigit() else 10**9)
    for key in sorted(set(fr.get('gpu') or {}) | set(ar.get('gpu') or {}), key=gpu_order):
        add(f'100% GPU share {key}', ((fr.get('gpu') or {}).get(key) or {}).get('full_gpu_share'),
            ((ar.get('gpu') or {}).get(key) or {}).get('full_gpu_share'), 'pct')
    for ctx in sorted(set(fr.get('by_num_ctx') or {}) | set(ar.get('by_num_ctx') or {}), key=lambda c: int(c) if c.isdigit() else 10**9):
        add(f'Live calls at num_ctx {ctx}', ((fr.get('by_num_ctx') or {}).get(ctx) or {}).get('live_calls'),
            ((ar.get('by_num_ctx') or {}).get(ctx) or {}).get('live_calls'), 'num')
    for label, name, kind in CTX_LOAD[1:]:
        add(label, derived(f_metrics, name), derived(a_metrics, name), kind)
    change = lambda name, kind='pct': delta(derived(f_metrics, name), derived(a_metrics, name), kind)
    judge = (a_manifest.get('models') or {}).get('judge') or (f_manifest.get('models') or {}).get('judge')
    judge_speed = [((r.get('tokens_per_second') or {}).get('by_model') or {}).get(judge) for r in (fr, ar)]
    n_app = pick(f_metrics, ('applicability', 'n')) or 0
    conflict = f_metrics.get('conflict') or {}
    positives = (conflict.get('tp') or 0) + (conflict.get('fn') or 0)
    resolution = (f'Resolution on these cases (the fixed leg\'s rows): '
                  + (f'one applicability row is {100 / n_app:.1f} pp of accuracy ({n_app} rows)' if n_app else 'no applicability row')
                  + '; ' + (f'one conflict-positive row is {100 / positives:.1f} pp of conflict recall ({positives} rows)' if positives
                            else 'no conflict-positive row, so conflict recall is undefined')
                  + f'; coverage is scored on {pick(f_metrics, ("coverage", "n")) or 0} rows.')
    same_cases = scored_cases(f_manifest, f_metrics) == scored_cases(a_manifest, a_metrics)
    lines = ['# Judge window: fixed 8k / adaptive 8k-16k', '', *configuration_lines([fixed, adaptive]), '',
             *([] if same_cases else ['> **The two legs were scored on different case sets**; use `compare --common-cases` so the deltas compare '
                                      'like for like.', '']),
             *window_warnings(fixed, adaptive),
             table(['METRIC', f'fixed 8k `{f_name}`', f'adaptive 8k/16k `{a_name}`', 'DELTA (adaptive − fixed)'], rows), '',
             CTX_RULE,
             f'Measured deltas (adaptive − fixed 8k): applicability accuracy {change("applicability_accuracy")}, coverage macro F1 '
             f'{change("coverage_macro_f1")}, conflict precision {change("conflict_precision")}, conflict recall {change("conflict_recall")}; '
             f'runtime / obligation {change("seconds_per_obligation", "rel")}; judge ({judge or "—"}) tokens/s '
             f'{delta((judge_speed[0] or {}).get("mean"), (judge_speed[1] or {}).get("mean"), "rel")}; fallbacks to the large window '
             f'{num(derived(a_metrics, "window_fallbacks"))} of {num(ar.get("live_calls"))} live calls.',
             resolution, '', NO_WINNER]
    return '\n'.join(lines) + '\n'


# The Phase A metric list: every number a frozen baseline records, flattened (None where undefined).
BASELINE_METRICS = [
    ('applicability_accuracy', ('applicability', 'accuracy')),
    ('applies_precision', ('applicability', 'per_class', 'APPLIES', 'precision')),
    ('applies_recall', ('applicability', 'per_class', 'APPLIES', 'recall')),
    ('applies_f1', ('applicability', 'per_class', 'APPLIES', 'f1')),
    ('does_not_apply_precision', ('applicability', 'per_class', 'DOES_NOT_APPLY', 'precision')),
    ('does_not_apply_recall', ('applicability', 'per_class', 'DOES_NOT_APPLY', 'recall')),
    ('does_not_apply_f1', ('applicability', 'per_class', 'DOES_NOT_APPLY', 'f1')),
    ('unknown_rate', ('reliability', 'unknown_rate')),
    ('unknown_rate_applicability', ('applicability', 'unknown_rate_applicability')),
    ('coverage_macro_f1', ('coverage', 'macro', 'f1')),
    *[(f'coverage_{label.lower()}_{key}', ('coverage', 'per_class', label, key)) for label in COVERAGE for key in ('precision', 'recall', 'f1', 'support')],
    ('conflict_precision', ('conflict', 'precision')),
    ('conflict_recall', ('conflict', 'recall')),
    ('conflict_fpr', ('conflict', 'false_positive_rate')),
    ('entity_gate_false_inclusion_rate', ('entity_gate', 'false_inclusion_rate')),
    ('entity_gate_false_exclusion_rate', ('entity_gate', 'false_exclusion_rate')),
    ('entity_gate_false_inclusions', ('entity_gate', 'false_inclusions')),
    ('entity_gate_false_exclusions', ('entity_gate', 'false_exclusions')),
    ('recall_at_3', ('retrieval', 'recall_at_3')),
    ('recall_at_5', ('retrieval', 'recall_at_5')),
    ('recall_at_10', ('retrieval', 'recall_at_10')),
    ('mrr', ('retrieval', 'mrr')),
    ('evidence_hit_rate', ('retrieval', 'evidence_hit_rate')),
    ('proposal_success_rate', ('proposals', 'success_rate')),
    ('runtime_seconds', ('performance', 'wall_seconds')),
    ('seconds_per_obligation', ('performance', 'seconds_per_obligation')),
    ('model_seconds', ('performance', 'model_seconds')),
    ('model_seconds_per_obligation', ('performance', 'model_seconds_per_obligation')),
    ('llm_calls', ('performance', 'llm_calls')),
    ('calls_per_obligation', ('performance', 'calls_per_obligation')),
    ('live_calls', ('performance', 'live_calls')),
    ('prompt_tokens', ('performance', 'prompt_tokens')),
    ('output_tokens', ('performance', 'output_tokens')),
    ('cache_hit_ratio', ('performance', 'cache_hit_ratio')),
    ('model_failures', ('reliability', 'failures')),
    ('model_failure_rate', ('reliability', 'model_failure_rate')),
    ('timeouts', ('reliability', 'timeouts')),
    ('malformed_responses', ('reliability', 'malformed_responses')),
    ('context_overflows', ('reliability', 'context_overflows')),
    ('retries', ('reliability', 'retries')),
    ('cases', ('performance', 'cases')),
    ('obligations', ('performance', 'obligations')),
    ('scored_applicability_rows', ('applicability', 'n'))]


def pick(metrics, path):
    value = metrics
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    return value


# v0.19 metric names that the comparison tables read but a frozen baseline does not record (BASELINE_METRICS stays the
# Phase A list, so a baseline frozen before v0.19 and one frozen after name the same metrics).
V019_METRICS = [
    ('thinking_calls', ('performance', 'thinking_calls')),
    ('thinking_calls_live', ('performance', 'thinking_calls_live')),
    ('thinking_calls_per_obligation', ('performance', 'thinking_calls_per_obligation')),
    ('strong_model_calls_per_obligation', ('performance', 'strong_model_calls_per_obligation')),
    ('live_calls_per_obligation', ('performance', 'live_calls_per_obligation')),
    ('total_tokens', ('performance', 'total_tokens')),
    # v0.19 context addendum (metrics.runtime)
    ('model_switches', ('performance', 'runtime', 'model_switches')),
    ('window_switches', ('performance', 'runtime', 'window_switches')),
    ('model_loads', ('performance', 'runtime', 'model_loads')),
    ('window_fallbacks', ('performance', 'runtime', 'fallbacks'))]
METRIC_PATHS = dict(BASELINE_METRICS + V019_METRICS)
PER_CLASS_F1 = {f'coverage_{label.lower()}_f1': label for label in COVERAGE}


def derived(metrics, name):
    """A metric by its Phase A (or v0.19) name; per-obligation cost is recomputed for a metrics file written before v0.18,
    and the v0.19 token total and live calls per obligation for one written before v0.19 (thinking calls cannot be:
    they need the call records, harness.with_call_costs). A coverage class's F1 is 0 when the class had support or
    predictions and was never right (class_f1), not undefined."""
    value = pick(metrics, METRIC_PATHS[name])
    if value is not None:
        return value
    if name in PER_CLASS_F1:
        return class_f1(pick(metrics, ('coverage', 'per_class', PER_CLASS_F1[name])))
    perf = metrics.get('performance') or {}
    obligations = perf.get('obligations')
    if name == 'calls_per_obligation' and perf.get('llm_calls') is not None and obligations:
        return round(perf['llm_calls'] / obligations, 4)
    if name == 'model_seconds_per_obligation' and perf.get('model_seconds') is not None and obligations:
        return round(perf['model_seconds'] / obligations, 4)
    if name == 'live_calls' and perf.get('llm_calls') is not None and perf.get('cache_hits') is not None:
        return perf['llm_calls'] - perf['cache_hits']
    if name == 'live_calls_per_obligation' and obligations and derived(metrics, 'live_calls') is not None:
        return round(derived(metrics, 'live_calls') / obligations, 4)
    if name == 'total_tokens' and perf.get('prompt_tokens') is not None and perf.get('output_tokens') is not None:
        return perf['prompt_tokens'] + perf['output_tokens']
    return None


def baseline_record(manifest, metrics, source_run, frozen_at):
    """baseline.json: what produced the run and the Phase A metric list, flat."""
    return {'format': 'cardaman-evaluation-baseline-v1', 'frozen_at': frozen_at, 'source_run': source_run,
            'run': {'cardaman_version': manifest.get('cardaman_version'), 'manifest_sha256': manifest.get('manifest_sha256'),
                    'source_tree_sha256': manifest.get('source_tree_sha256'), 'dataset': manifest.get('dataset'),
                    'mode': {k: v for k, v in (manifest.get('mode') or {}).items() if k in ('provider', 'retrieval', 'reranker', 'overrides')},
                    'models': manifest.get('models'), 'snapshots': manifest.get('snapshots'),
                    'completed_cases': metrics.get('completed_cases'), 'reviewed_labels': metrics.get('reviewed_labels')},
            'metrics': {name: derived(metrics, name) for name, _ in BASELINE_METRICS}}


def baseline_markdown(record):
    run = record['run']
    dataset = run.get('dataset') or {}
    mode = run.get('mode') or {}
    models = run.get('models') or {}
    lines = [f'# Frozen baseline — {dataset.get("id")} v{dataset.get("version")}', '',
             f'Frozen {record["frozen_at"]} from `{record["source_run"]}` · Cardaman {run.get("cardaman_version")} · '
             f'manifest `{str(run.get("manifest_sha256"))[:12]}` · {run.get("completed_cases")} cases', '',
             f'**Mode:** provider `{mode.get("provider")}`, retrieval `{mode.get("retrieval")}`; extraction `{models.get("extraction") or "—"}`, '
             f'judge `{models.get("judge") or "—"}` (thinking `{models.get("judge_thinking") or "default"}`), overrides `{mode.get("overrides") or {}}`.', '',
             table(['Metric', 'Value'], [[name, '—' if value is None else value] for name, value in record['metrics'].items()]), '',
             'Every file in this directory is listed with its sha256 in FROZEN.json and was made read-only; the regression gate '
             'fails when one of them changes.']
    return '\n'.join(lines) + '\n'


def delta(before, after, kind):
    if before is None or after is None:
        return '—'
    change = after - before
    if kind == 'pct':
        return f'{100 * change:+.1f} pp'
    text = f'{round(change, 4):+g}'
    return text + (f' ({100 * change / before:+.0f}%)' if kind == 'rel' and before else '')


def show(value, kind):
    return pct(value) if kind == 'pct' else num(value if not isinstance(value, float) else round(value, 4))


# (label, Phase A name, kind): pct = a 0-1 share (delta in points), rel = a cost (delta also relative), num = a count.
BEFORE_AFTER = [('Applicability accuracy', 'applicability_accuracy', 'pct'), ('APPLIES F1', 'applies_f1', 'pct'),
                ('DOES_NOT_APPLY F1', 'does_not_apply_f1', 'pct'), ('UNKNOWN rate', 'unknown_rate', 'pct'),
                ('Coverage macro F1', 'coverage_macro_f1', 'pct'), ('R@3', 'recall_at_3', 'pct'), ('Conflict FPR', 'conflict_fpr', 'pct'),
                ('Runtime / obligation (s)', 'seconds_per_obligation', 'rel'), ('LLM calls / obligation', 'calls_per_obligation', 'rel'),
                ('Model failures', 'model_failures', 'num'),
                ('APPLIES precision', 'applies_precision', 'pct'), ('APPLIES recall', 'applies_recall', 'pct'),
                ('DOES_NOT_APPLY precision', 'does_not_apply_precision', 'pct'), ('DOES_NOT_APPLY recall', 'does_not_apply_recall', 'pct'),
                ('Applicability rows left UNKNOWN', 'unknown_rate_applicability', 'pct'),
                ('R@5', 'recall_at_5', 'pct'), ('R@10', 'recall_at_10', 'pct'), ('MRR', 'mrr', 'num'), ('Evidence hit rate', 'evidence_hit_rate', 'pct'),
                ('Model seconds', 'model_seconds', 'rel'), ('Model seconds / obligation', 'model_seconds_per_obligation', 'rel'),
                ('Prompt tokens', 'prompt_tokens', 'rel'), ('Output tokens', 'output_tokens', 'rel'), ('Cache hit ratio', 'cache_hit_ratio', 'pct'),
                ('Obligations scored', 'scored_applicability_rows', 'num')]
MANDATORY_ROWS = [label for label, _, _ in BEFORE_AFTER[:10]]


def before_after_markdown(before_metrics, after_metrics, before_name='before', after_name='after', rows_spec=None, headers=('BEFORE', 'AFTER')):
    """The METRIC | BEFORE | AFTER | DELTA table (the first ten rows are the mandatory Phase U rows)."""
    rows = []
    for label, name, kind in rows_spec or BEFORE_AFTER:
        before, after = derived(before_metrics, name), derived(after_metrics, name)
        rows.append([label, show(before, kind), show(after, kind), delta(before, after, kind)])
    return (f'Before: `{before_name}` · after: `{after_name}`. Shares in %, their deltas in percentage points; a cost delta is '
            'also given relative to the before value.\n\n' + table(['METRIC', *headers, 'DELTA'], rows) + '\n')


# The v0.18 -> v0.19 diff (Phase 20): quality per class and per conflict, then the cost the escalation rule was meant to cut.
VERSION_DIFF = [('Applicability accuracy', 'applicability_accuracy', 'pct'), ('APPLIES F1', 'applies_f1', 'pct'),
                ('DOES_NOT_APPLY F1', 'does_not_apply_f1', 'pct'), ('UNKNOWN rate', 'unknown_rate', 'pct'),
                ('Coverage macro F1', 'coverage_macro_f1', 'pct'),
                *[(f'Coverage F1 {label}', f'coverage_{label.lower()}_f1', 'pct') for label in ASSESSED_COVERAGE],
                ('Conflict precision', 'conflict_precision', 'pct'), ('Conflict recall', 'conflict_recall', 'pct'),
                ('Conflict FPR', 'conflict_fpr', 'pct'), ('R@3', 'recall_at_3', 'pct'),
                ('Runtime / obligation (s)', 'seconds_per_obligation', 'rel'), ('LLM calls / obligation', 'calls_per_obligation', 'rel'),
                ('Thinking calls / obligation', 'thinking_calls_per_obligation', 'rel'), ('Model failures', 'model_failures', 'num'),
                ('Strong-model calls / obligation', 'strong_model_calls_per_obligation', 'rel'),
                ('Live calls / obligation', 'live_calls_per_obligation', 'rel'), ('Model seconds / obligation', 'model_seconds_per_obligation', 'rel'),
                ('Total tokens', 'total_tokens', 'rel'), ('Obligations scored', 'scored_applicability_rows', 'num')]


def version_diff_markdown(before_metrics, after_metrics, before_name, after_name, before_label='v0.18', after_label='v0.19'):
    """The METRIC | v0.18 | v0.19 | DELTA table. Pass metrics through harness.with_call_costs first when a side was
    written before v0.19, so its thinking calls per obligation are counted from its call records."""
    cases = [pick(x, ('performance', 'cases')) for x in (before_metrics, after_metrics)]
    warning = (f'> **The two sides were scored on different numbers of cases ({cases[0]} vs {cases[1]})**: the deltas mix the case '
               'selection with the version. Score both on the same cases (compare --cases / --common-cases).\n\n'
               if None not in cases and cases[0] != cases[1] else '')
    return warning + before_after_markdown(before_metrics, after_metrics, before_name, after_name, VERSION_DIFF, (before_label, after_label))
