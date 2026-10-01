"""Scoring. Every function takes plain rows and returns plain numbers; nothing here reads files."""
from collections import Counter
from statistics import mean, median

APPLICABILITY = ('APPLIES', 'DOES_NOT_APPLY', 'UNKNOWN')
COVERAGE = ('COVERS_TEXT', 'PARTIAL', 'CONFLICT', 'NO_EVIDENCE', 'UNKNOWN')
# v0.19 label provenance (schema.LabelSource), strongest first. A label that states none is RULE_DERIVED:
# tr-aml-v1's labels were computed by build_dataset.py's rule functions, so agreement with them measures
# agreement with that rule, never legal accuracy; the conservative reading is the default.
LABEL_SOURCES = ('HUMAN_REVIEWED', 'MANUAL_LEGAL_READING', 'SYNTHETIC_CONTROLLED', 'RULE_DERIVED')
LABEL_FIELDS = ('applicability', 'coverage', 'conflict', 'entity_gate')
LEGACY_LABEL_SOURCE = 'RULE_DERIVED'
# Per-obligation stages in pipeline order (v0.19 engine ROW_TIMINGS); other recorded stages follow them.
STAGE_ORDER = ('retrieval', 'relevance_screen', 'conflict_precheck', 'fast_classifier', 'thinking_verifier', 'support_judgement',
               'aggregation', 'applicability', 'enrichment', 'coverage', 'conflict', 'proposal_generation')
# v0.19 context addendum (WS6c). What providers.fit_call recorded as window_action, in the order it tries them.
# In ladder order (providers.WINDOW_ACTIONS); a compact retry runs on the same window, so it is no fallback (t6 review 1, F5).
WINDOW_ACTIONS = ('fits', 'compressed', 'fallback_large', 'trimmed', 'compact_after_truncation', 'fallback_large_after_truncation')
FALLBACK_ACTIONS = ('fallback_large', 'fallback_large_after_truncation')
# A call refused before it reached the model (admission, open circuit) loads nothing and generates nothing.
NOT_SENT = ('CONTEXT_BUDGET_EXCEEDED', 'CIRCUIT_OPEN')
# size_vram / size at or above this is "100% GPU" as `ollama ps` prints it (it rounds to whole percent).
FULL_GPU = 0.995
# A load_duration at or above this is a real (re)load of the model (providers.RELOAD_MS: warm calls measured 3-16 ms
# in the smoke run of 24 September 2026, real loads 1,835-40,698 ms).
RELOAD_MS = 500


def ratio(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else None


def label_source(expected: dict, field=None, case_default=None, dataset_default=None) -> str:
    """Where one asserted field of an expectation came from (expected is its dump, as score_case stores it).

    Order: the per-field source, the expectation's own source, the case default, the dataset default,
    RULE_DERIVED. A results.json written before v0.19 carries neither key and falls through to the
    defaults, so an old run is re-scored under the provenance its dataset states today.
    """
    by_field = expected.get('label_sources') or {}
    if field and by_field.get(field):
        return by_field[field]
    return expected.get('label_source') or case_default or dataset_default or LEGACY_LABEL_SOURCE


def thinking(call) -> bool:
    """A call that ran with reasoning on (ai-calls.jsonl records a bool; a string is tolerated)."""
    return call.get('thinking') is True or str(call.get('thinking')).lower() == 'true'


def prf(tp, fp, fn):
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = (round(2 * precision * recall / (precision + recall), 4) if precision is not None and recall is not None and (precision + recall)
          else (0.0 if precision is not None and recall is not None else None))
    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}


def confusion(rows, labels):
    """rows: [(expected, predicted)] -> {expected: {predicted: count}} over the given labels (others under 'OTHER')."""
    table = {label: {other: 0 for other in (*labels, 'OTHER')} for label in labels}
    for expected, predicted in rows:
        if expected not in labels:
            continue
        table[expected][predicted if predicted in labels else 'OTHER'] += 1
    return table


def classification(rows, labels, positive=None):
    """Accuracy, per-class P/R/F1, macro P/R/F1 and (optionally) one class as the headline positive."""
    rows = [(e, p) for e, p in rows if e in labels]
    per_class = {}
    for label in labels:
        tp = sum(1 for e, p in rows if e == label and p == label)
        fp = sum(1 for e, p in rows if e != label and p == label)
        fn = sum(1 for e, p in rows if e == label and p != label)
        per_class[label] = {**prf(tp, fp, fn), 'support': sum(1 for e, _ in rows if e == label)}
    present = [label for label in labels if per_class[label]['support']]
    macro = {key: (round(sum(per_class[l][key] or 0 for l in present) / len(present), 4) if present else None)
             for key in ('precision', 'recall', 'f1')}
    result = {'n': len(rows), 'accuracy': ratio(sum(1 for e, p in rows if e == p), len(rows)),
              'per_class': per_class, 'macro': macro, 'confusion': confusion(rows, labels)}
    if positive:
        result['positive_class'] = positive
        result.update({key: per_class[positive][key] for key in ('precision', 'recall', 'f1')})
    return result


def binary(rows):
    """rows: [(expected: bool, predicted: bool)] -> precision, recall, false positive rate, counts."""
    tp = sum(1 for e, p in rows if e and p)
    fp = sum(1 for e, p in rows if not e and p)
    fn = sum(1 for e, p in rows if e and not p)
    tn = sum(1 for e, p in rows if not e and not p)
    return {**prf(tp, fp, fn), 'tn': tn, 'n': len(rows), 'false_positive_rate': ratio(fp, fp + tn)}


def entity_gate(rows):
    """rows: [(expected_excluded: bool, predicted_excluded: bool)].

    False exclusion: the gate ruled a clause out that the expert says applies (or is unknown).
    False inclusion: the expert expected the gate to rule the clause out and it did not.
    """
    false_exclusion = sum(1 for e, p in rows if not e and p)
    false_inclusion = sum(1 for e, p in rows if e and not p)
    return {'n': len(rows), 'expected_excluded': sum(1 for e, _ in rows if e),
            'false_exclusions': false_exclusion, 'false_inclusions': false_inclusion,
            'false_exclusion_rate': ratio(false_exclusion, sum(1 for e, _ in rows if not e)),
            'false_inclusion_rate': ratio(false_inclusion, sum(1 for e, _ in rows if e))}


def retrieval(ranks, hits):
    """ranks: best rank of an expected passage per obligation (None = not retrieved); hits: decision cited an expected passage."""
    ranked = [r for r in ranks if r is not None]
    at = lambda k: ratio(sum(1 for r in ranked if r <= k), len(ranks))
    return {'n': len(ranks), 'recall_at_3': at(3), 'recall_at_5': at(5), 'recall_at_10': at(10),
            'mrr': ratio(sum(1 / r for r in ranked), len(ranks)) if ranks else None,
            'evidence_hit_rate': ratio(sum(1 for h in hits if h), len(hits))}


def proposals(rows):
    """rows: [{'needed': bool, 'produced': bool, 'draft_attempted': bool, 'draft_valid': bool}]."""
    needed = [r for r in rows if r['needed']]
    drafts = [r for r in rows if r['draft_attempted']]
    return {'needed': len(needed), 'produced': sum(1 for r in needed if r['produced']),
            'success_rate': ratio(sum(1 for r in needed if r['produced']), len(needed)),
            'drafts_attempted': len(drafts), 'schema_validity_rate': ratio(sum(1 for r in drafts if r['draft_valid']), len(drafts))}


def reliability(predictions, calls):
    """UNKNOWN share of the assessed judgements; failure, timeout and malformed-answer shares of the model calls."""
    assessed = [p for p in predictions if p['coverage_assessed']]
    unknown = sum(1 for p in predictions if p['applicability'] == 'UNKNOWN') + sum(1 for p in assessed if p['coverage'] == 'UNKNOWN')
    failures = [c for c in calls if c.get('status') != 'OK']
    timeouts = [c for c in failures if 'time limit' in str(c.get('error', '')).lower() or 'timeout' in str(c.get('error', '')).lower()]
    return {'judgements': len(predictions) + len(assessed), 'unknown': unknown,
            'unknown_rate': ratio(unknown, len(predictions) + len(assessed)),
            'calls': len(calls), 'failures': len(failures), 'model_failure_rate': ratio(len(failures), len(calls)),
            'timeouts': len(timeouts), 'timeout_rate': ratio(len(timeouts), len(calls)),
            'malformed_responses': sum(p.get('malformed_responses', 0) for p in predictions),
            'malformed_response_rate': ratio(sum(p.get('malformed_responses', 0) for p in predictions), len(calls)),
            # v0.18: the failure kinds a local runtime produces, apart. A refused prompt (context budget) and an
            # open circuit are counted among the failures above too; retries are the transient HTTP errors the
            # provider absorbed (a call that succeeded on its third attempt is one call with two retries).
            'context_overflows': sum(1 for c in calls if c.get('status') == 'CONTEXT_BUDGET_EXCEEDED'),
            'circuit_open': sum(1 for c in calls if c.get('status') == 'CIRCUIT_OPEN'),
            'retries': sum(int(c.get('retry_count') or 0) for c in calls),
            # v0.19: the provider's failure code (TIMEOUT, OUTPUT_TRUNCATED, EMPTY_RESPONSE ...) when the call record
            # carries one; a record written before v0.19 is counted under its status (PROVIDER_FAILURE ...).
            'failures_by_code': failures_by_code(calls)}


def failures_by_code(calls) -> dict:
    return dict(Counter(str(c.get('failure_code') or c.get('status') or 'UNKNOWN') for c in calls if c.get('status') != 'OK'))


def performance(case_results):
    wall = sum(r['wall_seconds'] for r in case_results)
    calls = [c for r in case_results for c in r['calls']]
    obligations = sum(len(r['predictions']) for r in case_results)
    hits = sum(1 for c in calls if c.get('cache_hit'))
    model_ms = sum(c.get('elapsed_ms') or 0 for c in calls)
    by_task_ms = {}
    for call in calls:
        task = call.get('task') or 'unlabelled'
        by_task_ms[task] = by_task_ms.get(task, 0) + (call.get('elapsed_ms') or 0)
    # The packet's stage timings are milliseconds (the harness adds 'parse': snapshot and policy reading);
    # counters such as embedding_cache_hits sit in the same dict and are not time.
    stage_ms = {}
    for result in case_results:
        for stage, value in (result.get('timings') or {}).items():
            if isinstance(value, (int, float)) and not stage.endswith(('_hits', '_count')):
                stage_ms[stage] = stage_ms.get(stage, 0) + value
    return {'cases': len(case_results), 'wall_seconds': round(wall, 1),
            'model_seconds': round(model_ms / 1000, 1),
            'llm_calls': len(calls), 'prompt_tokens': sum(c.get('prompt_tokens') or 0 for c in calls),
            'output_tokens': sum(c.get('output_tokens') or 0 for c in calls),
            'cache_hits': hits, 'cache_hit_ratio': ratio(hits, len(calls)),
            'obligations': obligations, 'seconds_per_obligation': ratio(wall, obligations),
            'calls_by_task': dict(Counter(c.get('task') or 'unlabelled' for c in calls)),
            'slowest_cases': sorted(({'case_id': r['case_id'], 'wall_seconds': round(r['wall_seconds'], 1)} for r in case_results),
                                    key=lambda x: -x['wall_seconds'])[:5],
            # v0.18: per-obligation cost, so runs over different case subsets compare; live calls are the ones
            # that reached the model (a cache hit costs nothing and says nothing about the model's speed).
            'calls_per_obligation': ratio(len(calls), obligations),
            'model_seconds_per_obligation': ratio(model_ms / 1000, obligations),
            'live_calls': len(calls) - hits,
            'model_seconds_by_task': {task: round(ms / 1000, 1) for task, ms in sorted(by_task_ms.items(), key=lambda kv: -kv[1])},
            'stage_seconds': {stage: round(ms / 1000, 1) for stage, ms in sorted(stage_ms.items(), key=lambda kv: -kv[1])},
            **call_costs(calls, obligations),
            'obligation_stages': obligation_stages([p for r in case_results for p in r['predictions']]),
            'runtime': runtime(calls)}


def call_costs(calls, obligations) -> dict:
    """v0.19 cost per obligation by kind of call.

    Reasoning calls are where the time goes: in the v0.18 final run 429 of 1254 calls had thinking on
    and took 82% of the live model seconds (6104.7 s), at about 22 s each against 4 s without it. So
    they are counted apart, all and live (a cache hit costs nothing), and so are the "strong model"
    calls: the reasoning judge's own questions (task judge.*), which v0.19 routes to the thinking
    verifier only when a passage is escalated. The denominator is every extracted obligation, as for
    calls_per_obligation.
    """
    reasoning = [c for c in calls if thinking(c)]
    strong = [c for c in reasoning if str(c.get('task') or '').startswith('judge.')]
    live = [c for c in calls if not c.get('cache_hit')]
    prompt, output = sum(c.get('prompt_tokens') or 0 for c in calls), sum(c.get('output_tokens') or 0 for c in calls)
    return {'thinking_calls': len(reasoning), 'thinking_calls_live': sum(1 for c in reasoning if not c.get('cache_hit')),
            'thinking_calls_per_obligation': ratio(len(reasoning), obligations),
            'strong_model_calls': len(strong), 'strong_model_calls_per_obligation': ratio(len(strong), obligations),
            'live_calls_per_obligation': ratio(len(live), obligations), 'total_tokens': prompt + output}


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def rate(values) -> dict | None:
    """{calls, mean, median} of per-call rates; None without any (cache hits and pre-v0.19 records carry none)."""
    values = [float(v) for v in values if _number(v)]
    return {'calls': len(values), 'mean': round(mean(values), 2), 'median': round(median(values), 2)} if values else None


def in_time_order(calls) -> tuple:
    """(calls, ordered): sorted by start time ('at') when every record has one, else as given.

    harness.drain_calls concatenated the call logs (the extraction provider's, then the judge's ...), so a
    results list or ai-calls.jsonl written before that was fixed is not in time order: in the smoke run of
    24 September 2026 three judge.verify calls (17:33:36-17:35:33) sit before an enrich call of 17:33:16.
    The sort is stable, so calls started in the same instant keep their order.
    """
    calls = list(calls)
    if calls and all(isinstance(c.get('at'), str) and c.get('at') for c in calls):
        return sorted(calls, key=lambda c: c['at']), True
    return calls, not calls


def runtime(calls) -> dict:
    """Where the model time went by model and context window, how fast it generated, how much ran on the GPU (v0.19 WS6c).

    Measured on the target laptop (RTX 4070, 8 GB): qwen3:8b at num_ctx 16,384 runs 19.5% on the CPU, at 8,192
    fully on the GPU, and Ollama keeps one model resident, so every switch between the 4b and the 8b model is
    a reload of 3-7 s. Live calls are the ones that were not cache hits (as performance.live_calls). Rates are
    the per-call tokens_per_second / prompt_tokens_per_second the provider derived from Ollama's eval
    durations; the GPU share is the share of live calls whose gpu_fraction (size_vram / size from /api/ps) was
    100%, by model and window. model_switches counts consecutive calls that reached the model (not refused by
    admission or an open circuit) with another model, in start-time order; window_switches the same model
    at another num_ctx (also a reload); model_loads the calls whose load_duration was a real load. A record
    without a field (a cache hit, a run before v0.19) adds nothing to that field's numbers.
    """
    live = [c for c in calls if not c.get('cache_hit')]
    model_of = lambda c: str(c.get('model') or 'unknown')
    ctx_of = lambda c: c.get('num_ctx') if _number(c.get('num_ctx')) else None
    ctx_key = lambda c: str(ctx_of(c)) if ctx_of(c) is not None else '?'
    pair_key = lambda c: f'{model_of(c)} @ {ctx_key(c)}'
    order = lambda c: (model_of(c), ctx_of(c) if ctx_of(c) is not None else 10**9)
    by_model, by_ctx, by_pair, gpu = {}, {}, {}, {}
    for call in sorted(live, key=order):
        for table, key in ((by_model, model_of(call)), (by_ctx, ctx_key(call)), (by_pair, pair_key(call))):
            entry = table.setdefault(key, {'live_calls': 0, 'model_ms': 0})
            entry['live_calls'] += 1
            entry['model_ms'] += int(call.get('elapsed_ms') or 0)
        share = gpu.setdefault(pair_key(call), {'live_calls': 0, 'observed': 0, 'full_gpu': 0, 'fractions': [], 'processors': Counter()})
        share['live_calls'] += 1
        if _number(call.get('gpu_fraction')):
            share['observed'] += 1
            share['full_gpu'] += call['gpu_fraction'] >= FULL_GPU
            share['fractions'].append(call['gpu_fraction'])
        if call.get('processor'):
            share['processors'][str(call['processor'])] += 1
    seconds = lambda table: {key: {'live_calls': e['live_calls'], 'model_seconds': round(e['model_ms'] / 1000, 1)} for key, e in table.items()}
    by_ctx = dict(sorted(by_ctx.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 10**9))
    models = sorted({model_of(c) for c in live})
    rates = lambda field: {'all': rate(c.get(field) for c in live),
                           'by_model': {model: rate(c.get(field) for c in live if model_of(c) == model) for model in models
                                        if rate(c.get(field) for c in live if model_of(c) == model)}}
    actions = Counter(str(c['window_action']) for c in live if c.get('window_action'))
    sent, ordered = in_time_order(c for c in live if c.get('status') not in NOT_SENT)
    pairs = list(zip(sent, sent[1:]))
    loads = [c['load_duration_ms'] for c in sent if _number(c.get('load_duration_ms')) and c['load_duration_ms'] >= RELOAD_MS]
    return {'live_calls': len(live), 'by_model': seconds(by_model), 'by_num_ctx': seconds(by_ctx), 'by_model_ctx': seconds(by_pair),
            'tokens_per_second': rates('tokens_per_second'), 'prompt_tokens_per_second': rates('prompt_tokens_per_second'),
            'gpu': {key: {'live_calls': s['live_calls'], 'observed': s['observed'], 'full_gpu': s['full_gpu'],
                          'full_gpu_share': ratio(s['full_gpu'], s['observed']),
                          'min_gpu_fraction': min(s['fractions']) if s['fractions'] else None, 'processors': dict(s['processors'])}
                    for key, s in gpu.items()},
            'window_actions': {**{a: actions[a] for a in WINDOW_ACTIONS if actions[a]}, **{a: n for a, n in sorted(actions.items()) if a not in WINDOW_ACTIONS}},
            'fallbacks': sum(actions[a] for a in FALLBACK_ACTIONS),
            'model_switches': sum(1 for a, b in pairs if model_of(a) != model_of(b)),
            'window_switches': sum(1 for a, b in pairs if model_of(a) == model_of(b) and ctx_of(a) != ctx_of(b)),
            'model_loads': len(loads), 'load_seconds': round(sum(loads) / 1000, 1), 'time_ordered': ordered}


def obligation_stages(predictions) -> dict:
    """Seconds per assessed obligation, stage by stage, from the packet rows' timings_ms (v0.19 engine).

    The case-level stage timers say where a run's time went but not what one obligation costs, and the
    v0.18 row total (elapsed_ms: mean 77.6 s, median 12.9 s, max 590 s over 93 rows) hides which stage
    made a slow row slow. Assessed = coverage was assessed (a duty the gates ruled out never reaches the
    passage stages). A stage an obligation did not run counts 0 s for it, so every stage mean is over the
    same rows and the stage means add up to the mean of the rows' stage sums. ``total`` is the row's own
    elapsed_ms when recorded (v0.18 packets have only that), else the sum of its stages.

    ``obligations`` counts the rows whose timings name the stage: the engine records a stage only when it
    ran, in whole milliseconds, so a stage that ran in under 1 ms is recorded as 0 and still ran (smoke run
    C12: retrieval and aggregation 0 ms in all four rows).
    """
    rows = [p for p in predictions if isinstance(p, dict) and p.get('coverage_assessed')]
    timed = [{k: v for k, v in p['timings_ms'].items() if isinstance(v, (int, float)) and not k.endswith(('_hits', '_count'))}
             for p in rows if isinstance(p.get('timings_ms'), dict)]
    names = [s for s in STAGE_ORDER if any(s in t for t in timed)] + sorted({s for t in timed for s in t} - set(STAGE_ORDER))
    seconds = lambda values: {'mean_s': round(sum(values) / len(values) / 1000, 2), 'median_s': round(median(values) / 1000, 2),
                              'max_s': round(max(values) / 1000, 2), 'total_s': round(sum(values) / 1000, 1)} if values else None
    stages = {}
    for name in names:
        values = [t.get(name, 0) for t in timed]
        stages[name] = {**seconds(values), 'obligations': sum(1 for t in timed if name in t)}
    totals = [p['elapsed_ms'] if isinstance(p.get('elapsed_ms'), (int, float)) else
              sum(v for k, v in (p.get('timings_ms') or {}).items() if isinstance(v, (int, float)) and not k.endswith(('_hits', '_count')))
              for p in rows if isinstance(p.get('elapsed_ms'), (int, float)) or isinstance(p.get('timings_ms'), dict)]
    return {'assessed_obligations': len(rows), 'with_stage_timings': len(timed), 'stages': stages,
            'total': {**seconds(totals), 'obligations': len(totals)} if totals else None}


def decisions(predictions):
    """Which gate or rule settled each applicability state (v0.18 trace.decided_by; older packets name their
    applicability_rule, whose values decided_by extends), and how often a row was sent to human review."""
    return {'decided_by': dict(Counter(p.get('decided_by') or p.get('applicability_rule') or 'NOT_RECORDED' for p in predictions)),
            'review_flags': dict(Counter(flag for p in predictions for flag in p.get('review_flags') or []))}


def extraction(pairs_by_case, extras_by_case):
    required = [(e, p) for pairs in pairs_by_case for e, p in pairs if e.required]
    found = sum(1 for _, p in required if p is not None)
    predicted = sum(len(pairs) for pairs in pairs_by_case) + sum(len(x) for x in extras_by_case)
    matched = sum(1 for pairs in pairs_by_case for _, p in pairs if p is not None)
    extras = sum(len(x) for x in extras_by_case)
    return {'expected_required': len(required), 'found_required': found, 'recall': ratio(found, len(required)),
            'predicted': matched + extras, 'matched': matched, 'unexpected': extras, 'precision': ratio(matched, matched + extras)}


# v0.19 (problem 1 F): why an expectation has no scored row (identifiers.MISSING_REASONS); a row scored before v0.19 has none.
MISSING_REASONS = ('EXTRACTION_MISSED', 'GROUNDING_REJECTED', 'MATCHING_FAILED')


def missing_rows(scored) -> dict:
    """The expected / required / scored / missing rows of scored expectations (harness.score_case items) and why each
    missing row is missing: counts per MISSING_REASONS member (NOT_RECORDED for rows scored before v0.19), for all
    missing rows and for the required ones, and the detail codes under each reason. Integers only."""
    rows = [item for item in scored if isinstance(item, dict)]
    missing = [item for item in rows if not item.get('matched')]
    required = lambda items: [item for item in items if (item.get('expected') or {}).get('required', True)]
    reason = lambda item: item.get('missing_reason') or 'NOT_RECORDED'
    by = lambda items: {**{key: 0 for key in MISSING_REASONS}, **dict(Counter(reason(item) for item in items))}
    details = {}
    for item in missing:
        entry = details.setdefault(reason(item), {})
        detail = item.get('missing_detail') or 'NOT_RECORDED'
        entry[detail] = entry.get(detail, 0) + 1
    return {'expected_rows': len(rows), 'required_rows': len(required(rows)), 'scored_rows': len(rows) - len(missing),
            'scored_required': len(required(rows)) - len(required(missing)), 'missing_rows': len(missing),
            'missing_required': len(required(missing)), 'missing_by_reason': by(missing), 'missing_required_by_reason': by(required(missing)),
            'missing_details': {key: dict(sorted(details[key].items())) for key in [*MISSING_REASONS, 'NOT_RECORDED'] if key in details}}
