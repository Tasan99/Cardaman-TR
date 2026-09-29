"""Run evaluation cases through the pilot's own analysis and score the results."""
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from regchain.extraction.providers import configured_judge, configured_provider, raw_answers, usage_summary
from regchain.extraction.rules import RulesProvider
from regchain.pilot.engine import DEFAULT_CATEGORIES, analyze
from regchain.pilot.policies import read_policy
from regchain.pilot.rerank import configured_reranker
from regchain.pilot.schema import Company
from regchain.pilot.semantic import configured_embedder
from regchain.pilot.sources import load_sources, select_targets
from . import metrics as m
from .identifiers import (article_of, basis_counts, clause_id, duty_fingerprint, duty_span, match_predictions, obligation_fingerprint,
                          obligation_key, pair_predictions, source_index)
from .manifest import build_manifest, snapshot_fingerprint
from .schema import Dataset

MODEL_ENV = {'extraction_model': ('LLM_MODEL', 'LLM_MODEL_DIGEST'), 'judge_model': ('JUDGE_MODEL', 'JUDGE_MODEL_DIGEST'),
             'embed_model': ('EMBED_MODEL', 'EMBED_MODEL_DIGEST')}
# v0.18 pipeline settings a run or benchmark leg may set (override key -> env key, allowed values).
# v0.19 adds the coverage pipeline (v18 code default, v19 = fast classifier + escalated thinking verifier)
# and the judge window: the mode (fixed, or adaptive: the large window only for a prompt that does not fit the
# base one) and the two windows. The code default of the base window stays 16,384 (a legacy test pins it), at
# which qwen3:8b runs 19.5% on the CPU of the 8 GB target laptop; a leg that does not set it gets that, so the
# v0.19 matrices set 8,192 as the product does. A window is any whole number of tokens the provider accepts.
WINDOW_TOKENS = range(2048, 131072 + 1)
PIPELINE_ENV = {'applicability_clear_match': ('APPLICABILITY_CLEAR_MATCH', ('model', 'quick', 'rule')),
                'relevance_screen': ('RELEVANCE_SCREEN', ('on', 'off')),
                'coverage_pipeline': ('COVERAGE_PIPELINE', ('v18', 'v19')),
                'ctx_mode': ('JUDGE_CTX_MODE', ('fixed', 'adaptive')),
                'judge_num_ctx': ('JUDGE_NUM_CTX', WINDOW_TOKENS),
                'judge_num_ctx_large': ('JUDGE_NUM_CTX_LARGE', WINDOW_TOKENS)}
# v0.19 t7: the replay sidecar of a run, one line per model answer (providers.raw_answers): the case, the call record's
# request_hash, raw_output_sha256, task and ids, and the answer text exactly as returned, so a live run can be replayed
# offline. Written only, never read during a run (it is no cache); ai-calls.jsonl stays text-free and packets are unchanged.
RAW_LOG = 'ai-raw.jsonl'


def load_dataset(path: Path) -> Dataset:
    return Dataset.model_validate_json(Path(path).read_text(encoding='utf-8'))


def ollama_digests(base_url: str) -> dict:
    try:
        with httpx.Client(timeout=10, trust_env=False) as client:
            return {item['name']: item['digest'] for item in client.get(base_url.rstrip('/') + '/api/tags').json().get('models', [])}
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return {}


def configure_models(overrides: dict):
    """Apply model overrides for a benchmark leg (env is how the providers are configured).

    A digest is resolved from the local Ollama catalogue when the model changes, so a leg
    never runs with the previous leg's pin.
    """
    applied = {}
    digests = None
    for key, (model_key, digest_key) in MODEL_ENV.items():
        model = overrides.get(key)
        if not model:
            continue
        if digests is None:
            digests = ollama_digests(os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'))
        if model not in digests:
            raise ValueError(f'{model} is not installed in the local Ollama; the benchmark leg cannot be pinned')
        os.environ[model_key], os.environ[digest_key] = model, digests[model]
        applied[model_key] = model
    if overrides.get('judge_thinking'):
        os.environ['JUDGE_THINKING'] = overrides['judge_thinking']
        applied['JUDGE_THINKING'] = overrides['judge_thinking']
    if 'rerank_model' in overrides:
        os.environ['RERANK_MODEL'] = overrides.get('rerank_model') or ''
        applied['RERANK_MODEL'] = os.environ['RERANK_MODEL']
    for key, (env_key, allowed) in PIPELINE_ENV.items():
        value = str(overrides.get(key) or '').strip().lower()
        if not value:
            continue
        if isinstance(allowed, range):
            if not (value.isdigit() and int(value) in allowed):
                raise ValueError(f'{key} must be a number of tokens from {allowed.start} to {allowed.stop - 1}')
            value = str(int(value))
        elif value not in allowed:
            raise ValueError(f'{key} must be one of {", ".join(allowed)}')
        os.environ[env_key] = applied[env_key] = value
    return applied


class Components:
    def __init__(self, provider: str, retrieval: str, reranker: bool):
        self.mode = {'provider': provider, 'retrieval': retrieval, 'reranker': bool(reranker)}
        if provider == 'rules':
            self.provider = RulesProvider()
            self.judge = self.provider
        else:
            self.provider = configured_provider('ollama')
            self.judge = configured_judge(self.provider)
        self.embedder = configured_embedder() if retrieval == 'hybrid' else None
        self.reranker, self.rerank_status = (configured_reranker() if reranker else (None, {'status': 'off'}))
        self.mode['reranker_status'] = self.rerank_status

    def logs(self):
        logs = []
        for candidate in (self.provider, self.judge, getattr(self.judge, 'quick', None), getattr(self.provider, 'quick', None),
                          getattr(self.provider, 'fast', None)):
            log = getattr(candidate, 'call_log', None)
            if isinstance(log, list) and not any(log is seen for seen in logs):
                logs.append(log)
        return logs

    def drain_calls(self):
        """The calls of one case from every log, merged in start-time order ('at'), and the logs emptied.

        Concatenated log by log (the extraction provider's, then the judge's), ai-calls.jsonl was not in time order
        (smoke run 20260924-173120: three judge.verify calls before an enrich call that started 20 s earlier), and
        with the v0.19 model roles (4b calls in the provider's log, 8b calls in the judge's) any count of model
        switches in file order reads about one per case whatever the real interleaving. Records without a start
        time (test doubles) keep the log order.
        """
        calls, _ = m.in_time_order(call for log in self.logs() for call in log)
        for log in self.logs():
            del log[:]
        return calls


def resolve(base: Path, relative: str) -> Path:
    path = (base / relative).resolve()
    if not path.exists():
        raise FileNotFoundError(f'Fixture not found: {relative} (from {base})')
    return path


def prediction_rows(payload, regulation_id):
    """One scoreable row per obligation of a packet.

    The v0.18 decision-chain fields (decided_by, gate statuses, review flags, the model's own
    answer) are copied when the packet has them; a v0.17 packet has none and gets None / {}.
    """
    rows = []
    sources = source_index(payload)
    for row in payload['obligations']:
        proposal = row['proposal']
        scope = proposal.get('applicability_scope') or {}
        trace = proposal.get('trace') if isinstance(proposal.get('trace'), dict) else {}
        child = scope.get('child_clause') or ''
        action = row['candidate'].get('required_action') or row['candidate'].get('prohibited_action') or ''
        applicability = proposal['applicability']
        assessed = proposal.get('coverage_assessed', True)
        order = list(dict.fromkeys([*(row.get('retrieved_policy_ids') or []), *(row.get('judged_policy_ids') or [])]))
        ranks = {s['source_id']: s['rank'] for s in (row.get('evidence_signals') or []) if s.get('rank')}
        for position, sid in enumerate(order, 1):
            ranks.setdefault(sid, position)
        malformed = 0
        for item in row.get('diagnostics') or []:
            if item.get('code') in ('PROPOSAL_INVALID',):
                malformed += 1
            for result in item.get('results') or []:
                malformed += sum(1 for note in result.get('notes') or [] if note.get('code') == 'JUDGEMENT_INVALID')
        remediation = proposal.get('remediation')
        rows.append({'key': obligation_key(regulation_id, article_of(row['source_label']), child, action),
                     'obligation_id': row['id'], 'source_label': row['source_label'], 'article': article_of(row['source_label']),
                     'clause': clause_id(child), 'subject': row['candidate'].get('subject') or '', 'action': action,
                     'applicability': 'UNKNOWN' if applicability == 'POSSIBLY_APPLIES' else applicability, 'applicability_raw': applicability,
                     'applicability_rule': proposal.get('applicability_rule'), 'entity_gate': scope.get('match') or 'NOT_RECORDED',
                     'coverage': proposal['coverage'] if assessed else 'NOT_ASSESSED', 'coverage_assessed': assessed,
                     'conflict': bool(row.get('signals')), 'policy_evidence': list(proposal.get('policy_evidence') or []),
                     'policy_checks': [{'source_id': c['source_id'], 'relation': c['relation']} for c in proposal.get('policy_checks') or []],
                     'ranks': ranks, 'remediation_status': proposal.get('remediation_status'), 'remediation': bool(remediation),
                     'draft_status': (remediation or {}).get('draft_status'), 'malformed_responses': malformed,
                     'applicability_reason': str(proposal.get('applicability_reason', ''))[:400],
                     'coverage_reason': str(proposal.get('coverage_reason', ''))[:400],
                     'decided_by': trace.get('decided_by'),
                     'gate_status': {g['gate']: g.get('status') for g in trace.get('gates') or [] if isinstance(g, dict) and g.get('gate')},
                     'review_flags': list(proposal.get('review_flags') or []), 'model_decision': trace.get('model_decision'),
                     'subject_gate_status': (trace.get('subject_gate') or {}).get('status'),
                     # v0.19: what this obligation cost, for metrics.obligation_stages (None in older packets).
                     'elapsed_ms': row.get('elapsed_ms'),
                     'timings_ms': dict(row['timings_ms']) if isinstance(row.get('timings_ms'), dict) else None,
                     # v0.19 (FIX 4): the duty's own fields, which identifiers.pair_predictions reads to pair it with a label.
                     **duty_identity(row['candidate'], article_of(row['source_label']), child, action),
                     # v0.19 (problem 1 E): the sentence the duty came from and the fingerprint that names it across runs.
                     **duty_source(row, regulation_id, article_of(row['source_label']), child, action, sources)})
    return rows


def duty_source(row, regulation_id, article, child, action, sources) -> dict:
    """{'source_span': {'start', 'end', 'origin'} or None (offsets into the provision text), 'fingerprint'}: identifiers.duty_span
    and identifiers.obligation_fingerprint (regulation, provision, clause, action/object stems, hash of the source sentence)."""
    source = sources.get(article)
    span = duty_span(row, source)
    obj = ((row.get('structure') or {}).get('object') or {}).get('text') if isinstance(row.get('structure'), dict) else None
    text = source['text'][span['start']:span['end']] if span else None
    return {'source_span': span, 'fingerprint': obligation_fingerprint(regulation_id, article, child, action, obj, text)}


def duty_identity(candidate, article, child, action) -> dict:
    """What names an extracted duty besides its action: modality, conditions, exceptions, deadline and its fingerprint."""
    listed = lambda value: [str(v) for v in value if v] if isinstance(value, list) else ([str(value)] if value else [])
    modality = candidate.get('modality') or None
    return {'modality': modality, 'conditions': listed(candidate.get('conditions')), 'exceptions': listed(candidate.get('exceptions')),
            'deadline': str(candidate['deadline']) if candidate.get('deadline') else None,
            'duty_fingerprint': duty_fingerprint(article, child, modality, action)}


def evidence_passages(payload, needles):
    """The passage ids whose text contains any of the expected evidence substrings (whitespace aside)."""
    squash = lambda text: ' '.join(str(text).split())
    wanted = [squash(n) for n in needles]
    return {c['source_id'] for p in payload['policies'] for c in p['chunks'] if any(n and n in squash(c['text']) for n in wanted)}


def pair_case(expected, predictions, payload):
    """identifiers.pair_predictions with the packet's provision texts: how score_case, the coverage-error report and the
    expert export pair one case, so the three never disagree."""
    return pair_predictions(expected, predictions, source_index(payload))


def score_case(case, payload, predictions):
    matches, extras = pair_case(case.expected_obligations, predictions, payload)
    scored = []
    for match in matches:
        expected, prediction = match['expected'], match['prediction']
        item = {'expected': expected.model_dump(), 'prediction': None if prediction is None else prediction['key'], 'matched': prediction is not None,
                # v0.19 (FIX 4): why the row was (not) paired, integers and names only (identifiers.pair_predictions).
                'match_basis': match['basis'],
                'match': {'candidates': match['candidates'], 'stems': match['stems'], 'stems_found': match['stems_found'],
                          'polarity': match['polarity'], 'obligation_id': None if prediction is None else prediction.get('obligation_id'),
                          # v0.19 (problem 1 E): the label's sentences in its clause, the candidates that came from them.
                          'label_sentences': match['label_sentences'], 'from_sentence': match['from_sentence'],
                          'fingerprint': None if prediction is None else prediction.get('fingerprint')}}
        if prediction is None:
            # v0.19 (problem 1 F): why this expectation has no scored row (identifiers.MISSING_REASONS).
            item.update(missing_reason=match['missing_reason'], missing_detail=match['missing_detail'])
        if prediction is not None:
            item['applicability'] = (expected.applicability, prediction['applicability'])
            if expected.coverage != 'ANY':
                item['coverage'] = (expected.coverage, prediction['coverage'])
            if expected.conflict is not None:
                item['conflict'] = (expected.conflict, prediction['conflict'])
            if expected.entity_gate != 'ANY':
                item['gate'] = (expected.entity_gate == 'MISMATCH', prediction['applicability_rule'] == 'ENTITY_GATE')
            if expected.evidence and prediction['coverage_assessed']:
                passages = evidence_passages(payload, expected.evidence)
                ranks = [prediction['ranks'][sid] for sid in passages if sid in prediction['ranks']]
                item['retrieval'] = {'rank': min(ranks) if ranks else None,
                                     'hit': any(q['source_id'] in passages for q in prediction['policy_evidence']),
                                     'expected_passages': len(passages)}
            item['proposal'] = {'needed': prediction['coverage'] in ('CONFLICT', 'NO_EVIDENCE', 'PARTIAL') and prediction['applicability'] != 'DOES_NOT_APPLY',
                                'produced': prediction['remediation'], 'draft_attempted': prediction['draft_status'] is not None,
                                'draft_valid': prediction['draft_status'] == 'DRAFTED'}
        scored.append(item)
    return scored, extras


def run_case(case, dataset_dir: Path, components: Components, out_dir: Path):
    started = time.monotonic()
    module, chapter = case.regulation_id.split(':', 1)
    result = {'case_id': case.case_id, 'title': case.title, 'categories': case.categories, 'error': None}
    workdir = Path(tempfile.mkdtemp(prefix='eval-'))
    try:
        parse_started = time.monotonic()
        _, sections = load_sources(resolve(dataset_dir, case.regulation_fixture))
        policies = []
        for relative in [*case.policy_documents, *case.optional_control_records]:
            source = resolve(dataset_dir, relative)
            copy = workdir / source.name
            shutil.copyfile(source, copy)
            policies.append(read_policy(copy))
        # Reading the snapshot and the policies happens before analyze(), so the packet's own stage
        # timings leave it out; it is added here in the packet's unit (ms) under its own key.
        parse_ms = int((time.monotonic() - parse_started) * 1000)
        company = Company(**case.company_profile)
        labels = select_targets(sections, module, chapter, 'all', case.target_sections)
        packet = analyze(company, policies, sections, components.provider, labels, embedder=components.embedder, judge=components.judge,
                         categories=list(DEFAULT_CATEGORIES), reranker=components.reranker, target_filter=case.target_sections)
        payload = packet['events'][0]['payload']
        predictions = prediction_rows(payload, case.regulation_id)
        scored, extras = score_case(case, payload, predictions)
        result.update({'head': packet['head'], 'predictions': predictions, 'scored': scored,
                       'extras': [{'key': e['key'], 'article': e['article'], 'clause': e['clause'], 'action': e['action'][:120]} for e in extras],
                       'extraction': [{'article': article_of(c['source']['printed_label']), 'reason': c['reason'],
                                       'candidates': len(c['output']['obligations'])} for c in payload['cases']],
                       'timings': {**(payload.get('timings') or {}), 'parse': parse_ms}, 'targets': payload['regulation'].get('targets'),
                       'helper_count': payload['regulation'].get('helper_count')})
        (out_dir / 'packets').mkdir(exist_ok=True)
        (out_dir / 'packets' / f'{case.case_id}.json').write_text(json.dumps(packet, ensure_ascii=False), encoding='utf-8')
    except Exception as exc:  # a failing case is a result, not the end of the run
        result.update({'error': f'{type(exc).__name__}: {str(exc)[:500]}', 'predictions': [], 'scored': [], 'extras': [], 'extraction': []})
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    result['calls'] = components.drain_calls()
    result['wall_seconds'] = time.monotonic() - started
    return result


class Rows:
    """The scored pairs of one slice of a run (all expectations, or those of one label source)."""

    def __init__(self):
        self.app, self.cov, self.conflict, self.gate, self.ranks, self.hits, self.required = [], [], [], [], [], [], []
        self.expectations = 0

    def classification(self):
        assessed_cov = [(e, p) for e, p in self.cov if e in m.COVERAGE]
        return {'expectations': self.expectations, 'applicability': m.classification(self.app, m.APPLICABILITY, positive='APPLIES'),
                'coverage': m.classification(assessed_cov, m.COVERAGE), 'conflict': m.binary(self.conflict),
                'entity_gate': m.entity_gate(self.gate), 'retrieval': m.retrieval(self.ranks, self.hits),
                # Extraction precision has no label source (an unexpected candidate matches no expectation).
                'extraction': {'expected_required': len(self.required), 'found_required': sum(self.required),
                               'recall': m.ratio(sum(self.required), len(self.required))}}


def item_sources(item, case, dataset) -> dict:
    """{field: label source} of one scored expectation; key None is the expectation as a whole."""
    defaults = (getattr(case, 'default_label_source', None), getattr(dataset, 'default_label_source', None))
    return {field: m.label_source(item['expected'], field, *defaults) for field in (None, *m.LABEL_FIELDS)}


def aggregate(dataset, case_results):
    """Every metric family over the scored pairs of all cases, with the mismatches named.

    v0.19: the same families again per label source (metrics['by_label_source']); each field of an
    expectation counts under its own source, so the slices of one family add up to its total.
    """
    cases = list(dataset.cases)
    # Cases and results are paired by position (zip); a results list out of the dataset's order would pair a
    # case's labels with another case's predictions and score silently wrong numbers.
    if len(cases) != len(case_results) or any(case.case_id != result.get('case_id', case.case_id) for case, result in zip(cases, case_results)):
        raise ValueError('aggregate: the results do not follow the dataset cases one to one '
                         f'({[c.case_id for c in cases][:8]} vs {[r.get("case_id") for r in case_results][:8]})')
    total, by_source = Rows(), {}
    proposal_rows, predictions, calls = [], [], []
    pairs_by_case, extras_by_case, mismatches = [], [], []
    for case, result in zip(cases, case_results):
        predictions.extend(result['predictions'])
        calls.extend(result['calls'])
        pairs_by_case.append(scored_pairs(case, result))
        extras_by_case.append(result['extras'])
        for item in result['scored']:
            expected = item['expected']
            where = f'{case.case_id} md.{expected["article"]}{expected["clause"] or ""}'
            sources = item_sources(item, case, dataset)
            slice_of = lambda field: by_source.setdefault(sources[field], Rows())
            for rows in (total, slice_of(None)):
                rows.expectations += 1
                if expected['required']:
                    rows.required.append(bool(item['matched']))
            if not item['matched']:
                if expected['required']:
                    mismatches.append({'kind': 'extraction_miss', 'where': where, 'expected': expected.get('action_keywords') or expected['notes'],
                                       'label_source': sources[None], 'missing_reason': item.get('missing_reason') or 'NOT_RECORDED',
                                       'missing_detail': item.get('missing_detail')})
                continue
            for rows in (total, slice_of('applicability')):
                rows.app.append(item['applicability'])
            if item['applicability'][0] != item['applicability'][1]:
                kind = ('applicability_false_positive' if item['applicability'][1] == 'APPLIES' else
                        'applicability_false_negative' if item['applicability'][0] == 'APPLIES' else 'applicability_mismatch')
                mismatches.append({'kind': kind, 'where': where, 'expected': item['applicability'][0], 'predicted': item['applicability'][1],
                                   'label_source': sources['applicability']})
            if 'coverage' in item:
                for rows in (total, slice_of('coverage')):
                    rows.cov.append(item['coverage'])
                if item['coverage'][0] != item['coverage'][1]:
                    mismatches.append({'kind': 'coverage_mismatch', 'where': where, 'expected': item['coverage'][0], 'predicted': item['coverage'][1],
                                       'label_source': sources['coverage']})
            if 'conflict' in item:
                for rows in (total, slice_of('conflict')):
                    rows.conflict.append(item['conflict'])
                if item['conflict'][0] != item['conflict'][1]:
                    mismatches.append({'kind': 'conflict_false_positive' if item['conflict'][1] else 'conflict_false_negative', 'where': where,
                                       'label_source': sources['conflict']})
            if 'gate' in item:
                for rows in (total, slice_of('entity_gate')):
                    rows.gate.append(item['gate'])
                if item['gate'][0] != item['gate'][1]:
                    mismatches.append({'kind': 'entity_gate_false_inclusion' if item['gate'][0] else 'entity_gate_false_exclusion', 'where': where,
                                       'label_source': sources['entity_gate']})
            if 'retrieval' in item:
                # The expected evidence text belongs to the coverage label, so retrieval follows its source.
                for rows in (total, slice_of('coverage')):
                    rows.ranks.append(item['retrieval']['rank'])
                    rows.hits.append(item['retrieval']['hit'])
            proposal_rows.append(item['proposal'])
        for extra in result['extras']:
            mismatches.append({'kind': 'unexpected_candidate', 'where': f'{case.case_id} md.{extra["article"]}({extra["clause"]})', 'predicted': extra['action']})
    assessed_cov = [(e, p) for e, p in total.cov if e in m.COVERAGE]
    # The headline UNKNOWN rate stays reliability.unknown_rate (the v0.17 definition, over applicability and
    # coverage judgements); this one is the share of scored applicability rows the run left UNKNOWN.
    applicability = {**m.classification(total.app, m.APPLICABILITY, positive='APPLIES'),
                     'unknown_rate_applicability': m.ratio(sum(1 for _, p in total.app if p == 'UNKNOWN'), len(total.app)), **m.decisions(predictions)}
    return {'applicability': applicability,
            'coverage': {**m.classification(assessed_cov, m.COVERAGE),
                         'not_assessed_expected': sum(1 for e, _ in total.cov if e == 'NOT_ASSESSED'),
                         'not_assessed_correct': sum(1 for e, p in total.cov if e == 'NOT_ASSESSED' and p == 'NOT_ASSESSED')},
            'conflict': m.binary(total.conflict), 'entity_gate': m.entity_gate(total.gate), 'retrieval': m.retrieval(total.ranks, total.hits),
            'proposals': m.proposals(proposal_rows), 'reliability': m.reliability(predictions, calls), 'performance': m.performance(case_results),
            'extraction': m.extraction(pairs_by_case, extras_by_case),
            # v0.19 (FIX 4): how the scored rows were paired (identifiers.BASES; NOT_RECORDED for rows scored before v0.19);
            # (problem 1 F) the expected, required, scored and missing rows, and why each missing row is missing.
            'matching': {**basis_counts([item for result in case_results for item in result['scored']]),
                         **m.missing_rows([item for result in case_results for item in result['scored']])},
            'by_label_source': {source: by_source[source].classification() for source in
                                [*[s for s in m.LABEL_SOURCES if s in by_source], *sorted(set(by_source) - set(m.LABEL_SOURCES))]},
            'errors': [{'case_id': r['case_id'], 'error': r['error']} for r in case_results if r['error']],
            'mismatches': mismatches}


def scored_pairs(case, result) -> list:
    """[(expectation, prediction or None)] of a case as its scored rows paired them.

    The extraction counts are read from the same pairing as every other family: re-pairing the saved
    predictions here would score a run written with an older matcher with two matchers at once (its
    rows and extras by the old one, its extraction counts by the current one). Rows that do not follow
    the case's expectations one to one (a failed case has none) are paired again, as before v0.19.
    """
    expected, scored = list(case.expected_obligations), result.get('scored') or []
    if len(scored) != len(expected):
        return match_predictions(expected, result['predictions'])[0]
    by_key = {p['key']: p for p in result['predictions'] if isinstance(p, dict) and 'key' in p}
    return [(item, (by_key.get(row['prediction']) or {'key': row['prediction']}) if row['matched'] else None)
            for item, row in zip(expected, scored)]


def snapshot_fingerprints(dataset_dir: Path, cases) -> dict:
    """{regulation_fixture: fingerprint of the parsed snapshot} for the fixtures the cases read.

    The regression gate compares these between a run and its baseline: the same retained bytes
    read by a changed parser give other labels or texts, and the two runs no longer score the
    same regulation. A fixture that does not load is recorded as such (the gate then fails).
    """
    fingerprints = {}
    for fixture in sorted({case.regulation_fixture for case in cases}):
        try:
            _, sections = load_sources(resolve(dataset_dir, fixture))
            fingerprints[fixture] = snapshot_fingerprint(sections)
        except (OSError, ValueError, KeyError) as exc:
            fingerprints[fixture] = f'unreadable: {type(exc).__name__}'
    return fingerprints


def run(dataset_path: Path, out_root: Path, provider='rules', retrieval='lexical', reranker=False, cases=None, label='', overrides=None):
    dataset = load_dataset(dataset_path)
    dataset_dir = Path(dataset_path).resolve().parent
    applied = configure_models(overrides or {})
    if overrides and overrides.get('rerank_model'):
        reranker = True
    components = Components(provider, retrieval, reranker)
    selected = [c for c in dataset.cases if not cases or c.case_id in cases]
    if not selected:
        raise ValueError('No case matched the selection')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    run_dir = Path(out_root) / (stamp + (f'-{label}' if label else '') + f'-{provider}')
    run_dir.mkdir(parents=True, exist_ok=False)
    mode = {**components.mode, 'cases': [c.case_id for c in selected], 'overrides': applied}
    # v0.19 (B1): the dataset path is written into manifest.json (top level, outside manifest_sha256: the same
    # configuration run from another folder is the same configuration). Before, it was set on the mode after
    # the manifest was written, so `report --run DIR` without --dataset raised a KeyError on every real run.
    manifest = build_manifest(dataset, mode, snapshots=snapshot_fingerprints(dataset_dir, selected), dataset_path=Path(dataset_path).resolve())
    (run_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    mode['dataset_path'] = str(Path(dataset_path).resolve())
    results = []
    with (run_dir / 'ai-calls.jsonl').open('a', encoding='utf-8') as log, (run_dir / RAW_LOG).open('a', encoding='utf-8') as raw_log:
        for case in selected:
            print(f'[{datetime.now().strftime("%H:%M:%S")}] {case.case_id} ...', flush=True)
            with raw_answers() as answers:
                result = run_case(case, dataset_dir, components, run_dir)
            for call in result['calls']:
                log.write(json.dumps({'case_id': case.case_id, **call}, ensure_ascii=False) + '\n')
            log.flush()
            for answer in answers:
                raw_log.write(json.dumps({'case_id': case.case_id, **answer}, ensure_ascii=False) + '\n')
            raw_log.flush()
            results.append(result)
            # Written after every case: a run stopped half-way still leaves scoreable results.
            (run_dir / 'results.json').write_text(json.dumps([{k: v for k, v in r.items() if k != 'calls'} for r in results], ensure_ascii=False, indent=1),
                                                  encoding='utf-8')
            (run_dir / 'progress.json').write_text(json.dumps({'done': len(results), 'total': len(selected), 'last': case.case_id,
                                                               'updated_at': datetime.now(timezone.utc).isoformat()}), encoding='utf-8')
            print(f'    {round(result["wall_seconds"])}s, {len(result["predictions"])} obligations, {len(result["calls"])} calls'
                  + (f', ERROR {result["error"]}' if result['error'] else ''), flush=True)
    return finalize(run_dir, dataset, selected, results, manifest)


def finalize(run_dir: Path, dataset, selected, results, manifest, overlay=None):
    """Score what was run, write metrics.json and report.md; also used to score a run stopped early."""
    # model_copy keeps every dataset-level field (v0.19 default_label_source); rebuilding the Dataset by hand dropped them.
    subset = dataset.model_copy(update={'cases': list(selected)})
    metrics = aggregate(subset, results)
    metrics['usage'] = usage_summary([[c for r in results for c in r['calls']]])
    metrics['completed_cases'] = len(results)
    if overlay is not None:
        metrics['reviewed_labels'] = overlay_record(overlay)
    (run_dir / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    from .report import render_markdown
    (run_dir / 'report.md').write_text(render_markdown(manifest, metrics, results), encoding='utf-8')
    return run_dir, manifest, metrics


# What metrics.json and the report say about a reviewed-labels overlay (labels.apply_reviewed): the file, what was applied
# and skipped, and (v0.19) whether only entries the audit log approved counted (strict) with the log's verification problems.
OVERLAY_KEYS = ('path', 'sha256', 'applied', 'skipped', 'strict', 'audit')


def overlay_record(overlay) -> dict:
    return {key: overlay[key] for key in OVERLAY_KEYS if key in overlay}


def rescore(run_dir: Path, dataset_path=None, reviewed_labels=None):
    """Score a run again from its saved packets against the dataset now on disk.

    Labels get corrected after a run (the 23 September review moved three of them); the model
    output does not change, so the packets are re-read and re-paired instead of re-running the
    models. The original results.json and manifest.json stay as they were: the new scores go to
    results.rescored.json, and rescore.json records which dataset they were scored against.

    ``reviewed_labels`` (v0.18) names a reviewed-labels file whose ACCEPTED entries are laid over
    the labels in memory; the dataset file is never written, and rescore.json records the overlay.
    """
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    dataset_path, _ = resolve_dataset(run_dir, manifest, dataset_path)
    dataset = load_dataset(dataset_path)
    overlay = None
    if reviewed_labels:
        from .labels import apply_reviewed
        dataset, overlay = apply_reviewed(dataset, Path(reviewed_labels), dataset_path)
    results = json.loads((run_dir / 'results.json').read_text(encoding='utf-8'))
    cases = {c.case_id: c for c in dataset.cases}
    rescored = 0
    for result in results:
        packet_path = run_dir / 'packets' / f"{result['case_id']}.json"
        if result.get('error') or not packet_path.is_file() or result['case_id'] not in cases:
            continue
        case = cases[result['case_id']]
        payload = json.loads(packet_path.read_text(encoding='utf-8'))['events'][0]['payload']
        predictions = prediction_rows(payload, case.regulation_id)
        scored, extras = score_case(case, payload, predictions)
        result.update({'predictions': predictions, 'scored': scored,
                       'extras': [{'key': e['key'], 'article': e['article'], 'clause': e['clause'], 'action': e['action'][:120]} for e in extras]})
        rescored += 1
    (run_dir / 'results.rescored.json').write_text(json.dumps(results, ensure_ascii=False), encoding='utf-8')
    from hashlib import sha256
    (run_dir / 'rescore.json').write_text(json.dumps({
        'at': datetime.now(timezone.utc).isoformat(), 'cases_rescored': rescored,
        'dataset_path': str(dataset_path), 'dataset_sha256': sha256(dataset_path.read_bytes()).hexdigest(),
        'dataset_sha256_at_run': manifest['dataset']['sha256'], 'dataset_cases': len(dataset.cases),
        **({'reviewed_labels': overlay} if overlay is not None else {})}, ensure_ascii=False, indent=2), encoding='utf-8')
    return score_partial(run_dir, results_name='results.rescored.json', dataset_path=dataset_path, dataset=dataset, overlay=overlay)


def score_partial(run_dir: Path, results_name='results.json', dataset_path=None, dataset=None, overlay=None):
    """Score a run directory from results.json and ai-calls.jsonl (for a run stopped before it finished).

    ``dataset`` is the already loaded (possibly label-overlaid) dataset; ``overlay`` the record of
    that overlay, which the metrics and the report then name.
    """
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    dataset = dataset or load_dataset(resolve_dataset(run_dir, manifest, dataset_path)[0])
    results = json.loads((run_dir / results_name).read_text(encoding='utf-8'))
    calls = read_calls(run_dir)
    for result in results:
        result['calls'] = calls.get(result['case_id'], [])
    by_id = {r['case_id']: r for r in results}
    selected = [c for c in dataset.cases if c.case_id in by_id]
    # In the dataset's order, one result per case (B6): aggregate pairs them by position.
    return finalize(run_dir, dataset, selected, [by_id[c.case_id] for c in selected], manifest, overlay=overlay)


def read_calls(run_dir) -> dict:
    """{case_id: [model call records]} from a run's ai-calls.jsonl ({} without one)."""
    calls, path = {}, Path(run_dir) / 'ai-calls.jsonl'
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                call = json.loads(line)
                calls.setdefault(call.get('case_id'), []).append(call)
    return calls


def resolve_dataset(run_dir, manifest, dataset_path=None):
    """(dataset file, how it was found) for scoring a run: the argument, the path the manifest records
    (top level since v0.19; the mode key only ever existed in memory), else <evaluation>/datasets/<id>.json
    beside the runs folder."""
    if dataset_path:
        return Path(dataset_path), 'argument'
    recorded = manifest.get('dataset_path') or (manifest.get('mode') or {}).get('dataset_path')
    if recorded and Path(recorded).is_file():
        return Path(recorded), 'manifest'
    name = f'{(manifest.get("dataset") or {}).get("id") or "dataset"}.json'
    for candidate in (Path(run_dir).parent.parent / 'datasets' / name, Path(run_dir).parent / 'datasets' / name):
        if candidate.is_file():
            return candidate, 'manifest dataset id (evaluation/datasets)'
    raise FileNotFoundError(f'No dataset for {run_dir}: pass --dataset (the manifest records no path and {name} was not found)')


def completed_cases(run_dir) -> list:
    """The case ids a run finished without a case error, in run order (results.rescored.json laid over results.json)."""
    from .taxonomy import load_results
    results, _ = load_results(Path(run_dir))
    return [r['case_id'] for r in results if not r.get('error')]


def attach_row_costs(run_dir, results):
    """Copy each packet row's elapsed_ms / timings_ms onto its prediction when results.json lacks them.

    Runs scored before v0.19 have the row costs only in their packets; reading them here lets a
    subset score report seconds per obligation without re-scoring anything. In memory only.
    """
    for result in results:
        predictions = result.get('predictions') or []
        if not predictions or all('elapsed_ms' in p for p in predictions if isinstance(p, dict)):
            continue
        path = Path(run_dir) / 'packets' / f'{result["case_id"]}.json'
        if not path.is_file():
            continue
        rows = {row.get('id'): row for row in json.loads(path.read_text(encoding='utf-8'))['events'][0]['payload'].get('obligations') or []}
        result['predictions'] = [{**p, 'elapsed_ms': (rows.get(p.get('obligation_id')) or {}).get('elapsed_ms'),
                                  'timings_ms': (rows.get(p.get('obligation_id')) or {}).get('timings_ms')} if isinstance(p, dict) else p
                                 for p in predictions]
    return results


def score_subset(run_dir, case_ids=None, dataset_path=None, reviewed_labels=None, rematch=False):
    """(manifest, metrics) of some cases of a run, scored in memory; nothing is written, least of all into the run.

    `report`, `rescore` and `score_partial` rewrite metrics.json and report.md in the run folder, and
    the v0.18 final run's files are cited by PHASE18.md. This reads results.json (with
    results.rescored.json laid over it, as the run's own metrics were), attaches each case's calls from
    ai-calls.jsonl and aggregates the chosen cases in dataset order. Scoring all cases of a finished
    run reproduces its metrics.json (verified on 20260924-121921-v018-final-ollama); two runs scored on
    the same case ids compare like for like whatever else each of them ran. ``reviewed_labels`` lays the
    ACCEPTED entries of a reviewed-labels file over the dataset in memory and re-pairs the packets.
    ``rematch`` (v0.19, FIX 4) re-pairs every packet with the current prediction rows and matcher, so runs
    scored by different matchers (v0.18 and v0.19) are compared by the same one; nothing is written either.
    """
    from .taxonomy import load_results
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    dataset_file, found_by = resolve_dataset(run_dir, manifest, dataset_path)
    dataset = load_dataset(dataset_file)
    overlay = None
    if reviewed_labels:
        from .labels import apply_reviewed
        dataset, overlay = apply_reviewed(dataset, Path(reviewed_labels), dataset_file)
    results, sources = load_results(run_dir)
    by_id = {r['case_id']: r for r in results}
    wanted = list(dict.fromkeys(case_ids or by_id))
    selected = [c for c in dataset.cases if c.case_id in by_id and c.case_id in set(wanted)]
    if not selected:
        raise ValueError(f'None of the cases {wanted[:8]} is both in {dataset_file.name} and in the results of {run_dir.name}')
    calls = read_calls(run_dir)
    chosen = []
    for case in selected:
        result = {**by_id[case.case_id], 'calls': calls.get(case.case_id, [])}
        packet = run_dir / 'packets' / f'{case.case_id}.json'
        if (overlay is not None or rematch) and not result.get('error') and packet.is_file():
            payload = json.loads(packet.read_text(encoding='utf-8'))['events'][0]['payload']
            predictions = prediction_rows(payload, case.regulation_id)
            scored, extras = score_case(case, payload, predictions)
            result.update({'predictions': predictions, 'scored': scored,
                           'extras': [{'key': e['key'], 'article': e['article'], 'clause': e['clause'], 'action': e['action'][:120]} for e in extras]})
        chosen.append(result)
    attach_row_costs(run_dir, chosen)
    metrics = aggregate(dataset.model_copy(update={'cases': selected}), chosen)
    metrics['usage'] = usage_summary([[c for r in chosen for c in r['calls']]])
    metrics['completed_cases'] = len(chosen)
    metrics['subset'] = {'run': str(run_dir), 'dataset_path': str(dataset_file), 'dataset_found_by': found_by,
                         'cases': [c.case_id for c in selected], 'not_scored': [c for c in wanted if c not in {s.case_id for s in selected}],
                         'results_from': sorted({sources[c.case_id] for c in selected}), 'rematched': bool(rematch)}
    if overlay is not None:
        metrics['reviewed_labels'] = overlay_record(overlay)
    return manifest, metrics


def with_call_costs(run_dir, metrics):
    """A copy of a run's metrics with the v0.19 call costs filled in from its ai-calls.jsonl when absent.

    A metrics.json written before v0.19 has no thinking-call counts; the call records do (each one
    carries its thinking flag), so a v0.18 run's thinking calls per obligation are counted from them.
    Only the calls of the cases the metrics cover are read (metrics['subset'] when scored as a subset).
    The runtime profile by model and window (metrics.runtime, v0.19 addendum) is filled in the same way;
    for records written before v0.19 its throughput and GPU share stay empty (shown as a dash).
    """
    metrics = json.loads(json.dumps(metrics))
    perf = metrics.setdefault('performance', {})
    costs = perf.get('thinking_calls') is None or 'failures_by_code' not in (metrics.get('reliability') or {})
    if not costs and 'runtime' in perf:
        return metrics
    calls_by_case = read_calls(run_dir)
    cases = (metrics.get('subset') or {}).get('cases')
    calls = [c for case_id, items in calls_by_case.items() if cases is None or case_id in cases for c in items]
    if not calls and perf.get('llm_calls'):
        return metrics                          # no call records to count from: leave the metrics as they are
    if costs:
        for key, value in m.call_costs(calls, perf.get('obligations')).items():
            perf.setdefault(key, value)
        metrics.setdefault('reliability', {}).setdefault('failures_by_code', m.failures_by_code(calls))
    perf.setdefault('runtime', m.runtime(calls))
    return metrics
