"""Coverage and conflict error analysis (v0.19): every wrong coverage or conflict row of a run, with its evidence chain.

The error taxonomy (taxonomy.py) names a coverage error by what the scores show; this module lays
out what the pipeline read and decided for each such row, so a reviewer can see where the chain
broke: the regulation clause the duty was cut from, the extracted duty, every policy passage the
judge read (its retrieval signals, the relevance screen, the "contradicts?" answer, the second
reading and the support answer), the counted coverage, and where the label's expected evidence
text sits in the packet. Nothing is re-run and no dataset or label file is written.

Rows are re-paired with the dataset exactly as the harness scores them (harness.prediction_rows,
harness.score_case, harness.pair_case). A row is an error when its expected coverage is
COVERS_TEXT, PARTIAL, CONFLICT or NO_EVIDENCE and the predicted coverage differs (the rows behind
metrics.coverage), or when its conflict flag is a false positive or a false negative (the rows
behind metrics.conflict). Expectations no extracted duty matched are not scored by the harness;
those with an asserted coverage are listed apart ("unscored"), each with its missing reason
(identifiers.MISSING_REASONS: EXTRACTION_MISSED, GROUNDING_REJECTED, MATCHING_FAILED, and a detail code).

Automatic categories: every rule below that holds is recorded; the first is the primary category,
the others are secondary.

 1. LABEL_AMBIGUITY                  the label names expected evidence text that occurs in no policy
                                     passage of the packet.
 2. CONTEXT_TRUNCATION               a passage could not be judged because an answer or prompt did not
                                     fit (a ProviderFailure whose model call ended "truncated" or
                                     done_reason=length, ContextBudgetError, OUTPUT_TRUNCATED), and
                                     that passage holds the expected evidence or left the row UNKNOWN.
 3. PREFILTER_FALSE_NEGATIVE         an expected-evidence passage was set aside by the relevance screen
                                     (SCREENED_OUT) or the evidence gate (filtered) and none was judged.
 4. RETRIEVAL_MISS                   expected-evidence passages exist but none reached the judge.
 5. expected CONFLICT (or a conflict false negative):
    VERIFIER_OVERRULE_ERROR          an expected-evidence passage (any passage when the label names
                                     none) was answered contradicts=YES and then withdrawn
                                     (CONFLICT_WITHDRAWN, by the second reading or by rule); v0.19
                                     pipeline: the passage was escalated and the thinking verifier
                                     answered conflict=false (VERIFIER_NO_CONFLICT), or its conflict
                                     claim was withdrawn by rule (no exact span, restated prohibition);
    PREFILTER_FALSE_NEGATIVE         v0.19 pipeline: the fast classifier called that passage IRRELEVANT
                                     and it was never escalated, so the verifier never read it;
    CONFLICT_MISSED                  otherwise.
 6. predicted CONFLICT, not expected (a conflict false positive):
    OBLIGATION_EXTRACTION_TRUNCATION the extracted duty (subject + action) is much shorter than the
                                     clause sentence it was cut from (duty_truncation);
    FALSE_CONFLICT                   always (secondary when the truncation rule holds).
 7. COVERS_AS_PARTIAL / PARTIAL_AS_COVERS  expected COVERS_TEXT predicted PARTIAL, and the reverse.
 8. SEMANTIC_MATCH_MISSED            expected COVERS_TEXT or PARTIAL, predicted otherwise (NO_EVIDENCE,
                                     or any state when the label names its evidence), and an
                                     expected-evidence passage was judged in full and called UNRELATED
                                     (v0.19: by the fast classifier's IRRELEVANT without escalation, or
                                     by the verifier).
 9. IRRELEVANT_PASSAGE_SELECTED      expected NO_EVIDENCE and a passage was judged favourable; or (as a
                                     secondary) the predicted coverage rests only on passages without
                                     the expected evidence.
10. AGGREGATION_ERROR                a passage carries the relation the label needs (SUPPORTS for
                                     COVERS_TEXT, PARTIAL, CONFLICTS) but the count (engine.coverage_of)
                                     gave another state: UNCLEAR passages outweighed it, or it was a
                                     control register row counted apart from written coverage.
11. OBLIGATION_EXTRACTION_TRUNCATION as a secondary on any other row whose extracted duty is truncated.
12. LABEL_AMBIGUITY                  as a secondary when the coverage label is disputed: in
                                     taxonomy.DISPUTED_LABELS, as a DISPUTED reason of the label-review
                                     file, or by the wording of the label's notes. An ALL_RUNS_DISAGREE
                                     reason is recorded but does not flag the label: agreement between
                                     runs of the same models is not evidence against the label.
13. OTHER                            when nothing else holds.

A manual annotations file (evaluation/reports/coverage-error-annotations-<run>.json), keyed by
'<case_id> md.<article>(<clause>)', overrides the category and the root cause of a row; the report
keeps the automatic reading beside it.

v0.19 packets: a passage result of the v0.19 coverage pipeline carries 'pipeline': 'v19', the fast
classifier's answer ('fast'), the escalation reason ('escalation', None when the fast answer stood)
and the thinking verifier's answer ('verifier', None when it was not asked). Its first judgement is
then the verifier's conflict answer (NOT_ASKED without escalation) and its support answer comes from
the verifier when it answered, else from the fast classifier. v0.18 packets read as before.
"""
import json
import re
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from .identifiers import clause_id, normalize
from .taxonomy import LEGACY_DATASET, disputed

FORMAT = 'cardaman-coverage-errors-v1'
ANNOTATIONS_FORMAT = 'cardaman-coverage-error-annotations-v1'
CATEGORIES = ('OBLIGATION_EXTRACTION_TRUNCATION', 'RETRIEVAL_MISS', 'SEMANTIC_MATCH_MISSED', 'FALSE_CONFLICT', 'CONFLICT_MISSED',
              'VERIFIER_OVERRULE_ERROR', 'PARTIAL_AS_COVERS', 'COVERS_AS_PARTIAL', 'IRRELEVANT_PASSAGE_SELECTED',
              'PREFILTER_FALSE_NEGATIVE', 'CONTEXT_TRUNCATION', 'AGGREGATION_ERROR', 'LABEL_AMBIGUITY', 'OTHER')
ASSESSED = ('COVERS_TEXT', 'PARTIAL', 'CONFLICT', 'NO_EVIDENCE')
# The passage relation a coverage label needs at least one passage to carry (engine.coverage_of).
NEEDED_RELATION = {'COVERS_TEXT': 'SUPPORTS', 'PARTIAL': 'PARTIAL', 'CONFLICT': 'CONFLICTS'}
FAVOURABLE = ('SUPPORTS', 'PARTIAL')
FAVOURABLE_COVERAGE = ('COVERS_TEXT', 'PARTIAL')
TRUNCATION_CODES = {'ContextBudgetError', 'OUTPUT_TRUNCATED', 'PROMPT_CUT_BY_RUNTIME', 'CONTEXT_OVERFLOW', 'PROMPT_CUT'}
TRUNCATION_ERROR = re.compile(r'truncated|cut to fit|context window', re.I)
VERIFIER_CODES = ('CONFLICT_WITHDRAWN', 'CONFLICT_UNCONFIRMED')
# The verifier answers that take a contradiction back: v0.18's second reading or rule, v0.19's verifier saying conflict=false.
OVERRULED = ('CONFLICT_WITHDRAWN', 'VERIFIER_NO_CONFLICT')
V19 = 'v19'
CONFIRMED = re.compile(r'\[Confirmed on a second reading:(.*)\]\s*$', re.S)
# A duty (subject + action) is truncated when it has at most SHORT_DUTY_WORDS words and under SHORT_DUTY_RATIO of
# its clause sentence, or whatever its length under CUT_DUTY_RATIO (a long enumerated sentence reduced to its verb).
SHORT_DUTY_WORDS = 4
SHORT_DUTY_RATIO = 0.35
CUT_DUTY_RATIO = 0.2
AMBIGUOUS_NOTE = re.compile(r'tartışmalı|disputed|savunulabilir|defensible|etiket belirsiz|ambiguous label', re.I)
# A sentence ends at a full stop before a capital or a bracket ("T.C. kimlik" and "K.) Gerçek" do not split);
# semicolons separate list items inside one legal sentence and do not end it.
SENTENCE_END = re.compile(r'(?<=\.)\s+(?=[A-ZÇĞİÖŞÜ(])')
CLAUSE_MARKER = re.compile(r'^\s*(?:\(\d{1,2}\)\s*)?(?:\((?:Ek|Değişik|Mülga|Yeniden düzenleme)[^()]*\)\s*)?')
HEX_ID = re.compile(r'[0-9a-f]{64}')


def squash(text) -> str:
    """Whitespace-insensitive text, as harness.evidence_passages compares evidence substrings."""
    return ' '.join(str(text or '').split())


def clip(text, size=400) -> str:
    text = squash(text)
    return text if len(text) <= size else text[:size - 1] + '…'


def row_key(case_id, article, clause) -> str:
    """'C13 md.5(2)': the key of a row in the report and in the annotations file."""
    number = clause_id(clause or '')
    return f'{case_id} md.{article}' + (f'({number})' if number else '')


def error_kinds(item) -> list:
    """The ways a scored pair (harness.score_case item) is a coverage or conflict error; [] when it is not."""
    kinds = []
    coverage, conflict = item.get('coverage'), item.get('conflict')
    if coverage and coverage[0] in ASSESSED and coverage[0] != coverage[1]:
        kinds.append('coverage_mismatch')
    if conflict and conflict[0] != conflict[1]:
        kinds.append('conflict_false_positive' if conflict[1] else 'conflict_false_negative')
    return kinds


# ---------------------------------------------------------------- reading one packet row

def passage_index(payload) -> dict:
    """{source_id: chunk} over every policy document and control register of the packet."""
    return {chunk['source_id']: {**chunk, 'filename': chunk.get('filename') or policy.get('filename')}
            for policy in payload.get('policies') or [] for chunk in policy.get('chunks') or []}


def clause_text(payload, obligation) -> str:
    """The exact unit (numbered sub-paragraph, or the whole provision) the duty was extracted from."""
    scope = (obligation.get('proposal') or {}).get('applicability_scope') or {}
    offset = scope.get('child_offset')
    if offset is None:
        offset = (obligation.get('multipart') or {}).get('unit_offset')
    for case in payload.get('cases') or []:
        source = case.get('source') or {}
        if source.get('printed_label') != obligation.get('source_label'):
            continue
        text = source.get('text') or ''
        for unit in case.get('units') or []:
            if offset is not None and unit.get('offset') == offset and unit.get('chars'):
                return text[offset:offset + unit['chars']].strip()
        if offset in (None, 0) and len(case.get('units') or []) <= 1:
            return text.strip()
    candidate = obligation.get('candidate') or {}
    return scope.get('child_excerpt') or candidate.get('source_quote') or ''


def duty_view(candidate) -> dict:
    return {key: candidate.get(key) for key in ('subject', 'modality', 'required_action', 'prohibited_action', 'conditions', 'exceptions',
                                                'deadline', 'source_quote')}


def words(text) -> list:
    return normalize(text).split()


def duty_truncation(candidate, clause) -> dict:
    """How much of its clause sentence the extracted duty kept.

    The sentence is the one of the clause that shares most words with the action (ties: the first);
    the duty is subject + action. ``truncated`` when the duty has at most SHORT_DUTY_WORDS words and
    under SHORT_DUTY_RATIO of the sentence's words (e.g. the prohibition "uygulayamazlar" cut from a
    sentence that names what may not be applied, and when), or under CUT_DUTY_RATIO of it (e.g. "uygun
    ve etkili tedbirleri almak" left of a sentence that lists the measures and where they apply).
    """
    action = candidate.get('required_action') or candidate.get('prohibited_action') or ''
    action_words = words(action)
    duty = words(' '.join([candidate.get('subject') or '', action]))
    sentences = [s for s in SENTENCE_END.split(CLAUSE_MARKER.sub('', clause or '')) if s.strip()] or [clause or '']
    wanted = set(action_words)
    sentence = max(sentences, key=lambda s: len(wanted & set(words(s)))) if wanted else sentences[0]
    sentence_words = words(sentence)
    ratio = round(len(duty) / len(sentence_words), 3) if sentence_words else None
    return {'action_words': len(action_words), 'duty_words': len(duty), 'sentence_words': len(sentence_words), 'ratio': ratio,
            'sentence': clip(sentence, 600),
            'truncated': bool(action_words) and ratio is not None and (ratio < CUT_DUTY_RATIO or (len(duty) <= SHORT_DUTY_WORDS and ratio < SHORT_DUTY_RATIO))}


def passage_stage(obligation) -> dict:
    """The diagnostics entry {'stage': 'passages', 'results': [...], 'filtered': [...]} of a packet row."""
    for item in obligation.get('diagnostics') or []:
        if isinstance(item, dict) and item.get('stage') == 'passages':
            return item
    return {'results': [], 'filtered': []}


def first_judgement(result, check_quote='', calls=()) -> dict:
    """The first reading of one passage: the "contradicts?" answer and the support answer.

    engine.judge_passage records the contradicts answer as ``screen`` (YES, NO, UNCLEAR, FAILED, or
    WITHDRAWN for a YES the rule or the second reading took back); SCREENED_OUT means the relevance
    screen set the passage aside before any question. The support answer is the final relation
    unless the passage stands as CONFLICTS or could not be judged. ``support_by`` says who gave it:
    the unreasoned quick model alone (its UNRELATED is final), or the reasoning judge as well (asked
    only after the quick model nominated SUPPORTS or PARTIAL). A v0.19 result is read by v19_judgement.
    """
    if result.get('pipeline') == V19:
        return v19_judgement(result, check_quote)
    screen = result.get('screen')
    notes = result.get('notes') or []
    contradicts = {'YES': 'YES', 'WITHDRAWN': 'YES', 'NO': 'NO', 'UNCLEAR': 'UNCLEAR', 'FAILED': 'FAILED',
                   'SCREENED_OUT': 'NOT_ASKED'}.get(screen, screen)
    withdrawn = [n for n in notes if n.get('code') == 'CONFLICT_WITHDRAWN']
    quote = withdrawn[0].get('quote') if withdrawn else (check_quote if result.get('relation') == 'CONFLICTS' else '')
    if screen in ('SCREENED_OUT', 'FAILED') or result.get('relation') == 'CONFLICTS':
        support = 'NOT_ASKED'
    elif result.get('unjudged'):
        support = 'FAILED'
    else:
        support = result.get('relation')
    asked = [c for c in calls if c.get('task') == 'judge.supports']
    support_by = (None if not asked else 'reasoning' if any(str(c.get('thinking')) == 'True' for c in asked) else 'quick')
    return {'contradicts': contradicts, 'contradicts_quote': quote or '', 'support': support, 'support_by': support_by}


def v19_judgement(result, check_quote='') -> dict:
    """The v0.19 reading of one passage: the fast classifier, the escalation and the verifier's conflict answer.

    ``contradicts`` is what the thinking verifier said (YES / NO / FAILED), NOT_ASKED when the passage was
    not escalated or was screened out; ``support_by`` names who gave the final relation: 'verifier' when
    it answered, else 'fast' (the no-think classifier's answer stood).
    """
    fast = result.get('fast') if isinstance(result.get('fast'), dict) else {}
    verifier = result.get('verifier') if isinstance(result.get('verifier'), dict) else None
    failed = any(n.get('code') == 'VERIFIER_FAILED' for n in result.get('notes') or [])
    if result.get('screen') == 'SCREENED_OUT' or verifier is None and not failed:
        contradicts = 'NOT_ASKED'
    elif failed or verifier.get('conflict') is None:
        contradicts = 'FAILED'
    else:
        contradicts = 'YES' if verifier.get('conflict') else 'NO'
    quote = (verifier or {}).get('contradiction_span') or (check_quote if result.get('relation') == 'CONFLICTS' else '')
    if result.get('screen') == 'SCREENED_OUT' or result.get('relation') == 'CONFLICTS':
        support = 'NOT_ASKED'
    elif result.get('unjudged') or failed:
        support = 'FAILED'
    else:
        support = result.get('relation')
    return {'contradicts': contradicts, 'contradicts_quote': str(quote or ''), 'support': support,
            'support_by': 'verifier' if verifier is not None and contradicts in ('YES', 'NO') else 'fast' if fast else None,
            'fast': fast.get('label'), 'escalation': result.get('escalation')}


def verifier_view(result) -> dict | None:
    """The second reading of an alleged contradiction (or the rule that withdrew it), or None when there was none."""
    if result.get('pipeline') == V19:
        return v19_verifier_view(result)
    for note in result.get('notes') or []:
        if note.get('code') in VERIFIER_CODES:
            detail = str(note.get('detail') or '')
            return {'code': note['code'], 'by': 'rule' if detail.startswith('Rule:') else 'model', 'detail': detail,
                    'quote': note.get('quote') or ''}
    confirmed = CONFIRMED.search(str(result.get('reason') or ''))
    if result.get('relation') == 'CONFLICTS':
        return {'code': 'CONFLICT_CONFIRMED', 'by': 'model', 'detail': squash(confirmed.group(1)) if confirmed else '', 'quote': ''}
    return None


def v19_verifier_view(result) -> dict | None:
    """What the v0.19 thinking verifier (or a code rule on its answer) decided about a passage; None when it was not asked.

    The verifier is the only reading of an escalated passage (no second "confirm" call in v0.19), so its
    conflict=false is the overrule a missed conflict is traced back to.
    """
    notes = result.get('notes') or []
    for note in notes:
        if note.get('code') == 'VERIFIER_FAILED':
            code = note.get('failure_code') or note.get('detail') or ''
            return {'code': 'VERIFIER_FAILED', 'by': 'model', 'detail': str(code), 'quote': ''}
    verifier = result.get('verifier') if isinstance(result.get('verifier'), dict) else None
    for note in notes:
        if note.get('code') in VERIFIER_CODES or note.get('code') == 'CONFLICT_CLAIM_WITHOUT_SPAN':
            detail = str(note.get('detail') or note['code'])
            return {'code': 'CONFLICT_WITHDRAWN' if note['code'] == 'CONFLICT_CLAIM_WITHOUT_SPAN' else note['code'],
                    'by': 'rule' if note['code'] == 'CONFLICT_CLAIM_WITHOUT_SPAN' or detail.startswith('Rule:') else 'model',
                    'detail': detail if detail.startswith(('Rule:', note['code'])) else f'{note["code"]}: {detail}',
                    'quote': note.get('quote') or (verifier or {}).get('contradiction_span') or ''}
    if verifier is None:
        return None
    detail = ' — '.join(str(x) for x in (verifier.get('contradiction_type'), verifier.get('confidence'), verifier.get('rationale')) if x)
    if result.get('relation') == 'CONFLICTS':
        return {'code': 'CONFLICT_CONFIRMED', 'by': 'model', 'detail': detail, 'quote': verifier.get('contradiction_span') or ''}
    if verifier.get('conflict') is False:
        return {'code': 'VERIFIER_NO_CONFLICT', 'by': 'model',
                'detail': f'conflict=false, relation {verifier.get("relation_if_no_conflict")}' + (f': {verifier.get("rationale")}' if verifier.get('rationale') else ''),
                'quote': verifier.get('support_quote') or ''}
    if verifier.get('conflict') is True:
        return {'code': 'CONFLICT_WITHDRAWN', 'by': 'rule', 'detail': 'Rule: the conflict claim did not stand (' + detail + ')',
                'quote': verifier.get('contradiction_span') or ''}
    return None


def recovered(result, note) -> bool:
    """Whether the question a failure note belongs to got its answer in the end.

    v0.19 (engine.model_answer): an OUTPUT_TRUNCATED or TIMEOUT answer is asked once more uncached, a
    CONTEXT_OVERFLOW once more trimmed, and a failed fast reading goes to the verifier (escalation FAST_FAILED);
    the attempt notes stay in the record. The fast question is answered when the fast reading or the verifier
    answered, the verify question when the verifier did. v0.18 records only a trimmed retry ('retry' key) that
    way; it counts as recovered unless the passage was left unjudged.
    """
    if result.get('pipeline') == V19:
        verifier = isinstance(result.get('verifier'), dict)
        if note.get('question') == 'verify':
            return verifier
        if note.get('question') == 'fast':
            return verifier or isinstance(result.get('fast'), dict)
        return False
    return bool(note.get('retry')) and not result.get('unjudged')


def truncation_signal(result, calls) -> str | None:
    """Why a passage's judgement was cut off by a size limit, or None.

    Only a question that ended without an answer counts (v0.19 review): a verifier that answered after its
    truncated first attempt, a fast reading answered after trimming, or a failed fast reading the verifier
    answered in its place left the passage judged, and calling that CONTEXT_TRUNCATION (rule 2, before the
    overrule) blamed the context window for the model's own answer. At 8k the thinking budget is 4,096, so
    first-attempt truncations are expected; only an unrecovered one is a cut.
    """
    notes = [n for n in result.get('notes') or [] if isinstance(n, dict) and not recovered(result, n)]
    for note in notes:
        if note.get('code') in TRUNCATION_CODES:
            return f'{note.get("question")}: {note["code"]}'
        # v0.19: a failed verifier names the provider's failure code (VERIFIER_FAILED + OUTPUT_TRUNCATED ...).
        if note.get('failure_code') in TRUNCATION_CODES or note.get('code') == 'VERIFIER_FAILED' and any(
                code in str(note.get('detail') or '') for code in TRUNCATION_CODES):
            return f'{note.get("question") or "verifier"}: {note["code"]} {note.get("failure_code") or note.get("detail") or ""}'.strip()
    failed = [n for n in notes if n.get('code') in ('ProviderFailure', 'VERIFIER_FAILED', 'FAST_FAILED')]
    if any(n.get('failure_code') for n in failed):
        return None                             # v0.19 names why the question failed, and it was not a size limit
    for call in calls:
        if (call.get('done_reason') == 'length' or call.get('failure_code') in TRUNCATION_CODES
                or (call.get('status') != 'OK' and TRUNCATION_ERROR.search(str(call.get('error') or '')))):
            if failed or result.get('unjudged'):
                return f'{call.get("task")}: {call.get("error") or call.get("failure_code") or "done_reason=length"} ({call.get("elapsed_ms")} ms)'
    return None


def call_view(call) -> dict:
    return {key: call.get(key) for key in ('task', 'status', 'done_reason', 'elapsed_ms', 'cache_hit', 'thinking', 'output_tokens', 'error')
            if call.get(key) not in (None, '', 'None')}


def calls_by_passage(calls) -> dict:
    """{source_id: [model calls about that passage]} of one obligation (ai-calls.jsonl records), in time order."""
    out = {}
    for call in sorted(calls, key=lambda c: str(c.get('at') or '')):
        for sid in HEX_ID.findall(str(call.get('evidence_ids') or '')):
            out.setdefault(sid, []).append(call)
    return out


def contains(text, needles) -> list:
    body = squash(text)
    return [needle for needle in needles if squash(needle) and squash(needle) in body]


def passage_rows(payload, obligation, needles, calls=()) -> list:
    """One record per passage the judge read, the screen set aside, or the evidence gate kept from it (by rank)."""
    chunks = passage_index(payload)
    stage = passage_stage(obligation)
    signals = {s['source_id']: s for s in obligation.get('evidence_signals') or [] if s.get('source_id')}
    quotes = {c['source_id']: c.get('quote') for c in (obligation.get('proposal') or {}).get('policy_checks') or []}
    cited = {q['source_id'] for q in (obligation.get('proposal') or {}).get('policy_evidence') or [] if isinstance(q, dict)}
    by_passage = calls_by_passage(calls)
    rows = []
    for result in stage.get('results') or []:
        sid = result['source_id']
        rows.append({'source_id': sid, 'status': 'SCREENED_OUT' if result.get('screen') == 'SCREENED_OUT' else 'JUDGED', 'result': result})
    seen = {row['source_id'] for row in rows}
    filtered = [*(stage.get('filtered') or []), *((obligation.get('proposal') or {}).get('filtered_passages') or [])]
    for item in filtered:
        if item.get('source_id') and item['source_id'] not in seen:
            seen.add(item['source_id'])
            rows.append({'source_id': item['source_id'], 'status': 'FILTERED', 'result': {'reason': item.get('reason')}})
    out = []
    for row in rows:
        sid, result = row['source_id'], row['result']
        chunk, signal = chunks.get(sid, {}), signals.get(sid, {})
        passage_calls = by_passage.get(sid, [])
        judged = row['status'] == 'JUDGED'
        out.append({'source_id': sid, 'filename': chunk.get('filename'), 'locator': chunk.get('locator'), 'text': chunk.get('text') or '',
                    'status': row['status'], 'relation': result.get('relation'), 'screen': result.get('screen'),
                    'reason': result.get('reason'), 'notes': result.get('notes') or [], 'unjudged': bool(result.get('unjudged')),
                    'control_row': bool(result.get('control_row')) or chunk.get('locator') == 'control_row', 'cited': sid in cited,
                    'first_judgement': first_judgement(result, quotes.get(sid) or '', passage_calls) if row['status'] != 'FILTERED' else None,
                    'verifier': verifier_view(result) if judged else None,
                    'truncation': truncation_signal(result, passage_calls) if judged else None,
                    'retrieval': {key: signal.get(key) for key in ('rank', 'similarity', 'lexical_score', 'rrf', 'rerank_score', 'semantic_rank',
                                                                   'lexical_rank', 'gate')},
                    'expected_evidence': contains(chunk.get('text'), needles),
                    'calls': [call_view(c) for c in passage_calls],
                    **(v19_view(result) if result.get('pipeline') == V19 else {})})
    rank = lambda p: p['retrieval'].get('rank') if isinstance(p['retrieval'].get('rank'), int) else 10**6
    return sorted(out, key=rank)


def v19_view(result) -> dict:
    """The v0.19 fields of a passage result a reviewer needs beside the relation (signals kept compact).

    covered_elements / missing_elements / quantity_match are the passage record's own, the values the
    aggregation (engine.coverage_of_v19) counted: after a verifier answer they replace the fast reading's
    lists (smoke run C12 md.5(2): fast missing ['action', 'deadline_1'], the verifier's SUPPORTS missing []).
    A record written before they existed falls back to the fast reading's lists.
    """
    fast = result.get('fast') if isinstance(result.get('fast'), dict) else {}
    listed = lambda key, fallback: list(result[key]) if isinstance(result.get(key), list) else list(fast.get(fallback) or [])
    return {'pipeline': V19, 'fast': {key: fast.get(key) for key in ('label', 'quote', 'covered', 'missing', 'reason') if key in fast},
            'escalation': result.get('escalation'), 'topic_overlap': result.get('topic_overlap'), 'uncertainty': result.get('uncertainty'),
            'covered_elements': listed('covered_elements', 'covered'), 'missing_elements': listed('missing_elements', 'missing'),
            'quantity_match': dict(result.get('quantity_match') or {}),
            'structural_signals': [f'{s.get("type")}/{s.get("strength")}' for s in result.get('signals') or [] if isinstance(s, dict)]}


# quantity_match verdicts (pilot.conflict.quantity_match) that meet a duty quantity; any other leaves the row PARTIAL.
QUANTITY_MET = ('SAME', 'STRICTER')


def partial_basis(passage, judged) -> str:
    """Why a SUPPORTS passage was counted PARTIAL (v0.19 aggregation): its missing elements, and every duty quantity no
    favourable policy passage states the same or stricter, with this passage's verdict on it."""
    missing = passage.get('missing_elements')
    missing = (passage.get('fast') or {}).get('missing') or [] if missing is None else missing
    favourable = [p for p in judged if p.get('relation') in FAVOURABLE and not p.get('control_row')]
    quantities = sorted({q for p in favourable for q in (p.get('quantity_match') or {})})
    unmet = [q for q in quantities if not any((p.get('quantity_match') or {}).get(q) in QUANTITY_MET for p in favourable)]
    parts = ([f'missing elements: {", ".join(map(str, missing))}'] if missing else []) + \
            ([f'quantities not stated or weaker: {", ".join(f"{q} {(passage.get("quantity_match") or {}).get(q) or "NOT_STATED"}" for q in unmet)}']
             if unmet else [])
    return f' ({"; ".join(parts)})' if parts else ''


def evidence_view(payload, obligation, needles, passages) -> dict:
    """Where each expected evidence substring sits: in which passages, and what became of them."""
    chunks = passage_index(payload)
    retrieved = set(obligation.get('retrieved_policy_ids') or []) | set(obligation.get('judged_policy_ids') or [])
    read = {p['source_id']: p for p in passages}
    items = []
    for needle in needles:
        found = []
        for sid, chunk in chunks.items():
            if not contains(chunk.get('text'), [needle]):
                continue
            passage = read.get(sid)
            if passage is None:
                status = 'RETRIEVED_NOT_JUDGED' if sid in retrieved else 'NOT_RETRIEVED'
            elif passage['status'] == 'JUDGED':
                status = 'JUDGED:' + str(passage['relation'])
            else:
                status = passage['status']
            found.append({'source_id': sid, 'filename': chunk.get('filename'), 'status': status,
                          'rank': (passage or {}).get('retrieval', {}).get('rank')})
        items.append({'text': needle, 'in_packet': bool(found), 'passages': found})
    statuses = [p['status'] for item in items for p in item['passages']]
    return {'needles': items, 'any_in_packet': any(i['in_packet'] for i in items),
            'judged_contains': any(s.startswith('JUDGED:') for s in statuses),
            'screened_out': 'SCREENED_OUT' in statuses, 'filtered': 'FILTERED' in statuses,
            'cited': any(p['cited'] and p['expected_evidence'] for p in passages)}


# ---------------------------------------------------------------- label signals

def load_label_review(path) -> dict:
    """{'C12 md.5(2)': [reasons]} from a label-audit file (labels.write_audit); {} without one."""
    if not path or not Path(path).is_file():
        return {}
    record = json.loads(Path(path).read_text(encoding='utf-8'))
    out = {}
    for item in record.get('items') or []:
        where = item.get('where') or ''
        match = re.match(r'^(\S+) md\.([^(\s]+)(?:\((\d+)\))?', where)
        key = row_key(match.group(1), match.group(2), match.group(3) or '') if match else where
        out.setdefault(key, []).extend(str(r) for r in item.get('reasons') or [])
    return out


def label_view(case_id, expected, review_reasons, dataset_id=LEGACY_DATASET) -> dict:
    """What says the coverage label itself may be wrong (a disputed label), and what only questions it.

    Disputes are looked up for the row's own dataset only (B5): an independent dataset's C12 is not tr-aml-v1's C12.
    """
    reasons = []
    dispute = disputed(case_id, expected['article'], expected.get('clause') or '', dataset_id)
    if dispute and dispute[0] == 'coverage':
        reasons.append('taxonomy.DISPUTED_LABELS: ' + dispute[1])
    for reason in review_reasons:
        text = reason.split(': ', 1)[-1]
        if reason.startswith('DISPUTED (coverage)') and not (dispute and text == dispute[1]):
            reasons.append('label review: ' + reason)
    if AMBIGUOUS_NOTE.search(expected.get('notes') or ''):
        reasons.append('dataset note: ' + clip(expected.get('notes'), 300))
    return {'disputed': bool(reasons), 'dispute_reasons': reasons,
            'review_reasons': [r for r in review_reasons if not r.startswith('DISPUTED')], 'notes': expected.get('notes') or ''}


# ---------------------------------------------------------------- categories

def category_rules(row) -> list:
    """[(category, rationale)] of every rule that holds for an error row, in rule order (module docstring)."""
    found = []
    exp, act = row['expected'], row['actual']
    exp_cov, act_cov = exp.get('coverage'), act.get('coverage')
    passages = row.get('passages') or []
    evidence = row.get('evidence') or {}
    needles = [n['text'] for n in evidence.get('needles') or []]
    holding = [p for p in passages if p.get('expected_evidence')]
    judged = [p for p in passages if p.get('status') == 'JUDGED']
    short = lambda p: f'{p["source_id"][:8]} (rank {p["retrieval"].get("rank")})'
    if needles and not evidence.get('any_in_packet'):
        found.append(('LABEL_AMBIGUITY', 'The expected evidence text ("' + '", "'.join(needles) + '") occurs in no policy passage of the packet.'))
    cut = [p for p in judged if p.get('truncation')]
    if cut and (act_cov == 'UNKNOWN' or any(p.get('expected_evidence') for p in cut)):
        p = cut[0]
        found.append(('CONTEXT_TRUNCATION', f'Passage {short(p)} could not be judged: {p["truncation"]}; the row was counted {act_cov}.'))
    held_judged = [p for p in holding if p.get('status') == 'JUDGED']
    held_aside = [p for p in holding if p.get('status') in ('SCREENED_OUT', 'FILTERED')]
    if held_aside and not held_judged:
        p = held_aside[0]
        found.append(('PREFILTER_FALSE_NEGATIVE', f'The expected-evidence passage {short(p)} was {p["status"]} and never judged in full.'))
    if needles and evidence.get('any_in_packet') and not holding:
        statuses = sorted({s['status'] for n in evidence.get('needles') or [] for s in n['passages']})
        found.append(('RETRIEVAL_MISS', f'The expected-evidence passage(s) never reached the judge ({", ".join(statuses)}).'))
    conflict_missed = exp_cov == 'CONFLICT' and act_cov != 'CONFLICT' or (exp.get('conflict') is True and act.get('conflict') is False)
    if conflict_missed:
        pool = holding if needles else passages
        withdrawn = [p for p in pool if (p.get('verifier') or {}).get('code') in OVERRULED]
        unescalated = [p for p in pool if p.get('pipeline') == V19 and p.get('status') == 'JUDGED' and not p.get('escalation')
                       and (p.get('fast') or {}).get('label') == 'IRRELEVANT']
        if withdrawn:
            p = withdrawn[0]
            v = p['verifier']
            if v['code'] == 'VERIFIER_NO_CONFLICT':
                found.append(('VERIFIER_OVERRULE_ERROR', f'Passage {short(p)} was escalated to the thinking verifier ({p.get("escalation")}; fast '
                                                         f'{(p.get("fast") or {}).get("label")}) and the verifier answered {clip(v["detail"], 240)}'))
            else:
                found.append(('VERIFIER_OVERRULE_ERROR', f'Passage {short(p)} was answered contradicts=YES ("{clip(p["first_judgement"]["contradicts_quote"], 160)}") '
                                                         f'and withdrawn by the {v["by"]}: {clip(v["detail"], 240)}'))
        elif unescalated:
            p = unescalated[0]
            found.append(('PREFILTER_FALSE_NEGATIVE', f'The fast classifier called passage {short(p)} IRRELEVANT (topic overlap {p.get("topic_overlap")}, '
                                                      f'signals {", ".join(p.get("structural_signals") or []) or "none"}) and it was not escalated: '
                                                      f'the thinking verifier never read it.'))
        else:
            answers = ', '.join(f'{short(p)} contradicts={p["first_judgement"]["contradicts"]}'
                                + (f' (fast {p["first_judgement"].get("fast")})' if p.get('pipeline') == V19 else '')
                                for p in pool if p.get('first_judgement')) or 'no passage read'
            found.append(('CONFLICT_MISSED', f'No expected-evidence passage was read as contradicting the duty ({answers}).'))
    false_conflict = act_cov == 'CONFLICT' and exp_cov != 'CONFLICT' or (exp.get('conflict') is False and act.get('conflict') is True)
    duty = row.get('duty_truncation') or {}
    if false_conflict:
        stands = [p for p in judged if p.get('relation') == 'CONFLICTS']
        where = ', '.join(f'{short(p)} "{clip(p["first_judgement"]["contradicts_quote"] or p["text"], 120)}"' for p in stands) or 'a control row'
        if duty.get('truncated'):
            found.append(('OBLIGATION_EXTRACTION_TRUNCATION', f'The extracted duty keeps {duty["duty_words"]} of the {duty["sentence_words"]} words of its '
                                                              f'clause sentence (ratio {duty["ratio"]}); the judge compared the policy with a '
                                                              f'fragment and read {where} as its contradiction.'))
        found.append(('FALSE_CONFLICT', f'A conflict the label does not have was confirmed on {where}.'))
    if exp_cov == 'COVERS_TEXT' and act_cov == 'PARTIAL':
        partial = [p for p in held_judged if p.get('relation') == 'PARTIAL'] or [p for p in judged if p.get('relation') == 'PARTIAL']
        supports = [p for p in judged if p.get('relation') == 'SUPPORTS' and not p.get('control_row')]
        if not partial and supports:
            # v0.19 aggregation: a SUPPORTS passage counts as covering only with no missing element and every numeric
            # duty quantity matched; otherwise the row is PARTIAL although no passage was judged PARTIAL. The record's
            # own element lists and quantity verdicts say which (the fast reading's lists are stale after a verifier answer).
            p = supports[0]
            found.append(('COVERS_AS_PARTIAL', f'Passage {short(p)} was judged SUPPORTS but the aggregation counted PARTIAL'
                                               + (partial_basis(p, judged) if p.get('pipeline') == V19 else '')
                                               + f': {clip((row.get("aggregation") or {}).get("coverage_reason"), 240)}'))
        else:
            why = f' Passage {short(partial[0])}: {clip(partial[0]["reason"], 240)}' if partial else ''
            found.append(('COVERS_AS_PARTIAL', 'The passage the label counts as full coverage was judged PARTIAL.' + why))
    if exp_cov == 'PARTIAL' and act_cov == 'COVERS_TEXT':
        found.append(('PARTIAL_AS_COVERS', 'A passage the label counts as partial coverage was judged SUPPORTS.'))
    if exp_cov in FAVOURABLE_COVERAGE and act_cov != exp_cov and (act_cov == 'NO_EVIDENCE' or needles):
        unrelated = [p for p in (held_judged if needles else judged) if p.get('relation') == 'UNRELATED']
        if unrelated:
            p = unrelated[0]
            by = (p.get('first_judgement') or {}).get('support_by')
            by = {'fast': 'fast classifier (IRRELEVANT, not escalated)' if not p.get('escalation') else 'fast classifier',
                  'verifier': 'thinking verifier'}.get(by, f'{by} model' if by else None)
            found.append(('SEMANTIC_MATCH_MISSED', f'The expected-evidence passage {short(p)} was judged in full and called UNRELATED'
                                                   + (f' by the {by}' if by else '') + f': {clip(p["reason"], 240)}'))
    favourable = [p for p in judged if p.get('relation') in FAVOURABLE]
    if exp_cov == 'NO_EVIDENCE' and act_cov in ('COVERS_TEXT', 'PARTIAL', 'CONFLICT') and (favourable or act_cov == 'CONFLICT'):
        chosen = favourable or [p for p in judged if p.get('relation') == 'CONFLICTS']
        found.append(('IRRELEVANT_PASSAGE_SELECTED', 'The label expects no written evidence, yet ' + ', '.join(
            f'{short(p)} was judged {p["relation"]} ("{clip(p["text"], 100)}")' for p in chosen[:3]) + '.'))
    elif needles and act_cov in ('COVERS_TEXT', 'PARTIAL', 'CONFLICT'):
        basis = [p for p in judged if p.get('relation') in (*FAVOURABLE, 'CONFLICTS')]
        if basis and not any(p.get('expected_evidence') for p in basis):
            found.append(('IRRELEVANT_PASSAGE_SELECTED', f'{act_cov} rests on passage(s) without the expected evidence: '
                                                         + ', '.join(short(p) for p in basis[:3]) + '.'))
    needed = NEEDED_RELATION.get(exp_cov)
    if needed and act_cov != exp_cov:
        policy = [p for p in judged if p.get('relation') == needed and not p.get('control_row')]
        control = [p for p in judged if p.get('relation') == needed and p.get('control_row')]
        unclear = [p for p in judged if p.get('relation') == 'UNCLEAR']
        if policy and act_cov in ('UNKNOWN', 'NO_EVIDENCE'):
            found.append(('AGGREGATION_ERROR', f'{len(policy)} passage(s) carry {needed} but {len(unclear)} UNCLEAR passage(s) made the count {act_cov}.'))
        elif control and not policy:
            found.append(('AGGREGATION_ERROR', f'Only a control register row carries {needed}; control rows are counted apart from written coverage.'))
    if duty.get('truncated') and not any(c == 'OBLIGATION_EXTRACTION_TRUNCATION' for c, _ in found):
        found.append(('OBLIGATION_EXTRACTION_TRUNCATION', f'The extracted duty keeps {duty["duty_words"]} of {duty["sentence_words"]} words of its clause sentence.'))
    label = row.get('label') or {}
    if label.get('disputed'):
        found.append(('LABEL_AMBIGUITY', 'Disputed label: ' + clip('; '.join(label['dispute_reasons']), 400)))
    if not found:
        found.append(('OTHER', f'No rule matched: expected {exp_cov} / conflict {exp.get("conflict")}, predicted {act_cov} / conflict {act.get("conflict")}.'))
    return found


def auto_category(row) -> dict:
    """{'primary', 'secondary', 'root_cause', 'rules'} of an error row, from category_rules."""
    rules = category_rules(row)
    primary, root_cause = rules[0]
    secondary = list(dict.fromkeys(c for c, _ in rules[1:] if c != primary))
    return {'primary': primary, 'secondary': secondary, 'root_cause': root_cause, 'rules': [{'category': c, 'rationale': r} for c, r in rules]}


def load_annotations(path) -> dict:
    """{row key: {'primary', 'secondary', 'root_cause', ...}} from a manual annotations file; {} without one.

    The file is {"format": ANNOTATIONS_FORMAT, ..., "annotations": {"C13 md.5(2)": {...}}}; a bare
    mapping of row keys is read too. An unknown category is an error, not a silent pass.
    """
    if not path or not Path(path).is_file():
        return {}
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    rows = data.get('annotations') if isinstance(data.get('annotations'), dict) else {k: v for k, v in data.items() if ' md.' in k}
    for key, value in rows.items():
        unknown = [c for c in [value.get('primary'), *(value.get('secondary') or [])] if c and c not in CATEGORIES]
        if unknown:
            raise ValueError(f'{key}: unknown categor{"y" if len(unknown) == 1 else "ies"} {", ".join(unknown)}; use one of {", ".join(CATEGORIES)}')
    return rows


def fixable_by(primary, secondary) -> str:
    """'label' when the label itself is the error, 'pipeline+label' when a pipeline error sits on a disputed label, else 'pipeline'."""
    if primary == 'LABEL_AMBIGUITY':
        return 'label'
    return 'pipeline+label' if 'LABEL_AMBIGUITY' in (secondary or []) else 'pipeline'


def settle(row, annotation=None) -> dict:
    """The row with its automatic category, the manual one when annotated, and the category that stands."""
    auto = auto_category(row)
    row['auto'] = auto
    if annotation:
        manual = {'primary': annotation.get('primary') or auto['primary'], 'secondary': list(annotation.get('secondary') or []),
                  'root_cause': annotation.get('root_cause') or auto['root_cause'], 'fix': annotation.get('fix'),
                  'label_verdict': annotation.get('label_verdict')}
        manual['fixable_by'] = annotation.get('fixable_by') or fixable_by(manual['primary'], manual['secondary'])
        row['manual'] = manual
        row['final'] = {**manual, 'source': 'manual'}
    else:
        row['manual'] = None
        row['final'] = {'primary': auto['primary'], 'secondary': auto['secondary'], 'root_cause': auto['root_cause'], 'fix': None,
                        'fixable_by': fixable_by(auto['primary'], auto['secondary']), 'source': 'auto'}
    return row


# ---------------------------------------------------------------- the run

def error_row(case, item, prediction, obligation, payload, calls=(), review=None, dataset_id=LEGACY_DATASET) -> dict:
    """The full record of one coverage/conflict error: expectation, prediction and the evidence chain between them."""
    expected = item['expected']
    obligation = obligation or {}
    proposal = obligation.get('proposal') or {}
    candidate = obligation.get('candidate') or {}
    needles = list(expected.get('evidence') or [])
    clause = clause_text(payload, obligation)
    passages = passage_rows(payload, obligation, needles, calls)
    coverage, conflict = item.get('coverage'), item.get('conflict')
    key = row_key(case.case_id, expected['article'], expected.get('clause') or '')
    return {'key': key, 'case_id': case.case_id, 'obligation_id': prediction.get('obligation_id'), 'obligation_key': prediction.get('key'),
            'article': expected['article'], 'clause': clause_id(expected.get('clause') or ''), 'provision': obligation.get('source_label'),
            'kinds': error_kinds(item), 'match_basis': item.get('match_basis'),
            'expected': {'coverage': coverage[0] if coverage else expected.get('coverage'), 'conflict': expected.get('conflict')},
            'actual': {'coverage': prediction.get('coverage'), 'conflict': prediction.get('conflict')},
            'applicability': {'expected': expected.get('applicability'), 'actual': prediction.get('applicability'),
                              'decided_by': prediction.get('decided_by')},
            'regulation_clause': clause, 'obligation': duty_view(candidate), 'duty_truncation': duty_truncation(candidate, clause),
            'passages': passages,
            'relevance_screen': next((d for d in obligation.get('diagnostics') or [] if isinstance(d, dict) and d.get('code') == 'RELEVANCE_SCREEN'), None),
            'aggregation': {'coverage': proposal.get('coverage'), 'coverage_reason': proposal.get('coverage_reason'),
                            'control_coverage': proposal.get('control_coverage'), 'review_flags': list(proposal.get('review_flags') or []),
                            'relations': dict(Counter(p['relation'] for p in passages if p['status'] == 'JUDGED')),
                            'cited': [q.get('source_id') for q in proposal.get('policy_evidence') or [] if isinstance(q, dict)],
                            'signals': len(obligation.get('signals') or [])},
            'evidence': evidence_view(payload, obligation, needles, passages),
            'label': label_view(case.case_id, expected, (review or {}).get(key, []), dataset_id),
            'model_calls': dict(Counter(str(c.get('task')) for c in calls)),
            'model_seconds': round(sum(int(c.get('elapsed_ms') or 0) for c in calls if str(c.get('cache_hit')) != 'True') / 1000, 1)}


def read_calls(run_dir) -> dict:
    """{(case_id, obligation_id): [model call records]} from ai-calls.jsonl."""
    path = Path(run_dir) / 'ai-calls.jsonl'
    out = {}
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                call = json.loads(line)
                out.setdefault((call.get('case_id'), call.get('obligation_id')), []).append(call)
    return out


def resolve_dataset(run_dir, manifest, dataset_path=None):
    """(dataset path, how it was found): harness.resolve_dataset (the argument, the manifest's path, or <evaluation>/datasets/<id>.json)."""
    from .harness import resolve_dataset as resolve
    return resolve(run_dir, manifest, dataset_path)


def default_label_review(run_dir):
    reports = Path(run_dir).parent.parent / 'reports'
    found = sorted(reports.glob('label-review-*.json')) if reports.is_dir() else []
    return found[-1] if found else None


def analyse_run(run_dir, dataset_path=None, annotations_path=None, label_review_path=None) -> dict:
    """The coverage-error report of a run directory (nothing is written)."""
    from .harness import load_dataset, pair_case, prediction_rows, score_case
    run_dir = Path(run_dir)
    manifest_path = run_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}
    dataset_file, found_by = resolve_dataset(run_dir, manifest, dataset_path)
    dataset = load_dataset(dataset_file)
    digest = sha256(dataset_file.read_bytes()).hexdigest()
    label_review_path = label_review_path or default_label_review(run_dir)
    review = load_label_review(label_review_path)
    annotations = load_annotations(annotations_path)
    calls = read_calls(run_dir)
    selected = set((manifest.get('mode') or {}).get('cases') or [])
    rows, unscored, skipped, scored_rows = [], [], [], 0
    for case in dataset.cases:
        if selected and case.case_id not in selected:
            continue
        packet_path = run_dir / 'packets' / f'{case.case_id}.json'
        if not packet_path.is_file():
            skipped.append({'case_id': case.case_id, 'reason': 'no packet (the case failed or was not run)'})
            continue
        payload = json.loads(packet_path.read_text(encoding='utf-8'))['events'][0]['payload']
        predictions = prediction_rows(payload, case.regulation_id)
        scored, _ = score_case(case, payload, predictions)
        # The same pairing score_case made (with the packet's provision texts: the SOURCE_SPAN pass needs them).
        pairs = [(match['expected'], match['prediction']) for match in pair_case(case.expected_obligations, predictions, payload)[0]]
        obligations = {o['id']: o for o in payload.get('obligations') or []}
        for (expected, prediction), item in zip(pairs, scored):
            if prediction is None:
                if expected.coverage in ASSESSED:
                    unscored.append({'key': row_key(case.case_id, expected.article, expected.clause), 'expected_coverage': expected.coverage,
                                     'expected_conflict': expected.conflict, 'reason': missing_text(item),
                                     'match_basis': item.get('match_basis'), 'missing_reason': item.get('missing_reason'),
                                     'missing_detail': item.get('missing_detail')})
                continue
            if 'coverage' in item and item['coverage'][0] in ASSESSED or 'conflict' in item:
                scored_rows += 1
            if not error_kinds(item):
                continue
            obligation = obligations.get(prediction['obligation_id'])
            row = error_row(case, item, prediction, obligation, payload, calls.get((case.case_id, prediction['obligation_id']), []), review,
                            dataset.dataset_id)
            rows.append(settle(row, annotations.get(row['key'])))
    unused = sorted(set(annotations) - {r['key'] for r in rows})
    return {'format': FORMAT, 'created_at': datetime.now(timezone.utc).isoformat(), 'run': str(run_dir), 'run_name': run_dir.name,
            'manifest_sha256': manifest.get('manifest_sha256'),
            'dataset': {'path': str(dataset_file), 'found_by': found_by, 'sha256': digest,
                        'sha256_at_run': (manifest.get('dataset') or {}).get('sha256'),
                        'same_as_run': digest == (manifest.get('dataset') or {}).get('sha256')},
            'label_review': str(label_review_path) if label_review_path else None,
            'annotations': str(annotations_path) if annotations_path and Path(annotations_path).is_file() else None,
            'annotations_unused': unused, 'categories': list(CATEGORIES), 'rules': 'regchain.evaluation.coverage_errors (module docstring)',
            'scored_rows': scored_rows, 'counts': counts(rows), 'rows': rows, 'unscored': unscored,
            'unscored_by_reason': dict(Counter(u['missing_reason'] or 'NOT_RECORDED' for u in unscored)), 'skipped': skipped}


# What each missing reason means for a coverage label that could not be scored (identifiers.MISSING_REASONS).
MISSING_TEXT = {'EXTRACTION_MISSED': 'no extraction attempt produced a duty from this clause',
                'GROUNDING_REJECTED': 'the model proposed a duty from this clause and the evidence gate rejected it',
                'MATCHING_FAILED': 'a duty was extracted from this article/clause but not paired with the label'}


def missing_text(item) -> str:
    reason = item.get('missing_reason')
    if not reason:
        return 'no extracted duty matched (not scored by the harness)'
    return f'{reason} ({item.get("missing_detail") or "—"}): {MISSING_TEXT.get(reason, reason)}; not scored by the harness'


def counts(rows) -> dict:
    by_class = {}
    for row in rows:
        entry = by_class.setdefault(row['expected']['coverage'] or 'ANY', {'errors': 0, 'pipeline': 0, 'label': 0, 'pipeline+label': 0})
        entry['errors'] += 1
        entry[row['final']['fixable_by']] = entry.get(row['final']['fixable_by'], 0) + 1
    return {'errors': len(rows), 'coverage_mismatch': sum('coverage_mismatch' in r['kinds'] for r in rows),
            'conflict_false_positive': sum('conflict_false_positive' in r['kinds'] for r in rows),
            'conflict_false_negative': sum('conflict_false_negative' in r['kinds'] for r in rows),
            'auto': dict(Counter(r['auto']['primary'] for r in rows)), 'final': dict(Counter(r['final']['primary'] for r in rows)),
            'final_secondary': dict(Counter(c for r in rows for c in r['final']['secondary'])),
            'by_expected_class': by_class}


# ---------------------------------------------------------------- Markdown

def cell(text, size=None) -> str:
    text = squash(text) if size is None else clip(text, size)
    return text.replace('|', '/')


def transition(row) -> str:
    exp, act = row['expected'], row['actual']
    text = f'{exp["coverage"]} → {act["coverage"]}'
    if exp.get('conflict') is not None and exp['conflict'] != act['conflict']:
        text += f' (çelişki {exp["conflict"]} → {act["conflict"]})'
    return text


def passage_table(row) -> list:
    lines = ['| # | Pasaj | Sıra / benzerlik / lexical / RRF / rerank | Ekran | contradicts | Doğrulayıcı | support (model) | Son ilişki | Beklenen kanıt |',
             '|---|---|---|---|---|---|---|---|---|']
    for index, p in enumerate(row['passages'], 1):
        r = p['retrieval']
        fj = p.get('first_judgement') or {}
        v = p.get('verifier')
        verifier = f'{v["code"]} ({v["by"]})' if v else ''
        support = f'{fj.get("support") or ""}' + (f' ({fj["support_by"]})' if fj.get('support_by') else '')
        relation = str(p.get('relation') or p['status']) + (' · kesildi' if p.get('truncation') else '') + (' · kontrol satırı' if p.get('control_row') else '')
        screen = str(p.get('screen') or p['status'])
        if p.get('pipeline') == V19:        # v0.19: the fast classifier's label and why (or whether) it was escalated
            screen += f' · hızlı {(p.get("fast") or {}).get("label") or "—"} · yükseltme {p.get("escalation") or "yok"}'
        lines.append(f'| {index} | `{p["source_id"][:8]}` {cell(p["text"], 160)} | {r.get("rank")} / {r.get("similarity")} / {r.get("lexical_score")} / '
                     f'{r.get("rrf")} / {r.get("rerank_score")} | {cell(screen)} | {fj.get("contradicts") or ""} | {verifier} | '
                     f'{support} | {relation} | {cell(", ".join(p["expected_evidence"]))} |')
    return lines


def markdown(report) -> str:
    rows = report['rows']
    c = report['counts']
    ds = report['dataset']
    lines = [f'# Kapsam / çelişki hata analizi — `{report["run_name"]}`', '',
             f'Koşu: `{report["run"]}`. Veri seti: `{ds["path"]}` ({ds["found_by"]}; sha256 `{ds["sha256"][:12]}`, koşu anındaki '
             f'`{str(ds["sha256_at_run"])[:12]}`{"" if ds["same_as_run"] else " — **koşudan sonra değişmiş; satırlar güncel etiketlerle yeniden eşlendi**"}). '
             f'Etiket incelemesi: `{report["label_review"]}`. Manuel açıklamalar: `{report["annotations"]}`.', '',
             f'Puanlanan kapsam/çelişki satırı: {report["scored_rows"]}. Hata: **{c["errors"]}** (kapsam uyuşmazlığı {c["coverage_mismatch"]}, '
             f'çelişki yanlış pozitif {c["conflict_false_positive"]}, yanlış negatif {c["conflict_false_negative"]}). '
             'Seçim ve kategori kuralları: `regchain.evaluation.coverage_errors` modül açıklaması. Hiçbir etiket değiştirilmedi.', '',
             '## Hata tablosu', '',
             '| Satır | Yükümlülük | Beklenen → gerçek | Otomatik kategori | Manuel kategori | İkincil (nihai) | Kök neden (nihai) |', '|---|---|---|---|---|---|---|']
    for row in rows:
        manual = row.get('manual') or {}
        lines.append(f'| {row["key"]} | `{row["obligation_id"][:12]}` {cell(row["obligation"].get("required_action") or row["obligation"].get("prohibited_action"), 70)} | '
                     f'{transition(row)} | {row["auto"]["primary"]} | {manual.get("primary") or "—"} | {", ".join(row["final"]["secondary"]) or ""} | '
                     f'{cell(row["final"]["root_cause"])} |')
    lines += ['', '## Kategori sayıları', '', '| Kategori | Otomatik (birincil) | Nihai (birincil) | Nihai (ikincil) |', '|---|---|---|---|']
    for name in CATEGORIES:
        a, f, s = c['auto'].get(name, 0), c['final'].get(name, 0), c['final_secondary'].get(name, 0)
        if a or f or s:
            lines.append(f'| {name} | {a} | {f} | {s} |')
    lines.append(f'| **toplam** | {len(rows)} | {len(rows)} | {sum(c["final_secondary"].values())} |')
    lines += ['', '## Beklenen kapsam sınıfına göre: boru hattı mı, etiket mi?', '',
              '`pipeline`: boru hattı değişikliği düzeltir; `label`: etiketin kendisi savunulamaz/tartışmalı; `pipeline+label`: boru hattı hatası, '
              'etiket de tartışmalı.', '', '| Beklenen sınıf | Hata | pipeline | pipeline+label | label | Satırlar |', '|---|---|---|---|---|---|']
    for name, entry in c['by_expected_class'].items():
        keys = ', '.join(f'{r["key"]} ({r["final"]["fixable_by"]})' for r in rows if r['expected']['coverage'] == name)
        lines.append(f'| {name} | {entry["errors"]} | {entry.get("pipeline", 0)} | {entry.get("pipeline+label", 0)} | {entry.get("label", 0)} | {keys} |')
    fixes = [r for r in rows if (r.get('manual') or {}).get('fix')]
    if fixes:
        lines += ['', '## Önerilen boru hattı değişiklikleri', '', '| Satır | Nihai kategori | Değişiklik |', '|---|---|---|']
        lines += [f'| {r["key"]} | {r["final"]["primary"]} | {cell(r["manual"]["fix"])} |' for r in fixes]
    lines += ['', '## Satır ayrıntıları']
    for row in rows:
        duty, trunc, ev, agg = row['obligation'], row['duty_truncation'], row['evidence'], row['aggregation']
        lines += ['', f'### {row["key"]} — {transition(row)}', '',
                  f'- obligation_id: `{row["obligation_id"]}` · obligation_key: `{row["obligation_key"]}` · hüküm: {row["provision"]} · türler: {", ".join(row["kinds"])}',
                  f'- Uygulanabilirlik: beklenen {row["applicability"]["expected"]}, gerçek {row["applicability"]["actual"]} ({row["applicability"]["decided_by"]})',
                  f'- **Yönetmelik fıkrası:** {cell(row["regulation_clause"], 900)}',
                  f'- **Çıkarılan yükümlülük:** subject `{duty.get("subject")}` · modality `{duty.get("modality")}` · required_action `{duty.get("required_action")}` · '
                  f'prohibited_action `{duty.get("prohibited_action")}` · conditions {duty.get("conditions")} · exceptions {duty.get("exceptions")} · deadline {duty.get("deadline")}',
                  f'- Yükümlülük uzunluğu: {trunc["duty_words"]} / {trunc["sentence_words"]} kelime (oran {trunc["ratio"]}; kesik: {trunc["truncated"]})',
                  f'- **Beklenen kanıt:** ' + ('; '.join(f'"{n["text"]}" → ' + (', '.join(f'`{p["source_id"][:8]}` {p["status"]} (sıra {p["rank"]})' for p in n['passages'])
                                                                           or 'pakette yok') for n in ev['needles']) or 'etiket kanıt metni vermiyor')
                  + f' · yargılanan pasajda var: {ev["judged_contains"]} · ekranda elendi: {ev["screened_out"]} · filtrelendi: {ev["filtered"]} · alıntılandı: {ev["cited"]}',
                  f'- **Toplama:** {agg["coverage"]} — {cell(agg["coverage_reason"])} (ilişkiler {agg["relations"]}; kontrol {agg["control_coverage"]}; bayraklar {agg["review_flags"] or "—"})',
                  f'- Etiket: tartışmalı {row["label"]["disputed"]}' + (f' — {cell("; ".join(row["label"]["dispute_reasons"]), 300)}' if row['label']['disputed'] else '')
                  + (f' · inceleme: {cell("; ".join(row["label"]["review_reasons"]), 200)}' if row['label']['review_reasons'] else '')
                  + f' · not: {cell(row["label"]["notes"], 200)}',
                  f'- Model çağrıları: {row["model_calls"]} ({row["model_seconds"]} sn, önbellek hariç)', '']
        lines += passage_table(row)
        for p in row['passages']:
            fj, v = p.get('first_judgement') or {}, p.get('verifier')
            extra = []
            if fj.get('contradicts_quote'):
                extra.append(f'contradicts alıntısı: "{cell(fj["contradicts_quote"], 300)}"')
            if v and v.get('detail'):
                extra.append(f'doğrulayıcı: {cell(v["detail"], 300)}')
            if p.get('truncation'):
                extra.append(f'kesilme: {cell(p["truncation"])}')
            if p['status'] == 'JUDGED' and (p.get('relation') != 'UNRELATED' or p.get('expected_evidence') or extra):
                extra.append(f'gerekçe: {cell(p.get("reason"), 300)}')
            if extra:
                lines.append(f'- `{p["source_id"][:8]}`: ' + ' · '.join(extra))
        lines += ['', f'**Otomatik:** {row["auto"]["primary"]}' + (f' (+ {", ".join(row["auto"]["secondary"])})' if row['auto']['secondary'] else '')
                  + f' — {cell(row["auto"]["root_cause"])}']
        if row.get('manual'):
            m = row['manual']
            lines += ['', f'**Manuel:** {m["primary"]}' + (f' (+ {", ".join(m["secondary"])})' if m['secondary'] else '') + f' — {cell(m["root_cause"])}']
            if m.get('label_verdict'):
                lines += ['', f'Etiket değerlendirmesi: {cell(m["label_verdict"])}']
            if m.get('fix'):
                lines += ['', f'Düzeltme: {cell(m["fix"])}']
    if report['unscored']:
        by_reason = report.get('unscored_by_reason') or {}
        lines += ['', '## Puanlanmayan (eşleşmeyen) kapsam etiketleri', '',
                  'Eksik nedeni: ' + (', '.join(f'{reason} {n}' for reason, n in by_reason.items()) or '—')
                  + ' (EXTRACTION_MISSED: fıkradan aday çıkmadı; GROUNDING_REJECTED: aday kanıt kapısında reddedildi; '
                    'MATCHING_FAILED: fıkradan aday var ama etiketle eşleşmedi).', '']
        lines += [f'- {u["key"]}: {u["expected_coverage"]} — {u["reason"]}' for u in report['unscored']]
    if report['annotations_unused']:
        lines += ['', '## Kullanılmayan açıklamalar', ''] + [f'- {k}' for k in report['annotations_unused']]
    if report['skipped']:
        lines += ['', '## Atlanan vakalar', ''] + [f'- {s["case_id"]}: {s["reason"]}' for s in report['skipped']]
    return '\n'.join(lines) + '\n'


def write_report(run_dir, dataset_path=None, annotations_path=None, out_json=None, out_md=None, label_review_path=None):
    """coverage-errors.json and coverage-errors.md (default: in the run directory); returns (report, json path, md path)."""
    run_dir = Path(run_dir)
    if annotations_path is None:
        guess = run_dir.parent.parent / 'reports' / f'coverage-error-annotations-{run_dir.name}.json'
        annotations_path = guess if guess.is_file() else None
    report = analyse_run(run_dir, dataset_path, annotations_path, label_review_path)
    json_path = Path(out_json) if out_json else run_dir / 'coverage-errors.json'
    md_path = Path(out_md) if out_md else run_dir / 'coverage-errors.md'
    for path in (json_path, md_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    md_path.write_text(markdown(report), encoding='utf-8')
    return report, json_path, md_path
