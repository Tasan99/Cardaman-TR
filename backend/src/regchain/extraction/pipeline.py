from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import time

from pydantic import ValidationError
from regchain.model_router import ModelPolicyError

from .grounding import preflight, verify, CONDITIONS, EXCEPTIONS
from .providers import Provider, ProviderFailure, ContextBudgetError, failure_code
from .schema import ExtractionOutput
from .spans import SOURCE_BOUND_OK, UNCOVERED_CODE, accepted_trace, action_of, cover_sentences, ground, repair_actions, salvage
from .quantities import evidence_safe


@dataclass(frozen=True)
class Result:
    output: ExtractionOutput
    reason: str
    attempts: int
    human_review_required: bool = True
    diagnostics: tuple[dict,...] = ()


def extract(text: str, provider: Provider, context=None) -> Result:
    if hasattr(provider, 'last_calls'):
        provider.last_calls = []
    issue = preflight(text,context)
    if issue:
        return Result(ExtractionOutput(status='INSUFFICIENT_EVIDENCE'), issue, 0)
    if context and not hasattr(provider,'generate_with_context') and any(
            i.get('required',i['reason'] in ('exact_reference','chapter_reference','glossary_definition'))
            for i in context.items):
        return Result(ExtractionOutput(status='INSUFFICIENT_EVIDENCE'),'CONTEXT_REQUIRES_LLM',0)
    diagnostics=[]
    feedback=None
    repaired=False
    # v0.19 t5: every answer the gate rejected, (attempt, output, spans.ground result), latest last.
    rejected=[]
    for attempt in range(1, 4):
        started=time.monotonic()
        raw=''
        stage='generation'
        try:
            if feedback is not None:
                repaired=True
                raw=provider.repair(text,context,feedback)
            else:
                raw = provider.generate_with_context(text,context) if hasattr(provider,'generate_with_context') else provider.generate(text)
            stage='schema'
            output = ExtractionOutput.model_validate_json(raw)
            # v0.19: a Turkish action the model cut short ("uygulayamazlar") gets its governing
            # clause back before the gate; see spans.py. Every provider and repair goes through it.
            stage='span_repair'
            repaired_output, repairs = repair_actions(text, output)
            stage='grounding'
            if repairs:
                try:
                    verify(text, repaired_output, context)
                    output = repaired_output
                except ValueError:
                    # The repair never turns an answer the gate accepts into one it rejects.
                    repairs = []
                    verify(text, output, context)
            else:
                verify(text, output,context)
            stage='review'
            # v0.19 t7: the review decides per candidate (second_reading, apply_review); only a review that rejects every
            # candidate, whose decisions cannot be paired with the candidates, or whose answer cannot be read (read_review)
            # ends the unit.
            decisions, ending, failed = read_review(provider,text,output,context) if output.status=='EXTRACTED' and hasattr(provider,'review') else (None, None, None)
            if ending is not None:
                diagnostics.append({'attempt':attempt,**withheld(output,decisions,ending,failed)})
                return Result(ExtractionOutput(status='INSUFFICIENT_EVIDENCE'),REVIEW_CODE,attempt,diagnostics=tuple(diagnostics))
            reason = 'CANDIDATE_REQUIRES_LEGAL_REVIEW' if output.status == 'EXTRACTED' else output.status
            diagnostics.extend({'attempt':attempt,**repair} for repair in repairs)
            trace = accepted_trace(output,repairs) if output.status=='EXTRACTED' else None
            output, trace, rejected_by_review, reviewed = apply_review(output, trace, decisions)
            if reviewed is not None:
                diagnostics.append({'attempt':attempt,**reviewed})
            # v0.19 t6: every duty sentence of the unit gets a duty of its own (spans.cover_sentences), after the review;
            # v0.19 t7: never again the sentence of a duty the review rejected.
            output, trace, covered = cover_output(text, output, context, trace, rejected_by_review)
            if covered is not None:
                diagnostics.append({'attempt':attempt,**covered})
            diagnostics.append({'attempt':attempt,'stage':'complete','code':reason,
                'elapsed_ms':int((time.monotonic()-started)*1000),'calls':evidence_safe(getattr(provider,'last_calls',[])[:]),
                **({'grounding':trace} if output.status=='EXTRACTED' else {})})
            return Result(output, reason, attempt,diagnostics=tuple(diagnostics))
        except ModelPolicyError:
            raise
        except ContextBudgetError as exc:
            diagnostics.append({'attempt':attempt,'stage':stage,'code':'CONTEXT_BUDGET_EXCEEDED','detail':str(exc)})
            return Result(ExtractionOutput(status='INSUFFICIENT_EVIDENCE'),'CONTEXT_BUDGET_EXCEEDED',attempt,diagnostics=tuple(diagnostics))
        except ProviderFailure as exc:
            diagnostics.append({'attempt':attempt,'stage':stage,'code':'PROVIDER_FAILURE','detail':str(exc)[:1000]})
            if attempt == 3:
                return Result(ExtractionOutput(status='INSUFFICIENT_EVIDENCE'), 'PROVIDER_FAILURE', attempt,diagnostics=tuple(diagnostics))
        except (ValidationError, ValueError) as exc:
            detail=str(exc)[:2000]
            response=getattr(provider,'last_raw',raw)
            grounded = stage=='grounding' and not isinstance(exc,ValidationError)
            ladder = ground_output(text, output, context) if grounded else None
            diagnostics.append({'attempt':attempt,'stage':stage,'code':'SCHEMA_INVALID' if isinstance(exc,ValidationError) else 'EVIDENCE_INVALID',
                'detail':detail,'response_excerpt':response[:6000],
                'response_sha256':sha256(response.encode()).hexdigest(),'response_truncated':len(response)>6000,
                'elapsed_ms':int((time.monotonic()-started)*1000),
                # v0.19 t5: why each candidate of this answer passes or fails the grounding ladder (spans.ground).
                **({'grounding':ladder[2]} if ladder is not None else {})})
            if grounded:
                rejected.append((attempt, output, ladder))
            if stage=='review' or repaired or not hasattr(provider,'repair') or attempt==3:
                # v0.19: the last answer still fails the gate; select source spans for it (spans.ground) and
                # ask the gate again. v0.19 t5: the last answer first, then the earlier ones - a repair that
                # made a correct answer worse no longer loses the duty. Only an answer that passes verify is kept.
                rescued = next(((number, found) for number, _, found in reversed(rejected) if found is not None and found[0] is not None), None)
                if rescued is not None:
                    number, (output, record, trace) = rescued
                    diagnostics.append({'attempt':number,**record})
                    # t7 review 1 (R4): this runs inside the except handler, so a failing review must never raise from here
                    # (it used to escape extract and lose the whole case): it ends the unit on record (REVIEW_FAILED).
                    decisions, ending, failed = read_review(provider,text,output,context,rescue=True) if hasattr(provider,'review') else (None, None, None)
                    if ending is not None:
                        diagnostics.append({'attempt':attempt,**withheld(output,decisions,ending,failed)})
                        return Result(ExtractionOutput(status='INSUFFICIENT_EVIDENCE'),REVIEW_CODE,attempt,diagnostics=tuple(diagnostics))
                    output, trace, rejected_by_review, reviewed = apply_review(output, trace, decisions)
                    if reviewed is not None:
                        diagnostics.append({'attempt':attempt,**reviewed})
                    output, trace, covered = cover_output(text, output, context, trace, rejected_by_review)
                    if covered is not None:
                        diagnostics.append({'attempt':attempt,**covered})
                    diagnostics.append({'attempt':attempt,'stage':'complete','code':'CANDIDATE_REQUIRES_LEGAL_REVIEW',
                        'elapsed_ms':int((time.monotonic()-started)*1000),'calls':evidence_safe(getattr(provider,'last_calls',[])[:]),
                        'grounding':trace})
                    return Result(output,'CANDIDATE_REQUIRES_LEGAL_REVIEW',attempt,diagnostics=tuple(diagnostics))
                if rejected:
                    last = rejected[-1][2]
                    diagnostics.append({'attempt':attempt,'stage':'grounding','code':'GROUNDING_REJECTED',
                                        'grounding':last[2] if last is not None else []})
                return Result(ExtractionOutput(status='INSUFFICIENT_EVIDENCE'),'GROUNDING_REJECTED',attempt,diagnostics=tuple(diagnostics))
            feedback=detail
    raise AssertionError('unreachable')


def ground_output(text: str, output: ExtractionOutput, context=None):
    """spans.ground, which never breaks extraction: any failure keeps the rejection (and says so)."""
    try:
        return ground(text, output, context)
    except Exception as exc:  # noqa: BLE001 - the ladder must never turn a rejection into a crash
        return None, None, [{'candidate': -1, 'code': 'STRUCTURED_MISMATCH', 'detail': 'LADDER_FAILED:' + type(exc).__name__}]


def cover_output(text: str, output: ExtractionOutput, context=None, trace=None, rejected=()):
    """spans.cover_sentences, which never breaks extraction: any failure keeps the accepted answer as it is (and says so).
    `rejected`: the candidates the review dropped (apply_review); the sentences of their markers are not filled again."""
    try:
        if rejected:
            return cover_sentences(text, output, context, trace, rejected=rejected)
        return cover_sentences(text, output, context, trace)
    except Exception as exc:  # noqa: BLE001 - completing an accepted answer must never lose it
        return output, trace, {'stage': 'grounding', 'code': UNCOVERED_CODE, 'uncovered': [], 'detail': 'COVERAGE_FAILED:' + type(exc).__name__}


def salvage_output(text: str, output: ExtractionOutput, context=None):
    """spans.salvage, which never breaks extraction: any failure keeps the rejection."""
    try:
        return salvage(text, output, context)
    except Exception:  # noqa: BLE001 - a salvage must never turn a rejection into a crash
        return None


# --- v0.19 t7: the second reading decides per candidate -------------------------------------------------------------
# provider.review answers one decision per candidate, in candidate order (providers.REVIEW_PROMPT). Up to t6 any
# decision other than SUPPORTED emptied the whole unit. Measured in the v019t6 micro run (Tedbirler md. 24/A(3)): the
# model gave the sub-paragraph's three duties, the labelled "... iade eder" one among them; the review doubted one of
# them, the unit ended SECOND_PASS_UNCERTAIN with no duty, and the sentence coverage step never ran. Now each decision
# acts on its own candidate:
#   SUPPORTED   -> KEEP       the candidate stays as it is;
#   UNSUPPORTED -> REJECT     that candidate alone is dropped; spans.cover_sentences does not fill its sentence again;
#   UNCERTAIN   -> UNCERTAIN  the candidate stays, with confidence REVIEW_UNCERTAIN_CONFIDENCE and its decision on its
#                             trace entry ('review'); a doubt about one candidate never removes another.
# The unit ends (reason SECOND_PASS_UNCERTAIN, as before) only when the review rejects every candidate (each by its own
# decision), when its decisions cannot be paired with the candidates (another count, a value outside the schema, a provider
# whose review answers one bool and says no), or when the review call fails or its answer cannot be read (read_review).
# Its terminal record names why (REVIEW_OUTCOMES) and keeps every decision and every withheld candidate: nothing is lost
# without a record. A review that confirms every candidate changes nothing: the output, the trace and the diagnostics are
# those of t6, with no record. No second check is asked for an UNCERTAIN candidate: its row goes to a person (the pilot
# flags it EXTRACTION_REVIEW_UNCERTAIN, review_uncertain), and one more draw of the same model is no stronger evidence.
# t7 review 1 (R1): a candidate's fate is a function of its own decision only. Before, a review with no SUPPORTED decision
# ended the unit, so an UNCERTAIN candidate was kept beside a SUPPORTED sibling ([S, U]) and withheld beside a rejected
# one ([R, U]) or beside another doubted one ([U, U]): a valid candidate lost because of an unrelated candidate's review.
# Now UNCERTAIN always keeps its candidate (flagged), whatever its siblings got; ALL_UNCERTAIN and NONE_SUPPORTED no longer
# end a unit and are kept only as names for older records.
KEEP, REJECT, UNCERTAIN = 'KEEP', 'REJECT', 'UNCERTAIN'
REVIEW_ACTIONS = {'SUPPORTED': KEEP, 'UNSUPPORTED': REJECT, 'UNCERTAIN': UNCERTAIN}
REVIEW_CODE = 'SECOND_PASS_UNCERTAIN'                   # the reason, and the terminal record's code, of a unit the review ends
PER_CANDIDATE_CODE = 'SECOND_PASS_PER_CANDIDATE'        # the record of a review that dropped or flagged some candidates
ALL_UNCERTAIN = 'REVIEW_ALL_UNCERTAIN'                  # (t7 P2 only) every decision UNCERTAIN; no longer an ending (R1)
NONE_SUPPORTED = 'REVIEW_NONE_SUPPORTED'                # (t7 P2 only) no SUPPORTED decision; no longer an ending (R1)
ALL_REJECTED = 'REVIEW_ALL_REJECTED'                    # every decision UNSUPPORTED: each candidate rejected by its own decision
MISALIGNED = 'REVIEW_DECISIONS_MISALIGNED'              # not one readable decision per candidate
NO_DECISIONS = 'REVIEW_NOT_PER_CANDIDATE'               # a review that answers one bool, and it said no
REVIEW_FAILED = 'REVIEW_FAILED'                         # t7 review 1 (R4): the review call failed or its answer was unreadable
REVIEW_OUTCOMES = (ALL_REJECTED, MISALIGNED, NO_DECISIONS, REVIEW_FAILED)
REVIEW_UNCERTAIN_CONFIDENCE = '0.2500'                  # contract.materialize gives every candidate 0.5000


def second_reading(provider, text: str, output: ExtractionOutput, context=None) -> tuple[list[str], str | None]:
    """(decisions, ending) for an accepted answer: the review's decisions, and the REVIEW_OUTCOMES code that ends the
    unit, or None when it goes on (one readable decision per candidate, not every one UNSUPPORTED).

    OllamaProvider.review answers the decisions in candidate order. A review that answers a bool (the t6 test doubles) is
    read as before: true confirms every candidate, false (or no answer) ends the unit (NO_DECISIONS). t7 review 1: any other
    answer (a string such as "UNCERTAIN", a dict) is no confirmation: it cannot be paired with the candidates (MISALIGNED); t6
    read any truthy value as confirming every candidate. A failing review raises (read_review reads the failure).
    """
    answer = provider.review(text, output, context)
    count = len(output.obligations)
    if answer is None or isinstance(answer, bool):
        return (['SUPPORTED'] * count, None) if answer else ([], NO_DECISIONS)
    if not isinstance(answer, (list, tuple)):
        return [str(answer)[:200]], MISALIGNED
    decisions = [str(decision) for decision in answer]
    if not decisions or len(decisions) != count or any(decision not in REVIEW_ACTIONS for decision in decisions):
        return decisions, MISALIGNED
    if all(REVIEW_ACTIONS[decision] == REJECT for decision in decisions):
        return decisions, ALL_REJECTED
    return decisions, None


def read_review(provider, text: str, output: ExtractionOutput, context=None, rescue: bool = False):
    """(decisions, ending, failed): second_reading, with a failing review read instead of raised (t7 review 1, R4).

    An answer that cannot be read (a cut or malformed JSON, a value the schema refuses: ValidationError, ValueError,
    KeyError, TypeError) ends the unit with REVIEW_FAILED, and `failed` names it (providers.failure_code, e.g.
    MALFORMED_JSON); before, the direct path reported it as GROUNDING_REJECTED with no withheld record. A provider failure
    (TIMEOUT, OUTPUT_TRUNCATED) or a context overflow is raised on the direct path, where extract handles it as before (the
    attempt is retried; a context overflow ends the unit CONTEXT_BUDGET_EXCEEDED). On the ladder `rescue` path, which runs
    inside extract's except handler, nothing is raised: it used to escape extract and lose the whole case. There every
    failure ends the unit with REVIEW_FAILED, on record.
    """
    try:
        decisions, ending = second_reading(provider, text, output, context)
        return decisions, ending, None
    except ModelPolicyError:
        raise
    except (ProviderFailure, ContextBudgetError) as exc:
        if not rescue:
            raise
        return [], REVIEW_FAILED, failure_code(exc)
    except (ValidationError, ValueError, KeyError, TypeError) as exc:
        return [], REVIEW_FAILED, failure_code(exc)


def review_uncertain(result) -> list[int]:
    """The positions in result.output.obligations of the candidates the review kept as UNCERTAIN (the 'uncertain' list of
    the unit's PER_CANDIDATE_CODE record; [] when there is none). The explicit marker the pilot reads to flag such a row
    EXTRACTION_REVIEW_UNCERTAIN (t7 review 1, R3); confidence_score is not read. The sentence coverage step only appends
    duties after the reviewed candidates (spans.cover_sentences), so the positions stay those of the final output."""
    count = len(result.output.obligations) if result.output.status == 'EXTRACTED' else 0
    records = [d for d in result.diagnostics or () if isinstance(d, dict) and d.get('stage') == 'review' and d.get('code') == PER_CANDIDATE_CODE]
    if not records:
        return []
    return sorted({position for position in records[-1].get('uncertain') or [] if isinstance(position, int) and 0 <= position < count})


def apply_review(output: ExtractionOutput, trace, decisions):
    """(output, trace, rejected, record) once the review's decisions (second_reading, one per candidate) are applied.

    No review, or every decision SUPPORTED: the output and the trace come back as they are, with no rejected candidate
    and no record. Otherwise the UNSUPPORTED candidates are dropped and returned as `rejected` (for
    spans.cover_sentences), the UNCERTAIN ones keep their place with confidence REVIEW_UNCERTAIN_CONFIDENCE, the trace
    entry of every doubted candidate gets 'review' (its decision; entries line up with candidates as in
    spans.cover_sentences), and the record is {'stage': 'review', 'code': PER_CANDIDATE_CODE, 'decisions', 'uncertain':
    [positions in the new output], 'rejected': [{'candidate', 'action'}]}. Every kept field is the one the gate accepted.
    Strings and ints only (digest-safe).
    """
    if not decisions or all(REVIEW_ACTIONS.get(decision) == KEEP for decision in decisions):
        return output, trace, [], None
    trace = [dict(entry) for entry in trace or []]
    kept_entries = [e for e in trace if e.get('code') == SOURCE_BOUND_OK and 'merged_into' not in e]
    entries = kept_entries if len(kept_entries) == len(output.obligations) else [None] * len(output.obligations)
    kept, rejected, uncertain, dropped = [], [], [], []
    for position, (value, decision, entry) in enumerate(zip(output.obligations, decisions, entries)):
        action = REVIEW_ACTIONS[decision]
        if action != KEEP and entry is not None:
            entry['review'] = decision
        if action == REJECT:
            rejected.append(value)
            dropped.append({'candidate': entry['candidate'] if entry else position, 'action': action_of(value)})
            continue
        if action == UNCERTAIN:
            uncertain.append(len(kept))
            value = value.model_copy(update={'confidence_score': Decimal(REVIEW_UNCERTAIN_CONFIDENCE)})
        kept.append(value)
    record = {'stage': 'review', 'code': PER_CANDIDATE_CODE, 'decisions': list(decisions), 'uncertain': uncertain, 'rejected': dropped}
    return ExtractionOutput(status='EXTRACTED', obligations=kept), trace, rejected, record


def withheld(output: ExtractionOutput, decisions, ending: str, failed: str | None = None) -> dict:
    """The terminal record of a unit the review ends: why (`ending`, a REVIEW_OUTCOMES code), every decision as answered,
    and every candidate withheld (subject, modality, action, conditions, exceptions), so the loss is on record and the
    unit can be replayed; with REVIEW_FAILED also 'failure', the failure code (read_review). Strings and ints only
    (digest-safe)."""
    return {'stage': 'review', 'code': REVIEW_CODE, 'review': ending, 'decisions': list(decisions or []),
            'candidates': len(output.obligations), **({'failure': str(failed)} if failed else {}),
            'withheld': [{'subject': value.subject, 'modality': value.modality, 'action': action_of(value),
                          'conditions': list(value.conditions), 'exceptions': list(value.exceptions)} for value in output.obligations]}
