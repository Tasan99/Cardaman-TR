"""Three-stage judgement evidence; correctness belongs only to offline evaluation.

The production trace contains observations, never an inferred gold label. Every
attempt is retained, including a rejected answer and a subsequent empty failure.
"""
from copy import deepcopy
from hashlib import sha256
import json
import math

OUTCOME_CODES = ('RAW_CORRECT_FINAL_WRONG', 'RAW_WRONG_FINAL_CORRECT',
                 'RAW_WRONG_FINAL_WRONG', 'RAW_CORRECT_FINAL_CORRECT')
FAILURE_TAXONOMY = ('MODEL_SEMANTIC_ERROR', 'MODEL_VARIABILITY', 'PROMPT_FAILURE',
                   'RETRIEVAL_FAILURE', 'VALIDATOR_OVERREJECT', 'VALIDATOR_UNDERREJECT',
                   'PIPELINE_SCOPE_ERROR', 'PIPELINE_INHERITANCE_ERROR', 'PARSING_ERROR',
                   'CONTEXT_ADMISSION', 'OUTPUT_TRUNCATION', 'DATASET_AMBIGUITY', 'UNKNOWN')


def plain(value):
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json')
    return deepcopy(value)


def decision(value, kind='relation'):
    value = plain(value)
    if not isinstance(value, dict):
        return None
    if kind == 'applicability':
        result = value.get('applicability')
        return result if result in ('APPLIES', 'DOES_NOT_APPLY', 'UNKNOWN', 'POSSIBLY_APPLIES') else None
    if type(value.get('conflict')) is bool:
        if value['conflict']:
            return 'CONFLICTS'
        result = value.get('relation_if_no_conflict')
        return result if result in ('SUPPORTS', 'PARTIAL', 'UNRELATED', 'NOT_APPLICABLE') else None
    result = value.get('relation')
    if result in ('CONFLICTS', 'SUPPORTS', 'PARTIAL', 'UNRELATED', 'NOT_APPLICABLE', 'UNCLEAR'):
        return result
    return {'IRRELEVANT': 'UNRELATED', 'POSSIBLE_SUPPORT': 'SUPPORTS', 'SUPPORTS': 'SUPPORTS',
            'POSSIBLE_PARTIAL': 'PARTIAL', 'PARTIAL': 'PARTIAL',
            'POSSIBLE_CONFLICT': 'CONFLICTS'}.get(value.get('label'))


def attempt_record(raw, validated=None, *, attempt=1, status='OK', error=None, call=None, kind='relation'):
    call = call or {}
    def reject_constant(value):
        raise ValueError(f'Non-finite JSON number: {value}')

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            return reject_constant(value)
        return number

    try:
        parsed = json.loads(raw, parse_constant=reject_constant, parse_float=finite_float) if raw else None
    except (ValueError, TypeError):
        parsed = None
    if not isinstance(parsed, dict):
        parsed = None
    checked = plain(validated)
    changes = {key: {'raw': parsed.get(key), 'validated': checked.get(key)}
               for key in sorted(set(parsed) | set(checked)) if parsed.get(key) != checked.get(key)} \
        if isinstance(parsed, dict) and isinstance(checked, dict) else {}
    valid_transport = status in ('OK', 'VALIDATION_REJECTED', 'JUDGEMENT_INVALID') and call.get('done_reason') != 'length'
    return {'attempt': attempt, 'status': status, 'request_hash': call.get('request_hash'),
            'raw_output_hash': sha256(raw.encode('utf-8')).hexdigest() if raw else None,
            'raw_text': raw if isinstance(raw, str) else None,
            'raw': parsed, 'validated': checked, 'raw_decision': decision(parsed, kind) if valid_transport else None,
            'validated_decision': decision(checked, kind), 'validation_changes': changes,
            'validation_error': str(error)[:500] if error else None,
            'budget': {key: call.get(key) for key in ('num_ctx', 'num_predict', 'reserved_output',
                        'estimated_prompt_tokens', 'prompt_tokens', 'output_tokens', 'done_reason',
                        'elapsed_ms', 'cache_hit', 'retry_count')}}


def finish_trace(attempts, final, *, notes=(), kind='relation', source_ids=()):
    """Last attempt semantics are explicit; do not resurrect an earlier failed claim."""
    attempts = plain(attempts)
    last = attempts[-1] if attempts else {}
    final_value = plain(final)
    gates = [plain(note) for note in notes if isinstance(note, dict) and note.get('code')]
    return {'format': 'cardaman-decision-trace-v1', 'kind': kind,
            'numeric_storage_encoding': {
                'packet_boundary': 'evidence_safe: finite parsed floats round to 6 decimal places; whole values become integers, others decimal strings',
                'original_response': 'attempts[].raw_text is verbatim; raw_output_hash hashes its UTF-8 bytes; parsed numeric JSON types may differ in packets'},
            'RAW_MODEL_DECISION': {'decision': last.get('raw_decision'), 'value': last.get('raw'),
                                   'status': last.get('status', 'NOT_CALLED'), 'request_hash': last.get('request_hash')},
            'VALIDATED_MODEL_DECISION': {'decision': last.get('validated_decision'), 'value': last.get('validated'),
                                         'changes': last.get('validation_changes', {}), 'error': last.get('validation_error')},
            'FINAL_PIPELINE_DECISION': {'decision': decision(final_value, kind), 'value': final_value},
            'attempts': attempts, 'gate_events': gates, 'source_ids': list(source_ids),
            'correctness': 'NOT_EVALUATED_NO_GOLD'}


def evaluation_outcome(trace, expected):
    """Evaluation-only annotation. The expected label must come from frozen external gold."""
    if expected is None:
        return None
    raw = trace.get('RAW_MODEL_DECISION', {}).get('decision')
    final = trace.get('FINAL_PIPELINE_DECISION', {}).get('decision')
    if raw is None or final is None:
        return None
    return f'RAW_{"CORRECT" if raw == expected else "WRONG"}_FINAL_{"CORRECT" if final == expected else "WRONG"}'
