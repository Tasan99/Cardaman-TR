"""Per-task evaluation for sector packs: one-vs-rest rates, sample sufficiency, unsupported conclusions, contrast pairs.

Kept apart from metrics.py so the AML regression gate and its frozen baselines read exactly the numbers
they always read. Every function takes plain rows and returns plain numbers; nothing here reads files.

A task report states what it may claim. ACCURACY needs expert gold (two blind reviews and adjudication,
expert_gold.py) and a sample at or above the task's target; anything else is INDICATIVE, and below the
target the task is INSUFFICIENT_SAMPLE whatever its numbers look like.
"""
from collections import Counter

from .metrics import prf, ratio

EXPERT_GOLD = 'EXPERT_GOLD'
DEVELOPMENT = 'DEVELOPMENT'
LABEL_STATUSES = (EXPERT_GOLD, DEVELOPMENT)
# The locked expert-gold release gate asks every critical task for at least this many reviewed examples.
MIN_TARGET = 100


def one_vs_rest(rows, labels) -> dict:
    """rows: [(expected, predicted)] -> per label: tp/fp/fn/tn, P/R/F1, false positive and false negative rate, support."""
    rows = [(e, p) for e, p in rows if e in labels]
    out = {}
    for label in labels:
        tp = sum(1 for e, p in rows if e == label and p == label)
        fp = sum(1 for e, p in rows if e != label and p == label)
        fn = sum(1 for e, p in rows if e == label and p != label)
        tn = len(rows) - tp - fp - fn
        out[label] = {**prf(tp, fp, fn), 'tn': tn, 'support': tp + fn,
                      'false_positive_rate': ratio(fp, fp + tn) if rows else None,
                      'false_negative_rate': ratio(fn, fn + tp)}
    return out


def unsupported_reasons(decision: dict, sources: dict) -> list[str]:
    """Why a conclusion is not supported ([] when it is).

    decision: {'status', 'stage', 'source': {'regulation_id', 'version_id', 'provision_ref', 'quote'},
    'facts': [{'path', 'value'}]}; sources: {'<regulation_id>@<version_id>': stored text}. UNKNOWN concludes
    nothing and is never unsupported. A ROUTED decision rests on pack metadata only: NOT_GROUNDED.
    """
    if decision.get('status') == 'UNKNOWN':
        return []
    if decision.get('stage') == 'ROUTED':
        return ['NOT_GROUNDED']
    reasons = []
    source = decision.get('source') or {}
    if not source.get('provision_ref'):
        reasons.append('NO_PROVISION')
    text = sources.get(f"{source.get('regulation_id')}@{source.get('version_id')}")
    if text is None:
        reasons.append('SOURCE_NOT_STORED')
    elif not source.get('quote') or source['quote'] not in text:
        reasons.append('QUOTE_NOT_IN_SOURCE')
    facts = decision.get('facts') or []
    if any(not f.get('path') or f.get('value') is None for f in facts):
        reasons.append('FACT_UNRESOLVED')
    if any(f.get('value') == 'UNKNOWN' for f in facts):
        reasons.append('FACT_UNKNOWN')
    return reasons


def task_report(task, rows, labels, target, label_status, positive=None, positive_target=None, unsupported=None,
                abstain='UNKNOWN') -> dict:
    """One task's report. rows: [(expected, predicted)]; unsupported: per row, its unsupported_reasons (aligned).

    overconfident counts answers where the gold is the abstain label and the prediction is not: a NO (or YES)
    the evidence did not allow. It is reported next to, not inside, the decision-level unsupported count.
    """
    if target < MIN_TARGET:
        raise ValueError(f'{task}: a task target is at least {MIN_TARGET} reviewed examples')
    if label_status not in LABEL_STATUSES:
        raise ValueError(f'{task}: label_status is one of {LABEL_STATUSES}')
    kept = [(i, e, p) for i, (e, p) in enumerate(rows) if e in labels]
    pairs = [(e, p) for _, e, p in kept]
    per_class = one_vs_rest(pairs, labels)
    present = [l for l in labels if per_class[l]['support']]
    macro = {key: (round(sum(per_class[l][key] or 0 for l in present) / len(present), 4) if present else None)
             for key in ('precision', 'recall', 'f1')}
    positives = sum(1 for e, _ in pairs if e == positive) if positive is not None else None
    short = len(pairs) < target or (positive_target is not None and (positives or 0) < positive_target)
    status = 'INSUFFICIENT_SAMPLE' if short else 'SUFFICIENT'
    reasons = [unsupported[i] for i, _, _ in kept] if unsupported is not None else []
    has_abstain = abstain in labels
    return {'task': task, 'n': len(pairs), 'target': target, 'positives': positives, 'positive_target': positive_target,
            'label_status': label_status, 'sample_status': status,
            'claim': 'ACCURACY' if label_status == EXPERT_GOLD and status == 'SUFFICIENT' else 'INDICATIVE',
            'accuracy': ratio(sum(1 for e, p in pairs if e == p), len(pairs)),
            'per_class': per_class, 'macro': macro,
            'abstention_rate': ratio(sum(1 for _, p in pairs if p == abstain), len(pairs)) if has_abstain else None,
            'overconfident': sum(1 for e, p in pairs if e == abstain and p != abstain) if has_abstain else None,
            'unsupported_conclusions': sum(1 for r in reasons if r),
            'unsupported_by_reason': dict(Counter(code for r in reasons for code in set(r)))}


def pair_consistency(pairs) -> dict:
    """pairs: [((expected_a, predicted_a), (expected_b, predicted_b))] of contrast cases (same regulation, two
    companies; same company, two products ...). A pair is consistent when both sides match gold; a contrast
    pair (gold differs) is detected when it is consistent."""
    consistent = [((ea, pa), (eb, pb)) for (ea, pa), (eb, pb) in pairs if ea == pa and eb == pb]
    contrast = [pair for pair in pairs if pair[0][0] != pair[1][0]]
    return {'n': len(pairs), 'consistent': len(consistent), 'consistency': ratio(len(consistent), len(pairs)),
            'contrast_pairs': len(contrast), 'contrast_detected': sum(1 for pair in contrast if pair in consistent)}
