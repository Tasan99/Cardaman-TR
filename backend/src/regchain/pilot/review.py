from datetime import datetime, timezone

from regchain.evidence import digest, make_event, verify_chain
from .engine import validate_evidence
from .schema import Review


def apply_review(packet, review: Review):
    if not verify_chain(packet['events'], packet['head'], packet['count']):
        raise ValueError('Input packet was modified')
    if packet['count'] != 1:
        raise ValueError('Review the original analysis; each review creates a separate retained artifact')
    if review.analysis_head != packet['head']:
        raise ValueError('Review belongs to a different analysis or source/company/policy version')
    analysis = packet['events'][0]['payload']
    by_id = {o['id']: o for o in analysis['obligations']}
    if set(d.obligation_id for d in review.decisions) != set(by_id):
        raise ValueError('Review must cover every extracted candidate exactly once')
    scopes = {s['id']: s for s in analysis['scope_sources']}
    # v0.18: the analysed provisions themselves are citable too (a rule decision quotes the clause it read,
    # the judge quotes the provision as p0); a quote must still be an exact substring of its source.
    scopes.update({c['source']['id']: c['source'] for c in analysis.get('cases', []) if c.get('source', {}).get('id') not in scopes})
    policies = {c['source_id']: c for p in analysis['policies'] for c in p['chunks']}
    outcomes, audit = [], []
    created_at = datetime.now(timezone.utc).isoformat()
    for decision in review.decisions:
        validate_evidence(decision, analysis['company'], scopes, policies)
        ai = by_id[decision.obligation_id]['proposal']
        # The audit entry pairs what the AI proposed with what the person decided. The
        # analysis event itself is never edited: this review is appended after it.
        same = lambda field: getattr(decision, field) == ai[field] or (field == 'applicability' and decision.applicability == 'UNKNOWN'
                                                                      and ai[field] in ('UNKNOWN', 'POSSIBLY_APPLIES'))
        changed = [field for field in ('applicability', 'coverage') if not same(field)]
        if decision.action == 'APPROVE' and changed:
            raise ValueError(f'APPROVE keeps the AI proposal as it is; {", ".join(changed)} differs, so record an OVERRIDE with its reason')
        entry = {'obligation_id': decision.obligation_id, 'action': decision.action, 'reviewer': review.reviewer,
                 'reviewer_role': review.reviewer_role, 'at': created_at,
                 'ai_proposal': {'applicability': ai['applicability'], 'coverage': ai['coverage']},
                 'human_decision': {'extraction': decision.extraction, 'applicability': decision.applicability,
                                    'coverage': decision.coverage, 'remediation': decision.remediation},
                 'changed_fields': changed, 'override_reason': decision.override_reason or None}
        # v0.18: which gate or rule settled the proposal and the flags that sent it to a person, as the
        # reviewer saw them on the card. Proposals before v0.18 have no decided_by and record None.
        trace = ai.get('trace') or {}
        if trace.get('decided_by') or ai.get('review_flags'):
            entry['ai_trace'] = {'decided_by': trace.get('decided_by'), 'review_flags': list(ai.get('review_flags') or [])}
        audit.append(entry)
        if decision.action == 'NEEDS_EVIDENCE':
            status = 'MORE_EVIDENCE_REQUIRED'
        elif decision.extraction == 'REJECT':
            status = 'EXTRACTION_REJECTED'
        elif decision.extraction == 'UNCERTAIN':
            status = 'EXTRACTION_REVIEW_REQUIRED'
        elif decision.applicability == 'UNKNOWN':
            status = 'APPLICABILITY_REVIEW_REQUIRED'
        elif decision.applicability == 'DOES_NOT_APPLY':
            status = 'REVIEWED_NOT_APPLICABLE'
        elif decision.coverage == 'COVERS_TEXT':
            status = 'REVIEWED_POLICY_COVERAGE'
        elif decision.coverage in ('PARTIAL', 'CONFLICT'):
            status = 'REVIEWED_POLICY_GAP'
        else:
            status = 'POLICY_REVIEW_REQUIRED'
        outcomes.append({'obligation_id': decision.obligation_id, 'status': status})
    payload = {'kind': 'conc-pilot-review-v1', 'created_at': created_at,
        'review': review.model_dump(), 'review_hash': digest(review.model_dump()), 'outcomes': outcomes,
        # v0.17: one audit entry per decision (action, AI proposal, human decision, override reason);
        # v0.18 adds the AI's decided_by and review flags.
        'audit': audit, 'ai_output_preserved': 'The analysis event is unchanged; decisions are appended as this event.',
        'identity_assurance': 'LOCAL_OPERATOR_ASSERTION',
        'limitations': 'Reviewer identity and expertise are not independently authenticated. Policy evidence does not prove operational compliance.'}
    event = make_event(payload, packet['head'])
    return {**packet, 'events': [*packet['events'], event], 'head': event['event_hash'], 'count': packet['count']+1}
