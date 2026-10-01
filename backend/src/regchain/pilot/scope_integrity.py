"""Applicability source lineage and explicit evidence for sharing a scope answer.

No gold labels or company-specific rules are used here. A source identifier must
belong to the original candidate set and to the request currently being checked.
"""
from copy import deepcopy
from hashlib import sha256
import re


PROFILE_FIELDS = ('jurisdictions', 'activities', 'licences', 'products', 'customer_types')


class ScopeDiagnostics(list):
    """Warnings stay a normal list; a successful call also exposes its audit trace."""
    decision_trace = None


def completeness(company, provision):
    missing = [key for key in PROFILE_FIELDS if getattr(company, key, None) is None]
    if missing:
        return {'code': 'PROFILE_INCOMPLETE', 'missing': ['Missing company field: ' + key for key in missing],
                'reason': 'Company profile is incomplete: ' + ', '.join(missing)}
    if not isinstance(provision, str) or not provision.strip():
        return {'code': 'SOURCE_INCOMPLETE', 'missing': ['Full provision text is unavailable.'],
                'reason': 'The provision has no source text; applicability cannot be established.'}
    return None


def payload_sources(payload):
    rows = [payload.get('provision') or {}, *(payload.get('scope') or [])]
    return {row['source_id']: row.get('text') or '' for row in rows if row.get('source_id')}


def source_snapshot(sources):
    return [{'source_id': sid, 'text_sha256': sha256(text.encode('utf-8')).hexdigest(), 'characters': len(text)}
            for sid, text in sources.items()]


class SourceLineage:
    """Request membership is separate from full-text quote normalization."""
    def __init__(self, sources):
        self.sources = dict(sources)
        self.original = None
        self.requests = []
        self.first_verified = None
        self.previous_verified = set()
        self.protected = set()

    def admitted(self, payload, attempt, notes):
        sent = payload_sources(payload)
        if self.original is None:
            self.original = source_snapshot(sent)
        before = set(row['source_id'] for row in self.requests[-1]['source_set']) if self.requests else set(self.sources)
        present = set(sent)
        removed, added = sorted(before - present), sorted(present - before)
        unknown = sorted(present - set(self.sources))
        record = {'attempt': attempt, 'phase': 'repair' if self.requests else 'original', 'source_set': source_snapshot(sent),
                  'removed_source_ids': removed, 'added_source_ids': added, 'invalid_source_ids': unknown,
                  'validation_source_ids': sorted(present & set(self.sources))}
        self.requests.append(record)
        if removed:
            notes.append({'attempt': attempt, 'code': 'BASIS_DROPPED', 'source_ids': removed,
                          'detail': 'Source IDs removed from the request; their original texts cannot validate this answer.'})
        if added:
            notes.append({'attempt': attempt, 'code': 'BASIS_ADDED', 'source_ids': added,
                          'detail': 'A repair can reintroduce only source IDs from the original candidate set.'})
        if unknown:
            notes.append({'attempt': attempt, 'code': 'BASIS_INVALID', 'source_ids': unknown,
                          'detail': 'The request contains source IDs outside the original candidate set.'})
        # A retained source may be clipped in the prompt; exact spans still resolve
        # against its immutable full text. A removed ID has no such privilege.
        return {sid: text for sid, text in self.sources.items() if sid in present and sid not in unknown}

    def verified(self, basis, quotes, attempt, notes):
        ids = {row.source_id for row in basis} | {row.source_id for row in quotes}
        if self.first_verified is None:
            self.first_verified = set(ids)
        if self.requests and self.requests[-1]['phase'] == 'repair':
            lost, added = sorted(self.previous_verified - ids), sorted(ids - self.previous_verified)
            for code, changed in (('BASIS_DROPPED', lost), ('BASIS_ADDED', added)):
                if changed:
                    notes.append({'attempt': attempt, 'code': code, 'source_ids': changed,
                                  'detail': 'Verified evidence source membership changed during repair.'})
            notes.append({'attempt': attempt, 'code': 'BASIS_REPAIRED', 'source_ids': sorted(ids),
                          'detail': 'Repair evidence was resolved against the source IDs admitted for this request.'})
        self.previous_verified = ids
        self.protected.update(ids)
        if self.requests:
            self.requests[-1]['verified_evidence_source_ids'] = sorted(ids)

    def snapshot(self, final_basis=(), final_quotes=()):
        return {'candidate_source_set': source_snapshot(self.sources), 'original_source_set': deepcopy(self.original or []),
                'repaired_source_set': deepcopy(self.requests[-1]['source_set']) if len(self.requests) > 1 else None,
                'requests': deepcopy(self.requests),
                'original_verified_source_ids': sorted(self.first_verified or []),
                'last_request_validation_source_set': self.requests[-1]['validation_source_ids'] if self.requests else [],
                'final_validated_source_set': sorted({x.source_id for x in final_basis} | {x.source_id for x in final_quotes})}


def local_scope(candidate, clause):
    """Retain source text, not a guessed actor recovered from another subsection."""
    subject = str(candidate.get('subject') or '').strip()
    text = str(clause.get('text') or '')
    match = re.search(re.escape(subject), text, re.IGNORECASE) if subject else None
    return {'subject': subject, 'actor_quote': match.group(0) if match else None,
            'actor_span': [clause.get('offset', 0) + match.start(), clause.get('offset', 0) + match.end()] if match else None,
            'clause_text': text, 'conditions': list(candidate.get('conditions') or []),
            'exceptions': list(candidate.get('exceptions') or [])}


def local_assessment_evidence(origin, current, chain, company, clause_gate, sources):
    """Permission to ask a local question, never permission to inherit its answer.

    A different subsection actor can independently match the stated company. The
    profile value, list quotation and local actor occurrence must all be inspectable;
    a parent APPLIES or a generic MATCH label alone supplies none of this evidence.
    ``sources`` contains canonical IDs and the existing candidate source texts.
    """
    result = {'allowed': False, 'reason_code': 'LOCAL_ASSESSMENT_UNSUPPORTED', 'support': [],
              'required_evidence': []}
    if chain.decision != 'OPEN' or completeness(company, current.get('clause_text')):
        return result
    origin = origin or {}
    if (origin.get('actor_quote') and current.get('actor_quote') and
            origin.get('subject', '').casefold() == current.get('subject', '').casefold()):
        result.update(allowed=True, reason_code='SAME_EXPLICIT_ACTOR_LOCAL_ASSESSMENT')
        result['support'].append({'kind': 'SAME_EXPLICIT_ACTOR', 'actor_quote': current['actor_quote']})
        return result
    actor = chain.by_name('COMPANY_ENTITY') or {}
    subject = chain.by_name('REGULATION_SUBJECT_SCOPE') or {}
    if not (actor.get('status') == subject.get('status') == 'MATCH' and actor.get('clear') and subject.get('clear')):
        return result
    evidence = subject.get('evidence') or {}
    field, value = evidence.get('company_field'), evidence.get('company_value')
    values = getattr(company, field, None) if field in PROFILE_FIELDS else None
    if not isinstance(values, (list, tuple)) or not value or value not in values:
        return result
    sid, quote = evidence.get('source_id'), evidence.get('quote')
    if not sid or not quote or quote not in sources.get(sid, ''):
        return result
    local_actors = []
    for row in clause_gate.get('required_entities', []):
        if row.get('role') != 'obliged_party' or row.get('match') != 'MATCH' or not row.get('text'):
            continue
        found = re.search(re.escape(row['text']), current['clause_text'], re.IGNORECASE)
        if found:
            local_actors.append({'kind': 'MATCHED_LOCAL_ACTOR', 'entity_type': row.get('type'),
                                 'quote': found.group(0), 'clause_span': [found.start(), found.end()],
                                 'role': row['role'], 'reading_source': row.get('source')})
    if not local_actors and chain.obliged_addressee and current.get('actor_quote'):
        local_actors.append({'kind': 'EXPLICIT_OBLIGED_ACTOR', 'quote': current['actor_quote']})
    if not local_actors:
        return result
    definitions = []
    for definition in (actor.get('evidence') or {}).get('definitions', []):
        definition_id, definition_quote = definition.get('source_id'), definition.get('quote')
        if not definition_id or not definition_quote or definition_quote not in sources.get(definition_id, ''):
            return result
        definitions.append({'kind': 'LOCAL_ACTOR_DEFINITION', 'source_id': definition_id,
                            'quote': definition_quote, 'category': definition.get('category')})
    result['support'] = [*local_actors, {'kind': 'SOURCE_BOUND_PROFILE_LIST_MATCH', 'source_id': sid,
                                       'quote': quote, 'company_field': field, 'company_value': value}, *definitions]
    result['required_evidence'] = [{'source_id': sid, 'quote': quote},
                                   *({'source_id': row['source_id'], 'quote': row['quote']} for row in definitions)]
    result.update(allowed=True, reason_code='INDEPENDENT_LOCAL_ACTOR_MATCH')
    return result


def inheritance_evidence(origin, current, chain, basis, obliged, financial_fact=''):
    """A provision answer is reusable only when this clause has actor/scope support.

    A verified local basis can carry a changed exception/condition. An explicit
    regulated business fact is retained as evidence, not inferred from a company
    name, product word or incidental transaction.
    """
    subject = chain.by_name('REGULATION_SUBJECT_SCOPE') or {}
    actor = chain.by_name('COMPANY_ENTITY') or {}
    evidence = {'original_local_scope': deepcopy(origin), 'current_local_scope': deepcopy(current),
                'shared_answer': origin is not None, 'support': [], 'reason_code': None}
    if origin is None:
        evidence.update(allowed=True, reason_code='LOCAL_MODEL_REQUEST')
        return evidence
    changed_exceptions = origin['exceptions'] != current['exceptions']
    changed_conditions = origin['conditions'] != current['conditions']
    evidence.update(exception_changed=changed_exceptions, condition_changed=changed_conditions,
                    explicit_local_subject=bool(current['actor_quote']))
    local_basis = [b for b in basis if b.match == 'YES' and b.regulatory_condition and
                   b.regulatory_condition in current['clause_text']]
    if local_basis:
        evidence['support'].append({'kind': 'VERIFIED_LOCAL_BASIS', 'source_ids': sorted({b.source_id for b in local_basis}),
                                    'quotes': [b.regulatory_condition for b in local_basis]})
    changed_exception_basis = [b for b in local_basis if any(
        exception and (exception in b.regulatory_condition or b.regulatory_condition in exception)
        for exception in current['exceptions'])]
    if changed_exceptions and not changed_exception_basis:
        evidence.update(allowed=False, reason_code='SCOPE_INCOMPLETE')
        return evidence
    if current['clause_text'] == origin['clause_text'] and current['subject'] == origin['subject']:
        evidence['support'].append({'kind': 'SAME_LOCAL_SCOPE', 'actor_quote': current['actor_quote']})
    if actor.get('evidence', {}).get('reason_code') == 'UNIVERSAL_ADDRESSEE':
        evidence['support'].append({'kind': 'EXPLICIT_UNIVERSAL_ACTOR', 'evidence': deepcopy(actor['evidence'])})
    origin_actor = origin.get('actor_evidence') or {}
    # An explicit reflexive/demonstrative recipient can refer back to a universal
    # actor. An unrelated new named actor cannot use this route.
    reference = re.match(r'^(?:kendisinden|kendilerinden|kendilerine|bunlar\b|bu (?:kişiler|kuruluşlar)\b|those (?:persons|parties)\b)',
                         current['subject'], re.IGNORECASE)
    if current['actor_quote'] and reference and origin_actor.get('evidence', {}).get('reason_code') == 'UNIVERSAL_ADDRESSEE':
        evidence['support'].append({'kind': 'EXPLICIT_ACTOR_REFERENCE', 'local_reference_quote': reference.group(0),
                                    'parent_actor_evidence': deepcopy(origin_actor['evidence'])})
    if chain.clear_match:
        evidence['support'].append({'kind': 'LOCAL_RULE_MATCH', 'evidence': deepcopy(actor.get('evidence', {}))})
    if chain.obliged_addressee:
        if subject.get('status') == 'MATCH' and subject.get('clear'):
            evidence['support'].append({'kind': 'EXPLICIT_OBLIGED_LIST_MATCH', 'evidence': deepcopy(subject.get('evidence', {}))})
        listed = [b for b in basis if b.match == 'YES' and obliged is not None and b.source_id == obliged.source_id]
        if listed:
            evidence['support'].append({'kind': 'VERIFIED_OBLIGED_LIST_BASIS', 'source_ids': sorted({b.source_id for b in listed}),
                                        'quotes': [b.regulatory_condition for b in listed]})
        if financial_fact:
            evidence['support'].append({'kind': 'EXPLICIT_REGULATED_BUSINESS_FACT', 'company_fact': financial_fact,
                                        'local_actor_quote': current['actor_quote']})
    same_actor = bool(current['actor_quote'] and origin['actor_quote'] and
                      current['subject'].casefold() == origin['subject'].casefold())
    if same_actor and not changed_conditions:
        evidence['support'].append({'kind': 'SAME_EXPLICIT_ACTOR', 'actor_quote': current['actor_quote']})
    # Every listed support item is independently inspectable; parent applicability
    # alone is deliberately absent from the allowed routes.
    evidence.update(allowed=bool(evidence['support']),
                    reason_code='INHERITANCE_SUPPORTED' if evidence['support'] else 'SCOPE_INCOMPLETE')
    return evidence
