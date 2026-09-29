"""Three-way, source-bound anchor audit; no model call and no semantic synonym list.

MATCH means a same-action/object anchor was observed, not that a contradiction was
proved. MISMATCH needs positive evidence. Unread wording remains UNRESOLVED rather
than becoming a fabricated assertion that the sentence concerns another duty.
"""
import re

from . import conflict
from .entities import OBLIGED, found

DIMENSIONS = ('actor', 'action', 'object', 'recipient', 'polarity', 'condition', 'scope', 'exception', 'normative_effect')
EN_SUBJECT = re.compile(r'^\s*(?:No\s+)?(?P<subject>[^.;:!?]{1,140}?)\s+'
                        r'(?:must|shall|should|may|cannot|can|need\s+not|is\s+required\s+to|are\s+required\s+to)\b', re.I)
COMPLEX_NEGATION = re.compile(r'\b(?:not\s+(?:only|necessarily|always|impossible|unlikely)|'
                             r'cannot\s+(?:rule\s+out|exclude)|may\s+not\s+always|no\s+longer\s+necessarily)\b', re.I)
GUARD = re.compile(r'\b(?:if|unless|except|provided\s+that|only\s+when)\b|'
                   r'(?<!\w)(?:halinde|hâlinde|şartıyla|koşuluyla|hariç|istisna\w*|\w+(?:madıkça|medikçe))(?!\w)', re.I)
QUOTED_ID = re.compile(r'(["“])([^"”\n]+)["”]')
BROAD_ACTORS = {'GENERIC', 'FINANCIAL_GENERIC', 'DNFBP_GENERIC', 'FOREIGN_HQ_OBLIGED'}


def _text(value):
    if isinstance(value, dict):
        return str(value.get('text') or value.get('name') or '')
    if isinstance(value, (list, tuple)):
        return ' '.join(_text(item) for item in value)
    return str(value or '')


def _field(duty, name, kinds=()):
    direct = _text(duty.get(name))
    elements = [_text(e.get('text')) for e in duty.get('elements') or [] if isinstance(e, dict) and e.get('kind') in kinds]
    return ' '.join(filter(None, [direct, *elements]))


def _locations(text, wanted):
    if not wanted:
        return []
    result, start = [], 0
    while (at := text.find(wanted, start)) >= 0:
        result.append({'start': at, 'end': at + len(wanted), 'text': wanted})
        start = at + max(1, len(wanted))
    return result


def _dimension(status='UNRESOLVED', reason='Not independently resolved from source wording', **facts):
    return {'status': status, 'reason': reason, **facts}


def _actor_subject(text):
    match = EN_SUBJECT.match(text)
    if match:
        return match.group('subject').strip()
    # A Turkish subject before an explicit comma is usable only as an entity
    # phrase; no scan of the whole sentence that could confuse objects with actors.
    head = text.split(',', 1)[0].strip() if ',' in text else ''
    return head if head and any(found(OBLIGED, head)) else ''


def _actor_types(subject):
    # English sentence-initial I must not become Turkish dotless ı in fold().
    return ({kind for kind, _ in found(OBLIGED, subject)} |
            {kind for kind, _ in found(OBLIGED, subject.lower())}) - BROAD_ACTORS


def _explicit_english_predicate(text):
    subject = EN_SUBJECT.match(text)
    if not subject:
        return set()
    verb = re.match(r'(?:\s+(?:not|never|also|\w+ly))*\s+([A-Za-z-]+)\b', text[subject.end():], re.I)
    return {key for key in conflict.stems(verb.group(1)) if key.startswith('@')} if verb else set()


def _explicit_polarity(sentence, keys):
    if COMPLEX_NEGATION.search(sentence):
        return set()
    observed = conflict.polarities(sentence, keys) if keys else set()
    # Unlisted English verbs can still have an explicit governing modal. This
    # reads the grammar, not a benchmark-specific action or a paraphrase pair.
    if not observed and EN_SUBJECT.match(sentence):
        if re.search(r'^\s*no\b.{0,140}\b(?:may|can)\b|\b(?:must|shall|may|can)\s+not\b|\bcannot\b', sentence, re.I):
            observed = {'PROHIBITED'}
        elif re.search(r'\b(?:may|can)\b', sentence, re.I):
            observed = {'PERMITTED'}
        elif re.search(r'\b(?:must|shall|should|is required to|are required to)\b', sentence, re.I):
            observed = {'REQUIRED'}
    return observed


def assess_anchor(duty: dict, span: str, passage: str) -> dict:
    """Return source binding, nine dimension audits and MATCH/MISMATCH/UNRESOLVED.

    Exact source text is required. This API has no source-ID argument and cannot
    distinguish two documents with identical text; callers must bind passage ID.
    """
    dimensions = {name: _dimension() for name in DIMENSIONS}
    result = {'schema_version': 'cardaman-normative-anchor-v1', 'status': 'UNRESOLVED',
              'reason_code': 'ANCHOR_UNRESOLVED', 'reason': '', 'dimensions': dimensions,
              'source_span': None, 'context_spans': [], 'source_occurrences': [],
              'dependency': {'used': False, 'status': 'NOT_USED'},
              'semantic_entailment_verified': False,
              'limitations': ['MATCH is an anchor, not a conflict verdict; source identity must be bound by the caller.']}
    if not isinstance(span, str) or not span.strip() or not isinstance(passage, str) or not passage.strip():
        result.update(reason_code='EMPTY_SOURCE_OR_SPAN', reason='Non-empty passage and exact evidence span are required')
        return result
    locations = _locations(passage, span)
    result['source_occurrences'] = locations
    if not locations:
        result.update(status='MISMATCH', reason_code='WRONG_SOURCE', reason='The evidence span is not an exact substring of this passage')
        return result
    if len(locations) != 1:
        result.update(reason_code='AMBIGUOUS_SPAN_LOCATION', reason='Repeated evidence text has no unique source offset')
        return result
    result['source_span'] = locations[0]
    # Root owns this helper; use it dynamically so adjacency policy has one owner.
    sentence = conflict.quoted_sentences(span, passage)
    context = conflict.evidence_context(span, passage)
    sentence_locations = _locations(passage, sentence)
    current = next((item for item in sentence_locations if item['start'] <= locations[0]['start'] and item['end'] >= locations[0]['end']), None)
    if current is None:
        result.update(reason_code='CONTEXT_NOT_SOURCE_BOUND', reason='Expanded sentence could not be bound to the exact input source')
        return result
    result['context_spans'].append(current)
    used_dependency = context != sentence
    if used_dependency:
        prefix = context[:-len(sentence)].rstrip() if context.endswith(sentence) else ''
        before = [item for item in _locations(passage, prefix) if item['end'] <= current['start']]
        if not before or not conflict.BACK_REFERENCE.search(conflict.fold(sentence)):
            result.update(reason_code='DEPENDENCY_NOT_SOURCE_BOUND', reason='Adjacent context lacks a source-bound explicit backward reference')
            return result
        result['context_spans'].insert(0, before[-1])
        result['dependency'] = {'used': True, 'status': 'EXPLICIT_BACK_REFERENCE',
                                'reason': 'An explicit backward reference licenses the immediately preceding source sentence'}
    safe_duty = dict(duty or {})
    for name in ('subject', 'required_action', 'prohibited_action', 'source_sentence', 'object', 'recipient'):
        safe_duty[name] = _text(safe_duty.get(name))
    safe_duty['conditions'] = [_text(x) for x in safe_duty.get('conditions') or []]
    safe_duty['exceptions'] = [_text(x) for x in safe_duty.get('exceptions') or []]
    keys, polarity = conflict.gate_acts(safe_duty)
    action_keys = {key for key in conflict.verb_keys(safe_duty) if key not in conflict.GENERIC}
    said_keys = conflict.stems(context)
    action_shared = action_keys & said_keys
    object_text = _field(safe_duty, 'object', ('object', 'item'))
    object_keys = conflict._content_keys(object_text) if object_text else conflict._object_keys(safe_duty, keys)
    object_shared = object_keys & conflict._content_keys(context)
    actor = _field(safe_duty, 'subject', ('subject', 'actor'))
    said_actor = _actor_subject(sentence)
    if not said_actor and used_dependency:
        said_actor = _actor_subject(result['context_spans'][0]['text'])
    mine, theirs = _actor_types(actor), _actor_types(said_actor)
    if mine and theirs and not mine & theirs:
        dimensions['actor'] = _dimension('MISMATCH', 'Both explicit subjects name different recognized obliged-party types', duty=actor, policy=said_actor)
    elif mine & theirs or (actor and said_actor and conflict.fold(actor) == conflict.fold(said_actor)):
        dimensions['actor'] = _dimension('MATCH', 'The explicit subjects share an actor type or exact wording', duty=actor, policy=said_actor)
    else:
        dimensions['actor'] = _dimension(duty=actor, policy=said_actor)
    # Only the helper that requires known predicates on both sides supplies a
    # mismatch; absent transfer words or an unknown verb cannot reject the source.
    foreign = conflict._foreign_action(safe_duty, keys, context)
    if not foreign and not action_shared:
        # The old predicate helper is largely Turkish. A governing English modal
        # plus an immediately following known verb is explicit predicate evidence.
        mine_acts = {key for key in conflict.verb_keys(safe_duty) if key.startswith('@')}
        policy_acts = _explicit_english_predicate(sentence)
        family = lambda acts: {conflict.ACT_FAMILY.get(key, key) for key in acts}
        if mine_acts and policy_acts and not family(mine_acts) & family(policy_acts):
            foreign = 'Both explicit predicates are recognized actions of different families: ' + ', '.join(sorted(mine_acts)) + ' / ' + ', '.join(sorted(policy_acts))
    if foreign:
        dimensions['action'] = _dimension('MISMATCH', foreign)
    elif action_shared:
        dimensions['action'] = _dimension('MATCH', 'Shared normalized action representation', shared_keys=sorted(action_shared))
    else:
        dimensions['action'] = _dimension(reason='Action wording is not resolved; absence of a known verb is not mismatch')
    if object_shared:
        dimensions['object'] = _dimension('MATCH', 'The duty object has source-grounded content overlap', shared_keys=sorted(object_shared), duty=object_text)
    else:
        # Distinct explicit quoted identifiers are positive object evidence;
        # disjoint ordinary words may be synonyms and remain UNRESOLVED.
        own_ids = {m.group(2) for m in QUOTED_ID.finditer(object_text)}
        other_ids = {m.group(2) for m in QUOTED_ID.finditer(sentence)}
        if own_ids and other_ids and not own_ids & other_ids and action_shared:
            dimensions['object'] = _dimension('MISMATCH', 'The same action addresses different explicit quoted object identifiers',
                                               duty_identifiers=sorted(own_ids), policy_identifiers=sorted(other_ids))
        else:
            dimensions['object'] = _dimension(duty=object_text)
    # Do not let a shared generic noun override distinct explicit identifiers.
    own_ids = {m.group(2) for m in QUOTED_ID.finditer(object_text)}
    other_ids = {m.group(2) for m in QUOTED_ID.finditer(sentence)}
    if own_ids and other_ids and not own_ids & other_ids and action_shared:
        dimensions['object'] = _dimension('MISMATCH', 'The same action addresses different explicit quoted object identifiers',
                                           duty_identifiers=sorted(own_ids), policy_identifiers=sorted(other_ids))
    own_recipient = _field(safe_duty, 'recipient', ('recipient',))
    duty_recipient = conflict.recipient(own_recipient) or conflict.duty_role(safe_duty)['recipient']
    said_recipient = conflict.communication_recipient(sentence)
    if duty_recipient and said_recipient:
        dimensions['recipient'] = _dimension('MATCH' if duty_recipient == said_recipient else 'MISMATCH',
            'Both communication recipients are explicitly readable', duty=duty_recipient, policy=said_recipient)
    own_groups, said_groups = conflict._scope_groups(safe_duty), conflict._party_groups(conflict.fold(sentence))
    if own_groups and said_groups:
        dimensions['scope'] = _dimension('MATCH' if own_groups & said_groups else 'MISMATCH',
            'Both sides name explicit customer groups', duty=sorted(own_groups), policy=sorted(said_groups))
    for name, field in (('condition', 'conditions'), ('exception', 'exceptions')):
        own = safe_duty[field]
        if own and all(piece and conflict.fold(piece) in conflict.fold(context) for piece in own):
            dimensions[name] = _dimension('MATCH', 'All supplied restrictions are present in the source context', duty=own)
        else:
            dimensions[name] = _dimension(duty=own, reason='Absence of an exact restriction does not prove incompatible scope')
    observed = _explicit_polarity(sentence, action_keys)
    if observed:
        dimensions['polarity'] = _dimension('MATCH', 'Source polarity is readable; MATCH does not mean same polarity',
                                             duty=polarity or None, policy=sorted(observed))
        incompatible = set(conflict.INCOMPATIBLE.get(polarity, ()))
        comparison = ('INCOMPATIBLE' if observed <= incompatible else 'SAME_DIRECTION' if observed == {polarity} else 'UNRESOLVED')
        dimensions['normative_effect'] = _dimension('MATCH' if comparison != 'UNRESOLVED' else 'UNRESOLVED',
            'Observed polarity comparison only; condition/exception scope still requires assessment', comparison=comparison)
    mismatch = [name for name in ('actor', 'action', 'object', 'recipient', 'scope') if dimensions[name]['status'] == 'MISMATCH']
    if mismatch:
        result.update(status='MISMATCH', reason_code='EXPLICIT_' + mismatch[0].upper() + '_MISMATCH',
                      reason='; '.join(dimensions[name]['reason'] for name in mismatch))
    elif COMPLEX_NEGATION.search(sentence):
        result.update(reason_code='NEGATIVE_QUALIFIER_UNRESOLVED', reason='A negative qualifier prevents a reliable normative reading')
    elif GUARD.search(sentence) and not used_dependency and any(safe_duty[field] for field in ('conditions', 'exceptions')) \
            and dimensions['condition']['status'] != 'MATCH' and dimensions['exception']['status'] != 'MATCH':
        result.update(reason_code='CONDITION_SCOPE_UNRESOLVED', reason='Guarded wording and duty restrictions are not aligned')
    elif action_shared or object_shared:
        result.update(status='MATCH', reason_code='SOURCE_BOUND_ACTION_OR_OBJECT',
                      reason='The exact source context anchors the duty action or object; unread dimensions remain explicit')
    else:
        result.update(reason_code='SEMANTIC_ANCHOR_UNRESOLVED', reason='No positive mismatch is proven, but no comparable action/object anchor is established')
    return result
