"""Regulatory change: changed clause -> changed obligation -> affected product, entity, facility, policy and control.

Two sources say that a clause changed, and both are official text, not a news summary:

  stored versions   two texts of one regulation in the corpus (corpus.py): clauses are paired by reference, or by wording
                    when a paragraph moved, and compared (version_changes);
  amendment notes   the consolidated text's own notes, "(Ek cümle:11/6/2026-7584/2 md.)": which clauses an instrument
                    added or changed since a date, from one stored version (note_changes). The old wording is not known.

A changed clause becomes a changed obligation by extracting both sides with the same rules (extraction.py) and comparing
their elements: modality, addressee, products, limits, conditions, exceptions. Each changed obligation is then routed on
the enterprise profile before and after, and joined with the policy register (compare.py): which targets it newly
reaches or leaves, which products, facilities and entities those are, which documents and controls speak about it, and
who has to act. A change that reaches no target of the profile is reported as NO_TARGET_AFFECTED, never dropped.

compare_versions / attach_targets are the hash-level records of the first milestone and stay as they were.
"""
import re
from datetime import date
from difflib import SequenceMatcher
from typing import Literal

from pydantic import Field

from ..evidence import digest
from ..pilot.schema import Strict
from .amendments import clause_amendments
from .clauses import Clause, split_clauses
from .corpus import CorpusStore
from .extraction import ExtractedObligation, extract_regulation, route
from .packs import Registry

ChangeKind = Literal['ADDED', 'REMOVED', 'MODIFIED']
ImpactDelta = Literal['NEWLY_APPLIES', 'NO_LONGER_APPLIES', 'CHANGED_REQUIREMENT', 'UNCHANGED', 'UNKNOWN',
                      'NO_TARGET_AFFECTED']
CHANGE_RULES_VERSION = 'tr-change-rules-v1'
MOVED_SIMILARITY = 0.6


class SourceVersion(Strict):
    regulation_id: str
    version_id: str
    content_hash: str = Field(min_length=1)
    previous_version_id: str | None = None


class ProvisionChange(Strict):
    provision_ref: str
    kind: ChangeKind
    old_hash: str | None = None
    new_hash: str | None = None
    impact_types: list[str] = []
    # Clause-level detail (empty on a hash-only record).
    old_ref: str | None = None
    old_text: str | None = None
    new_text: str | None = None
    basis: Literal['HASH_ONLY', 'VERSION_DIFF', 'AMENDMENT_NOTE'] = 'HASH_ONLY'
    instrument: str | None = None
    amended_on: date | None = None


class ObligationChange(Strict):
    """One duty before and after: what of it changed."""
    provision_ref: str
    kind: ChangeKind | Literal['WORDING_ONLY', 'NEW_OR_CHANGED']
    impact_types: list[str]
    old_obligation_id: str | None = None
    new_obligation_id: str | None = None
    element_changes: list[dict] = []
    topic: str = 'GENERAL'
    group: str = ''


class AffectedTarget(Strict):
    obligation_id: str
    decision_before: str | None = None
    decision_after: str | None = None
    delta: ImpactDelta
    entity_ids: list[str] = []
    facility_ids: list[str] = []
    product_ids: list[str] = []
    document_ids: list[str] = []
    control_ids: list[str] = []
    departments: list[str] = []
    actions: list[dict] = []
    # Filled by the clause-level impact.
    provision_ref: str | None = None
    level: str | None = None
    target_id: str | None = None
    coverage_after: str | None = None
    mapping_after: str | None = None


class ImpactRecord(Strict):
    change_id: str
    regulation_id: str
    from_version: str
    to_version: str
    hash_changed: bool
    provision_changes: list[ProvisionChange] = []
    affected: list[AffectedTarget] = []
    obligation_changes: list[ObligationChange] = []
    basis: Literal['HASH_ONLY', 'VERSION_DIFF', 'AMENDMENT_NOTE'] = 'HASH_ONLY'
    synthetic: bool = False
    rules_version: str = CHANGE_RULES_VERSION
    summary: dict = {}


# -- hash-level (milestone 1) ------------------------------------------------------------------------
def compare_versions(old: SourceVersion, new: SourceVersion) -> ImpactRecord:
    if old.regulation_id != new.regulation_id:
        raise ValueError(f'versions belong to different regulations: {old.regulation_id}, {new.regulation_id}')
    changed = old.content_hash != new.content_hash
    return ImpactRecord(change_id=f'{old.regulation_id}:{old.version_id}->{new.version_id}',
                        regulation_id=old.regulation_id, from_version=old.version_id, to_version=new.version_id,
                        hash_changed=changed,
                        provision_changes=[] if not changed else [ProvisionChange(provision_ref='UNRESOLVED',
                                                                                  kind='MODIFIED',
                                                                                  old_hash=old.content_hash,
                                                                                  new_hash=new.content_hash)])


def attach_targets(record: ImpactRecord, targets: list[AffectedTarget]) -> ImpactRecord:
    """Join changed provisions to the profile targets already linked to those obligations."""
    if not record.hash_changed:
        return record.model_copy(update={'affected': [AffectedTarget(obligation_id=t.obligation_id,
                                                                    delta='UNCHANGED',
                                                                    entity_ids=t.entity_ids, facility_ids=t.facility_ids,
                                                                    product_ids=t.product_ids)] for t in targets})
    if not targets:
        return record.model_copy(update={'affected': [AffectedTarget(obligation_id='NONE',
                                                                    delta='NO_TARGET_AFFECTED')]})
    return record.model_copy(update={'affected': targets})


# -- clause level ------------------------------------------------------------------------------------
def _clauses(sections: list[dict]) -> dict[str, list[Clause]]:
    """{printed article label: its clauses in order}."""
    out: dict[str, list[Clause]] = {}
    for section in sections:
        if 'DELETED_PROVISION' in section.get('quality_flags', ()):
            continue
        out[section['printed_label']] = split_clauses(section)
    return out


MARKER = re.compile(r'^\s*(?:\(\d{1,2}\)|[a-zçğıöşü]{1,2}\)|\d{1,2}\))\s+')


def _wording(clause: Clause) -> str:
    """The clause without its paragraph marker: "(4) Bu Tebliğin ..." and "Bu Tebliğin ..." are the same sentence when
    another sentence was put before it."""
    return MARKER.sub('', clause.text)


def clause_changes(old_sections: list[dict], new_sections: list[dict]) -> list[ProvisionChange]:
    """Clauses added, removed or reworded between two texts.

    The clauses of each article are aligned in order by their wording, so a sentence inserted before another one is one
    ADDED clause and the sentences after it, whose references moved, are unchanged. Inside a stretch that was replaced,
    an old and a new clause that share most of their wording are one MODIFIED clause; the rest is REMOVED and ADDED."""
    old, new = _clauses(old_sections), _clauses(new_sections)
    changes = []

    def added(clause):
        return ProvisionChange(provision_ref=clause.ref, kind='ADDED', new_text=clause.text, new_hash=digest(clause.text),
                               basis='VERSION_DIFF')

    def removed(clause):
        return ProvisionChange(provision_ref=clause.ref, old_ref=clause.ref, kind='REMOVED', old_text=clause.text,
                               old_hash=digest(clause.text), basis='VERSION_DIFF')

    def modified(before, after):
        return ProvisionChange(provision_ref=after.ref, old_ref=before.ref, kind='MODIFIED', old_text=before.text,
                               new_text=after.text, old_hash=digest(before.text), new_hash=digest(after.text), basis='VERSION_DIFF')

    for label in [*new, *(k for k in old if k not in new)]:
        before, after = old.get(label, []), new.get(label, [])
        matcher = SequenceMatcher(None, [_wording(c) for c in before], [_wording(c) for c in after], autojunk=False)
        for tag, a0, a1, b0, b1 in matcher.get_opcodes():
            if tag == 'equal':
                continue
            olds, news = list(before[a0:a1]), list(after[b0:b1])
            for clause in news:
                scored = [(SequenceMatcher(None, _wording(o), _wording(clause), autojunk=False).ratio(), index)
                          for index, o in enumerate(olds)]
                best = max(scored, default=(0.0, None))
                if best[0] >= MOVED_SIMILARITY:
                    changes.append(modified(olds.pop(best[1]), clause))
                else:
                    changes.append(added(clause))
            changes += [removed(clause) for clause in olds]
    return changes


def _limits(obligation: ExtractedObligation) -> dict:
    return {(q.attribute or q.unit, q.comparator): q.value for q in obligation.frame.quantities if q.role == 'LIMIT'}


def element_changes(old: ExtractedObligation, new: ExtractedObligation) -> tuple[list[dict], list[str]]:
    """(what changed between two readings of one duty, impact types)."""
    out, types = [], []
    if old.modality != new.modality:
        out.append({'element': 'modality', 'old': old.modality, 'new': new.modality})
        types.append('MODALITY_CHANGED')
    before, after = _limits(old), _limits(new)
    for key in sorted(set(before) | set(after), key=str):
        if before.get(key) == after.get(key):
            continue
        attribute, comparator = key
        entry = {'element': 'limit', 'attribute': attribute, 'comparator': comparator, 'old': before.get(key), 'new': after.get(key)}
        if key not in before:
            types.append('LIMIT_ADDED')
        elif key not in after:
            types.append('LIMIT_REMOVED')
        else:
            tighter = after[key] < before[key] if comparator in ('le', 'lt') else after[key] > before[key]
            entry['direction'] = 'TIGHTENED' if tighter else 'RELAXED'
            types.append('LIMIT_TIGHTENED' if tighter else 'LIMIT_RELAXED')
        out.append(entry)
    for name, label in (('product_classes', 'PRODUCT_SCOPE'), ('activity_classes', 'ADDRESSEE'), ('facility_classes', 'FACILITY_SCOPE')):
        a, b = set(getattr(old.scope, name)), set(getattr(new.scope, name))
        if a != b:
            out.append({'element': name, 'removed': sorted(a - b), 'added': sorted(b - a)})
            types.append(f'{label}_WIDENED' if b - a and not a - b else f'{label}_NARROWED' if a - b and not b - a else f'{label}_CHANGED')
    if old.scope.alcohol_scope != new.scope.alcohol_scope or old.scope.level != new.scope.level:
        out.append({'element': 'scope', 'old': [old.scope.level, old.scope.alcohol_scope], 'new': [new.scope.level, new.scope.alcohol_scope]})
        types.append('SCOPE_CHANGED')
    places = lambda o: sorted({m.id for m in o.frame.places})
    if places(old) != places(new):
        out.append({'element': 'places', 'removed': sorted(set(places(old)) - set(places(new))),
                    'added': sorted(set(places(new)) - set(places(old)))})
        types.append('PLACES_WIDENED' if set(places(new)) > set(places(old)) else 'PLACES_CHANGED')
    conditions = lambda o: sorted(f"{c['fact']} {c['op']} {c['value']}" for c in o.scope.conditions)
    if conditions(old) != conditions(new):
        out.append({'element': 'conditions', 'old': conditions(old), 'new': conditions(new)})
        types.append('CONDITION_CHANGED')
    exceptions = lambda o: sorted(e.get('quote', '') for e in o.scope.exceptions)
    if exceptions(old) != exceptions(new):
        added, removed = set(exceptions(new)) - set(exceptions(old)), set(exceptions(old)) - set(exceptions(new))
        out.append({'element': 'exceptions', 'removed': sorted(removed), 'added': sorted(added)})
        types.append('EXCEPTION_ADDED' if added and not removed else 'EXCEPTION_REMOVED' if removed and not added else 'EXCEPTION_CHANGED')
    return out, types


def obligation_changes(changes: list[ProvisionChange], old: list[ExtractedObligation], new: list[ExtractedObligation]) -> list[ObligationChange]:
    """The duties behind the changed clauses. A clause that was and is no duty (a definition, a delegation) yields none."""
    before = {(o.provision_ref, o.group): o for o in old}
    after = {(o.provision_ref, o.group): o for o in new}
    out = []
    for change in changes:
        olds = {g: o for (ref, g), o in before.items() if ref == change.old_ref} if change.old_ref else {}
        news = {g: o for (ref, g), o in after.items() if ref == change.provision_ref} if change.kind != 'REMOVED' else {}
        # One clause usually carries one duty; when its scope group is renamed by the change, the pair is still one duty.
        if len(olds) == 1 and len(news) == 1 and set(olds) != set(news):
            olds = {next(iter(news)): next(iter(olds.values()))}
        for group in sorted(set(olds) | set(news)):
            a, b = olds.get(group), news.get(group)
            if a is None:
                out.append(ObligationChange(provision_ref=change.provision_ref, kind='ADDED', impact_types=['NEW_DUTY'],
                                            new_obligation_id=b.obligation_id, topic=b.topic, group=group))
            elif b is None:
                out.append(ObligationChange(provision_ref=change.provision_ref, kind='REMOVED', impact_types=['DUTY_REMOVED'],
                                            old_obligation_id=a.obligation_id, topic=a.topic, group=group))
            else:
                details, types = element_changes(a, b)
                out.append(ObligationChange(provision_ref=change.provision_ref, kind='MODIFIED' if types else 'WORDING_ONLY',
                                            impact_types=types or ['WORDING_ONLY'], old_obligation_id=a.obligation_id,
                                            new_obligation_id=b.obligation_id, element_changes=details, topic=b.topic, group=group))
    return out


def version_changes(regulation_id: str, from_version: str, to_version: str, registry: Registry, store: CorpusStore):
    """(ImpactRecord without targets, old obligations, new obligations) for two stored versions of one regulation."""
    old, new = store.version(regulation_id, from_version), store.version(regulation_id, to_version)
    changes = clause_changes(store.sections(regulation_id, from_version), store.sections(regulation_id, to_version))
    before = extract_regulation(regulation_id, registry, store, from_version)[1]
    after = extract_regulation(regulation_id, registry, store, to_version)[1]
    duties = obligation_changes(changes, before, after)
    by_ref: dict[str, list[str]] = {}
    for duty in duties:
        by_ref.setdefault(duty.provision_ref, []).extend(duty.impact_types)
    changes = [c.model_copy(update={'impact_types': list(dict.fromkeys(by_ref.get(c.provision_ref, ['NO_DUTY_AFFECTED'])))}) for c in changes]
    record = ImpactRecord(change_id=f'{regulation_id}:{from_version}->{to_version}', regulation_id=regulation_id,
                          from_version=from_version, to_version=to_version, hash_changed=old.parsed_hash != new.parsed_hash,
                          provision_changes=changes, obligation_changes=duties, basis='VERSION_DIFF',
                          synthetic=old.synthetic or new.synthetic)
    return record, before, after


def note_changes(regulation_id: str, since: date, registry: Registry, store: CorpusStore, version_id: str | None = None):
    """(ImpactRecord without targets, [], obligations) from the amendment notes of one stored version: the clauses an
    instrument added or changed on or after `since`. The old wording is unknown, so a changed duty is NEW_OR_CHANGED."""
    version = store.version(regulation_id, version_id)
    changes = []
    for section in store.sections(regulation_id, version.version_id):
        if 'DELETED_PROVISION' in section.get('quality_flags', ()):
            continue
        clauses = split_clauses(section)
        for clause in clauses:
            notes = [n for n in clause_amendments(section, clauses)[clause.ref] if n.date >= since and n.action in ('ADDED', 'CHANGED', 'REENACTED')]
            if not notes:
                continue
            last = max(notes, key=lambda n: n.date)
            changes.append(ProvisionChange(provision_ref=clause.ref, kind='ADDED' if last.action == 'ADDED' else 'MODIFIED',
                                           new_text=clause.text, new_hash=digest(clause.text), basis='AMENDMENT_NOTE',
                                           instrument=last.instrument, amended_on=last.date))
    obligations = extract_regulation(regulation_id, registry, store, version.version_id)[1]
    refs = {c.provision_ref: c for c in changes}
    duties = [ObligationChange(provision_ref=o.provision_ref, kind='ADDED' if refs[o.provision_ref].kind == 'ADDED' else 'NEW_OR_CHANGED',
                               impact_types=['NEW_DUTY'] if refs[o.provision_ref].kind == 'ADDED' else ['DUTY_CHANGED_OLD_WORDING_UNKNOWN'],
                               new_obligation_id=o.obligation_id, topic=o.topic, group=o.group)
              for o in obligations if o.provision_ref in refs]
    with_duty = {d.provision_ref for d in duties}
    changes = [c.model_copy(update={'impact_types': ['NEW_DUTY' if c.kind == 'ADDED' else 'DUTY_CHANGED_OLD_WORDING_UNKNOWN']
                                    if c.provision_ref in with_duty else ['NO_DUTY_AFFECTED']}) for c in changes]
    record = ImpactRecord(change_id=f'{regulation_id}:notes-since-{since.isoformat()}@{version.version_id}', regulation_id=regulation_id,
                          from_version=f'before-{since.isoformat()}', to_version=version.version_id, hash_changed=bool(changes),
                          provision_changes=changes, obligation_changes=duties, basis='AMENDMENT_NOTE', synthetic=version.synthetic)
    return record, [], obligations


# -- impact on a profile -----------------------------------------------------------------------------
def _facilities(profile, entity_ids, product_ids, target) -> list[str]:
    if target.level == 'FACILITY':
        return [target.target_id]
    return sorted({f.facility_id for f in profile.facilities if f.entity_id in entity_ids
                   and (not product_ids or set(f.product_ids) & set(product_ids))})


def impact(record: ImpactRecord, old: list[ExtractedObligation], new: list[ExtractedObligation], profile, registry: Registry,
           store: CorpusStore, register=None) -> ImpactRecord:
    """The record with the targets of one enterprise profile each changed duty reaches, before and after.

    For every changed duty and every target: the routed status before and after, the delta, the entities, facilities and
    products behind the target, and (with a policy register) the documents and controls that speak about the duty as it
    now reads, the mapping status after the change and the action it calls for."""
    from .compare import compare_obligation, ownership
    by_id = {o.obligation_id: o for o in [*old, *new]}
    affected = []
    for duty in record.obligation_changes:
        before = by_id.get(duty.old_obligation_id) if duty.old_obligation_id else None
        after = by_id.get(duty.new_obligation_id) if duty.new_obligation_id else None
        routed_before = {d.target_id: d for d in route(before, profile, registry, store)[0]} if before else {}
        routed_after = {d.target_id: d for d in route(after, profile, registry, store)[0]} if after else {}
        reached = False
        unclear = [d for d in routed_after.values() if d.reason_codes == ['REGULATORY_SCOPE_UNCLEAR']]
        if after is not None and unclear and len(unclear) == len(routed_after):
            # The rules cannot say whom the changed clause binds: one row for the group, to be read by a person.
            affected.append(AffectedTarget(obligation_id=after.obligation_id, provision_ref=duty.provision_ref, level='GROUP',
                                           target_id=profile.group.group_id, decision_after='UNKNOWN', delta='UNKNOWN',
                                           departments=list(ownership()['topics'].get(duty.topic, ownership()['topics']['GENERAL'])),
                                           actions=[{'action': 'REVIEW', 'target': None, 'department': 'COMPLIANCE',
                                                     'why': ['REGULATORY_SCOPE_UNCLEAR']}]))
            continue
        for target_id in dict.fromkeys([*routed_after, *routed_before]):
            a, b = routed_before.get(target_id), routed_after.get(target_id)
            was = a is not None and a.status in ('APPLIES', 'PARTIAL')
            now = b is not None and b.status in ('APPLIES', 'PARTIAL')
            unknown = (b is not None and b.status == 'UNKNOWN') or (b is None and a is not None and a.status == 'UNKNOWN')
            if not was and not now and not unknown:
                continue
            decision = b or a
            if now and not was:
                delta = 'NEWLY_APPLIES' if before is not None or record.basis == 'VERSION_DIFF' else \
                    ('NEWLY_APPLIES' if duty.kind == 'ADDED' else 'CHANGED_REQUIREMENT')
            elif was and not now:
                delta = 'UNKNOWN' if unknown else 'NO_LONGER_APPLIES'
            elif was and now:
                delta = 'UNCHANGED' if duty.kind == 'WORDING_ONLY' else 'CHANGED_REQUIREMENT'
            else:
                delta = 'UNKNOWN'
            reached = True
            entity_ids = [decision.entity_id] if decision.entity_id else \
                sorted({e.entity_id for e in profile.legal_entities if decision.target_id in e.product_ids})
            product_ids = sorted(set(decision.applies_to_products) | ({decision.target_id} if decision.level == 'PRODUCT' else set()))
            settled = delta != 'UNKNOWN'                       # an undecided target names no facility or product as affected
            documents, controls, actions, coverage, status = [], [], [], None, None
            departments = list(ownership()['topics'].get(duty.topic, ownership()['topics']['GENERAL']))
            if register is not None and after is not None and now:
                row = compare_obligation(after, b, profile, register, registry)
                documents = sorted({r.document_id for r in row.readings})
                controls = list(row.mapping.control_ids)
                coverage, status = row.document_coverage, row.mapping.status
                actions = [dict(action) for action in row.actions]
                departments = list(dict.fromkeys([*departments, *(a_['department'] for a_ in actions)]))
            elif register is not None and before is not None and was and not now:
                row = compare_obligation(before, a, profile, register, registry)
                documents = sorted({r.document_id for r in row.readings})
                controls = list(row.mapping.control_ids)
                actions = [{'action': 'REVIEW_DOCUMENT', 'target': d, 'department': register.document(d).owner_department,
                            'why': ['DUTY_NO_LONGER_APPLIES']} for d in documents]
            affected.append(AffectedTarget(
                obligation_id=(after or before).obligation_id, provision_ref=duty.provision_ref, level=decision.level,
                target_id=target_id, decision_before=a.status if a else None, decision_after=b.status if b else None, delta=delta,
                entity_ids=entity_ids, facility_ids=_facilities(profile, entity_ids, product_ids, decision) if settled else [],
                product_ids=product_ids if settled else [],
                document_ids=documents, control_ids=controls, departments=departments, actions=actions, coverage_after=coverage,
                mapping_after=status))
        if not reached:
            affected.append(AffectedTarget(obligation_id=(after or before).obligation_id, provision_ref=duty.provision_ref,
                                           delta='NO_TARGET_AFFECTED'))
    if not record.obligation_changes:
        affected.append(AffectedTarget(obligation_id='NONE', delta='NO_TARGET_AFFECTED' if record.hash_changed else 'UNCHANGED'))
    counts: dict[str, int] = {}
    for target in affected:
        counts[target.delta] = counts.get(target.delta, 0) + 1
    collect = lambda name: sorted({value for t in affected for value in getattr(t, name)})
    summary = {'changed_clauses': len(record.provision_changes), 'changed_obligations': len(record.obligation_changes),
               'targets_by_delta': dict(sorted(counts.items())), 'entities': collect('entity_ids'), 'facilities': collect('facility_ids'),
               'products': collect('product_ids'), 'documents': collect('document_ids'), 'controls': collect('control_ids'),
               'departments': collect('departments'),
               'actions': sum(len(t.actions) for t in affected)}
    return record.model_copy(update={'affected': affected, 'summary': summary})
