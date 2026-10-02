"""QDMS export: regulation -> Cardaman reasoning -> impacted policy / control -> human approval -> QDMS action.

The gap rows of an assessment (sector engines, and the board-decision layer) become one export row each, grouped the
way a quality and document management system (QDMS) takes a change request: the entity, the provision, the
applicability, the reason, the impacted process, the policy status, the required action, the evidence and the human
approval. Nothing is sent anywhere: the export is a pair of files (JSON, and CSV for a spreadsheet import), and no action
in it can be carried out before a person approves its row.

  export_rows       gap rows + assessments -> QdmsRow, every action DRAFT, approval PENDING
  apply_approvals   a reviewer's approvals file -> rows APPROVED (actions READY_FOR_QDMS), REJECTED (CANCELLED) or
                    RETURNED (DRAFT, with the note); an approval names the row's fingerprint, and an approval of a row whose
                    reasoning changed since (another coverage, status, action or statement) is refused as STALE
  ready_actions     only the actions of approved rows: what may be handed to the QDMS

The action vocabulary is generic (document change request, control definition, evidence request, review task,
nonconformity); mapping it onto a given QDMS product's import format is the integrator's step and is not done here.
"""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from ..pilot.schema import Strict
from .compare import GapReport, Register, ownership

EXPORT_FORMAT = 'cardaman-tr-qdms-export/1'
APPROVALS_FORMAT = 'cardaman-tr-qdms-approvals/1'
QDMS_RULES_VERSION = 'tr-qdms-v1'
# What each gap action asks the QDMS to open.
QDMS_ACTION = {
    'UPDATE_DOCUMENT': ('DOCUMENT_CHANGE_REQUEST', 'REVISE_CONFLICTING_STATEMENT'),
    'ADD_POLICY_STATEMENT': ('DOCUMENT_CHANGE_REQUEST', 'ADD_STATEMENT'),
    'COMPLETE_STATEMENT': ('DOCUMENT_CHANGE_REQUEST', 'COMPLETE_STATEMENT'),
    'ADD_CONTROL': ('CONTROL_DEFINITION_REQUEST', 'DEFINE_CONTROL'),
    'COLLECT_EVIDENCE': ('EVIDENCE_REQUEST', 'COLLECT_EVIDENCE'),
    'REVIEW': ('COMPLIANCE_REVIEW_TASK', 'DECIDE_COVERAGE'),
    'PRODUCT_NONCONFORMITY': ('NONCONFORMITY', 'OPEN_NONCONFORMITY'),
}
PRIORITY = {'CONTRADICTED': 'HIGH', 'NOT_COVERED': 'MEDIUM', 'UNKNOWN': 'MEDIUM', 'PARTIALLY_COVERED': 'LOW', 'COVERED': 'NONE'}
ActionState = Literal['DRAFT', 'READY_FOR_QDMS', 'CANCELLED']
ApprovalStatus = Literal['PENDING', 'APPROVED', 'REJECTED', 'RETURNED']


class QdmsAction(Strict):
    action_id: str
    qdms_type: str
    change: str
    department: str
    department_tr: str
    target: str | None = None                  # a statement (passage id), a control, a product, or nothing
    target_document: str | None = None
    why: list[str] = []
    priority: str
    state: ActionState = 'DRAFT'


class Approval(Strict):
    required: bool = True
    status: ApprovalStatus = 'PENDING'
    approver: str | None = None
    decided_at: datetime | None = None
    note: str = ''


class QdmsRow(Strict):
    row_id: str
    fingerprint: str                           # what an approval is bound to: the reasoning and the actions of this row
    source_layer: Literal['REGULATION', 'DECISION']
    pack: str | None = None
    packs: list[str] = []                      # every pack whose analysis produced this row (a shared module: several)
    entity: dict
    provision: dict
    applicability: dict
    reason: dict
    impacted_process: dict
    policy_status: dict
    required_actions: list[QdmsAction]
    evidence: dict
    human_approval: Approval = Approval()


# What an UNKNOWN applicability asks of a person: a fact for the profile, or a reading of the source. Never a policy action.
REVIEW_ACTION = {
    'COMPLETE_PROFILE_FACT': ('PROFILE_DATA_REQUEST', 'COMPLIANCE'),
    'RESOLVE_PROFILE_CONFLICT': ('PROFILE_DATA_REQUEST', 'COMPLIANCE'),
    'CLARIFY_REGULATORY_SCOPE': ('APPLICABILITY_REVIEW_TASK', 'LEGAL'),
    'CHECK_SOURCE_GROUNDING': ('APPLICABILITY_REVIEW_TASK', 'LEGAL'),
}


class QdmsApplicabilityItem(Strict):
    """A duty whose applicability on a target is UNKNOWN: a review or a missing-information task, kept apart from the
    policy rows. Its policy coverage is not assessed; it is never a nonconformity and never a document change."""
    row_id: str
    fingerprint: str
    kind: Literal['APPLICABILITY_REVIEW'] = 'APPLICABILITY_REVIEW'
    source_layer: Literal['REGULATION', 'DECISION']
    packs: list[str] = []
    entity: dict
    provision: dict
    applicability: dict
    uncertainty: dict
    coverage_assessed: bool = False
    required_actions: list[QdmsAction]
    human_approval: Approval = Approval()


class QdmsExport(Strict):
    format: Literal['cardaman-tr-qdms-export/1'] = EXPORT_FORMAT
    profile_id: str
    register_synthetic: bool
    register_period: str
    generated_at: datetime
    rules_version: str = QDMS_RULES_VERSION
    note: str = ('Taslak: hiçbir satır insan onayı olmadan QDMS eylemine dönüşmez. Model önerileri ve incelemeye giden '
                 'satırlar karar değildir.')
    rows: list[QdmsRow]
    applicability_reviews: list[QdmsApplicabilityItem] = []
    # Where every analysis row and every UNKNOWN decision went: {'pack', 'row_id', 'outcome', 'reason'}.
    ledger: list[dict] = []
    summary: dict = {}


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode('utf-8')).hexdigest()


def _target_name(profile, level: str, target_id: str) -> str:
    try:
        if level == 'LEGAL_ENTITY':
            return profile.entity(target_id).name
        if level == 'FACILITY':
            return profile.facility(target_id).name
        if level == 'PRODUCT':
            return profile.product(target_id).name
        if level == 'ACTIVITY':
            return profile.activity(target_id).activity_class
    except (KeyError, StopIteration, ValueError):
        pass
    return target_id


def _entities_of(profile, row) -> list[str]:
    """The legal entities that act on a row: its own, the owner of its facility or activity, or every entity that
    handles its product (a product duty is the business of each of them)."""
    if row.entity_id:
        return [row.entity_id]
    try:
        if row.level == 'LEGAL_ENTITY':
            return [row.target_id]
        if row.level == 'FACILITY':
            return [profile.facility(row.target_id).entity_id]
        if row.level == 'ACTIVITY':
            return [profile.activity(row.target_id).entity_id]
    except (KeyError, StopIteration, ValueError):
        return []
    return [e.entity_id for e in profile.legal_entities if row.target_id in e.product_ids]


def export_row(row, assessment, profile, register: Register, registry, pack: str | None = None,
               source_layer: str = 'REGULATION') -> QdmsRow:
    """One gap row and its assessment as a QDMS row: every action DRAFT, the approval PENDING."""
    departments = ownership()['departments']
    meta = registry.regulations.get(row.regulation_id)
    entity_ids = _entities_of(profile, row)
    entity = {'entity_id': entity_ids[0] if len(entity_ids) == 1 else None, 'entity_ids': entity_ids,
              'entity_name': '; '.join(_target_name(profile, 'LEGAL_ENTITY', e) for e in entity_ids),
              'target_level': row.level, 'target_id': row.target_id, 'target_name': _target_name(profile, row.level, row.target_id),
              'products': list(row.applies_to_products)}
    provision = {'regulation_id': row.regulation_id, 'title': meta.title if meta else '', 'binding_status': meta.binding_status if meta else '',
                 'regulator': meta.regulator if meta else '', 'provision_ref': row.provision_ref, 'modality': row.modality,
                 'topic': row.topic, 'quote': row.quote, 'version_id': None,
                 'source_url': meta.source_url if meta else None}
    applicability = {'status': row.applicability, 'basis': assessment.applicability_basis}
    statements = [{'passage_id': r.passage_id, 'document_id': r.document_id, 'relation': r.relation, 'reasons': r.reasons, 'quote': r.quote}
                  for r in row.readings]
    reason = {'decision': assessment.decision, 'document_coverage': row.document_coverage, 'coverage_basis': row.coverage_basis,
              'coverage_reasons': row.coverage_reasons, 'rule_coverage': row.rule_coverage, 'model_proposal': assessment.proposal,
              'review_reasons': assessment.review_reasons, 'statements': statements, 'gap': row.gap}
    controls = {c.control_id: c for c in register.controls}
    impacted = {'topic': row.topic, 'departments': row.mapping.departments,
                'departments_tr': [departments.get(d, d) for d in row.mapping.departments],
                'documents_in_force': row.documents_in_force, 'documents_relied_on': row.mapping.document_ids,
                'controls': [{'control_id': c, 'description': controls[c].description if c in controls else '',
                              'owner_department': controls[c].owner_department if c in controls else ''} for c in row.mapping.control_ids]}
    policy = {'mapping_status': row.mapping.status, 'mapping_reasons': row.mapping.reasons,
              'documents': [{'document_id': d.document_id, 'title': d.title, 'version': d.version, 'owner_department': d.owner_department}
                            for d in register.documents if d.document_id in set(row.mapping.document_ids)]}
    by_id = {e.evidence_id: e for e in register.evidence}
    evidence = {'evidence': [{'evidence_id': e, 'control_id': by_id[e].control_id, 'evidence_type': by_id[e].evidence_type,
                              'period': by_id[e].period, 'in_period': by_id[e].in_period} for e in row.mapping.evidence_ids if e in by_id],
                'register_period': register.period, 'product_findings': [f.model_dump(mode='json') for f in row.product_findings]}
    passages = {p.passage_id: p for p in register.passages}
    actions = []
    review = assessment.decision != 'AUTO'
    plan = list(row.actions)
    if review and not any(a['action'] == 'REVIEW' for a in plan):
        # an open decision is a person's to take first: the review task leads, the other actions wait for its outcome
        owner = row.mapping.departments[0] if row.mapping.departments else 'COMPLIANCE'
        plan.insert(0, {'action': 'REVIEW', 'target': None, 'department': owner, 'why': assessment.review_reasons})
    for index, action in enumerate(plan, 1):
        qdms_type, change = QDMS_ACTION.get(action['action'], ('COMPLIANCE_REVIEW_TASK', action['action']))
        target = action.get('target')
        actions.append(QdmsAction(action_id=f'{row.obligation_id}:{row.target_id}:{index}', qdms_type=qdms_type, change=change,
                                  department=action['department'], department_tr=departments.get(action['department'], action['department']),
                                  target=target, target_document=passages[target].document_id if target in passages else None,
                                  why=list(action.get('why') or []), priority='HIGH' if review and row.mapping.status == 'CONTRADICTED'
                                  else PRIORITY.get(row.mapping.status, 'MEDIUM')))
    row_id = f'{row.obligation_id}:{row.target_id}'
    fingerprint = _digest({'provision': provision, 'applicability': applicability, 'reason': reason, 'policy': policy,
                           'actions': [a.model_dump(mode='json', exclude={'state'}) for a in actions]})
    return QdmsRow(row_id=row_id, fingerprint=fingerprint, source_layer=source_layer, pack=pack, entity=entity, provision=provision,
                   applicability=applicability, reason=reason, impacted_process=impacted, policy_status=policy,
                   required_actions=actions, evidence=evidence)


def export_rows(profile, register: Register, registry, reports: dict[str, GapReport], assessments: dict[str, list],
                include_covered: bool = False, source_layer: str = 'REGULATION', generated_at: datetime | None = None,
                store=None) -> QdmsExport:
    """The export of one profile: one row per gap row of every report (pack id -> report), skipping the covered automatic
    rows unless asked. With `store` (the corpus, or decisions.LayeredStore) each provision names the stored version it was
    read from (the engines read the head version), and the fingerprint covers it: a new version of the text makes an
    earlier approval stale."""
    heads: dict[str, str | None] = {}
    reviews: dict[str, QdmsApplicabilityItem] = {}
    exported: dict[str, QdmsRow] = {}
    ledger: list[dict] = []
    for pack, report in reports.items():
        for review in report.applicability_reviews:
            item = reviews.get(review.review_id)
            if item is None:
                reviews[review.review_id] = applicability_item(review, profile, registry, pack, source_layer)
                ledger.append({'pack': pack, 'row_id': f'APPLICABILITY:{review.review_id}', 'outcome': 'APPLICABILITY_REVIEW',
                               'reason': review.review_type})
            else:
                if pack not in item.packs:
                    item.packs.append(pack)
                ledger.append({'pack': pack, 'row_id': f'APPLICABILITY:{review.review_id}', 'outcome': 'MERGED_SAME_DUTY_AND_TARGET',
                               'reason': 'the same duty on the same target was read by another pack'})
        for row, assessment in zip(report.rows, assessments[pack]):
            row_id = f'{row.obligation_id}:{row.target_id}'
            if not include_covered and row.mapping.status == 'COVERED' and assessment.decision == 'AUTO' and not row.actions:
                ledger.append({'pack': pack, 'row_id': row_id, 'outcome': 'EXCLUDED_COVERED_AUTO',
                               'reason': 'covered by policy, control and evidence; automatic decision; no action'})
                continue
            qrow = export_row(row, assessment, profile, register, registry, pack, source_layer)
            if store is not None:
                if row.regulation_id not in heads:
                    head = store.head(row.regulation_id)
                    heads[row.regulation_id] = head.version_id if head is not None else None
                qrow.provision['version_id'] = heads[row.regulation_id]
                qrow = qrow.model_copy(update={'fingerprint': _refingerprint(qrow)})
            first = exported.get(qrow.row_id)
            if first is not None and first.fingerprint == qrow.fingerprint:
                first.packs.append(pack)
                ledger.append({'pack': pack, 'row_id': qrow.row_id, 'outcome': 'MERGED_SAME_DUTY_AND_TARGET',
                               'reason': 'the same duty on the same target was read by another pack with the same result'})
                continue
            if first is not None:
                # two packs read the same duty on the same target differently: both are kept, each under its own id
                qrow = qrow.model_copy(update={'row_id': f'{qrow.row_id}:{pack}'})
            qrow.packs.append(pack)
            exported[qrow.row_id] = qrow
            ledger.append({'pack': pack, 'row_id': qrow.row_id, 'outcome': 'EXPORTED' if first is None else 'EXPORTED_PACKS_DISAGREE',
                           'reason': qrow.policy_status['mapping_status']})
    rows, items = list(exported.values()), list(reviews.values())
    return QdmsExport(profile_id=profile.profile_id, register_synthetic=register.synthetic, register_period=register.period,
                      generated_at=generated_at or datetime.now().astimezone(), rows=rows, applicability_reviews=items, ledger=ledger,
                      summary=summarise(rows, items, ledger))


def applicability_item(review, profile, registry, pack: str | None, source_layer: str = 'REGULATION') -> QdmsApplicabilityItem:
    """One UNKNOWN applicability as a QDMS review task: the entity, the provision and its stored version, why it is
    undecided, the facts missing and the facts known, the source evidence, and the task, DRAFT until a person approves."""
    from .compare import REVIEW_TYPES
    departments = ownership()['departments']
    meta = registry.regulations.get(review.regulation_id)
    entity_ids = [review.entity_id] if review.entity_id else (
        [review.target_id] if review.level == 'LEGAL_ENTITY' else
        [e.entity_id for e in profile.legal_entities if review.target_id in e.product_ids] if review.level == 'PRODUCT' else [])
    entity = {'entity_id': entity_ids[0] if len(entity_ids) == 1 else None, 'entity_ids': entity_ids,
              'entity_name': '; '.join(_target_name(profile, 'LEGAL_ENTITY', e) for e in entity_ids),
              'target_level': review.level, 'target_id': review.target_id, 'target_name': _target_name(profile, review.level, review.target_id)}
    provision = {'regulation_id': review.regulation_id, 'title': meta.title if meta else '', 'binding_status': meta.binding_status if meta else '',
                 'provision_ref': review.provision_ref, 'modality': review.modality, 'topic': review.topic, 'quote': review.quote,
                 'version_id': review.version_id, 'source_url': meta.source_url if meta else None}
    applicability = {'status': 'UNKNOWN', 'reason_codes': review.reason_codes}
    uncertainty = {'review_type': review.review_type, 'why': REVIEW_TYPES.get(review.review_type, ''), 'missing_facts': review.missing_facts,
                   'company_evidence': review.company_evidence, 'source_evidence': review.source_evidence}
    qdms_type, department = REVIEW_ACTION.get(review.review_type, ('APPLICABILITY_REVIEW_TASK', 'COMPLIANCE'))
    action = QdmsAction(action_id=f'{review.review_id}:applicability', qdms_type=qdms_type, change=review.review_type, department=department,
                        department_tr=departments.get(department, department), target=None,
                        why=[*review.reason_codes, *(g['gate'] for g in review.missing_facts)], priority='MEDIUM')
    fingerprint = _digest({'provision': provision, 'applicability': applicability, 'uncertainty': uncertainty,
                           'action': action.model_dump(mode='json', exclude={'state'})})
    return QdmsApplicabilityItem(row_id=f'APPLICABILITY:{review.review_id}', fingerprint=fingerprint, source_layer=source_layer,
                                 packs=[pack] if pack else [], entity=entity, provision=provision, applicability=applicability,
                                 uncertainty=uncertainty, required_actions=[action])


def _refingerprint(row: QdmsRow) -> str:
    return _digest({'provision': row.provision, 'applicability': row.applicability, 'reason': row.reason, 'policy': row.policy_status,
                    'actions': [a.model_dump(mode='json', exclude={'state'}) for a in row.required_actions]})


def merge(first: QdmsExport, second: QdmsExport) -> QdmsExport:
    """Two exports of the same profile and register (the sector engines and the decision layer) as one."""
    if (first.profile_id, first.register_period) != (second.profile_id, second.register_period):
        raise ValueError('exports of different profiles or register periods are not merged')
    rows = [*first.rows, *[r for r in second.rows if r.row_id not in {x.row_id for x in first.rows}]]
    seen = {x.row_id for x in first.applicability_reviews}
    reviews = [*first.applicability_reviews, *[r for r in second.applicability_reviews if r.row_id not in seen]]
    ledger = [*first.ledger, *second.ledger]
    return first.model_copy(update={'rows': rows, 'applicability_reviews': reviews, 'ledger': ledger, 'summary': summarise(rows, reviews, ledger),
                                    'register_synthetic': first.register_synthetic or second.register_synthetic})


def summarise(rows: list[QdmsRow], reviews=(), ledger=()) -> dict:
    def count(values):
        out: dict[str, int] = {}
        for value in values:
            out[value] = out.get(value, 0) + 1
        return dict(sorted(out.items()))
    actions = [a for r in rows for a in r.required_actions]
    return {'rows': len(rows), 'by_layer': count(r.source_layer for r in rows), 'by_policy_status': count(r.policy_status['mapping_status'] for r in rows),
            'by_decision': count(r.reason['decision'] for r in rows), 'by_approval': count(r.human_approval.status for r in rows),
            'actions': len(actions), 'actions_by_type': count(a.qdms_type for a in actions), 'actions_by_state': count(a.state for a in actions),
            'actions_by_department': count(a.department for a in actions),
            'applicability_reviews': len(reviews), 'applicability_reviews_by_type': count(r.uncertainty['review_type'] for r in reviews),
            'applicability_reviews_by_approval': count(r.human_approval.status for r in reviews),
            'applicability_actions_by_state': count(a.state for r in reviews for a in r.required_actions),
            'ready_for_qdms': sum(a.state == 'READY_FOR_QDMS' for a in actions)
            + sum(a.state == 'READY_FOR_QDMS' for r in reviews for a in r.required_actions),
            'ledger': count(entry['outcome'] for entry in ledger)}


class ApprovalEntry(Strict):
    row_id: str
    fingerprint: str
    decision: Literal['APPROVED', 'REJECTED', 'RETURNED']
    approver: str = Field(min_length=1)
    decided_at: datetime
    note: str = ''


class Approvals(Strict):
    format: Literal['cardaman-tr-qdms-approvals/1']
    profile_id: str
    entries: list[ApprovalEntry]


def approvals_template(export: QdmsExport) -> dict:
    """A file a reviewer fills in: one entry per row, with the fingerprint the decision will be bound to."""
    return {'format': APPROVALS_FORMAT, 'profile_id': export.profile_id,
            'entries': [{'row_id': r.row_id, 'fingerprint': r.fingerprint, 'decision': '', 'approver': '', 'decided_at': '', 'note': '',
                         'provision_ref': r.provision['provision_ref'], 'entity_ids': r.entity['entity_ids'],
                         'policy_status': r.policy_status['mapping_status'], 'decision_of_cardaman': r.reason['decision'],
                         'actions': [f'{a.qdms_type}/{a.change} -> {a.department}' for a in r.required_actions]}
                        for r in export.rows]
            + [{'row_id': r.row_id, 'fingerprint': r.fingerprint, 'decision': '', 'approver': '', 'decided_at': '', 'note': '',
                'kind': 'APPLICABILITY_REVIEW', 'provision_ref': r.provision['provision_ref'], 'entity_ids': r.entity['entity_ids'],
                'review_type': r.uncertainty['review_type'], 'missing': [g['gate'] for g in r.uncertainty['missing_facts']],
                'actions': [f'{a.qdms_type}/{a.change} -> {a.department}' for a in r.required_actions]}
               for r in export.applicability_reviews]}


def apply_approvals(export: QdmsExport, approvals: Approvals) -> tuple[QdmsExport, list[dict]]:
    """(the export with the approvals applied, problems). An entry is applied only to its row and only while the row's
    fingerprint is the one the reviewer saw; anything else is reported and leaves the row PENDING."""
    if approvals.profile_id != export.profile_id:
        raise ValueError(f'approvals for {approvals.profile_id}, export of {export.profile_id}')
    by_id = {r.row_id: r for r in [*export.rows, *export.applicability_reviews]}
    problems, decided = [], {}
    for entry in approvals.entries:
        row = by_id.get(entry.row_id)
        if row is None:
            problems.append({'row_id': entry.row_id, 'problem': 'UNKNOWN_ROW'})
        elif entry.fingerprint != row.fingerprint:
            problems.append({'row_id': entry.row_id, 'problem': 'STALE_APPROVAL',
                             'detail': 'the row changed after it was reviewed; review it again'})
        elif entry.row_id in decided:
            problems.append({'row_id': entry.row_id, 'problem': 'DECIDED_TWICE'})
        else:
            decided[entry.row_id] = entry
    def decide(row):
        entry = decided.get(row.row_id)
        if entry is None:
            return row
        state = {'APPROVED': 'READY_FOR_QDMS', 'REJECTED': 'CANCELLED', 'RETURNED': 'DRAFT'}[entry.decision]
        return row.model_copy(update={
            'human_approval': Approval(status=entry.decision, approver=entry.approver, decided_at=entry.decided_at, note=entry.note),
            'required_actions': [a.model_copy(update={'state': state}) for a in row.required_actions]})
    rows = [decide(r) for r in export.rows]
    reviews = [decide(r) for r in export.applicability_reviews]
    return export.model_copy(update={'rows': rows, 'applicability_reviews': reviews,
                                     'summary': summarise(rows, reviews, export.ledger)}), problems


def ready_actions(export: QdmsExport) -> list[dict]:
    """The actions a QDMS may open: those of approved rows only, each with the row it comes from."""
    out = []
    for row in [*export.rows, *export.applicability_reviews]:
        if row.human_approval.status != 'APPROVED':
            continue
        for action in row.required_actions:
            if action.state == 'READY_FOR_QDMS':
                out.append({**action.model_dump(mode='json'), 'row_id': row.row_id, 'entity_ids': row.entity['entity_ids'],
                            'provision_ref': row.provision['provision_ref'], 'regulation_id': row.provision['regulation_id'],
                            'approver': row.human_approval.approver, 'approved_at': row.human_approval.decided_at.isoformat()
                            if row.human_approval.decided_at else None})
    return out


CSV_COLUMNS = ['row_id', 'source_layer', 'pack', 'entity_id', 'entity_name', 'target_level', 'target_id', 'regulation_id', 'regulation_title',
               'binding_status', 'provision_ref', 'provision_quote', 'applicability', 'applicability_basis', 'cardaman_decision',
               'document_coverage', 'coverage_basis', 'review_reasons', 'model_proposal', 'statements', 'departments', 'documents_relied_on',
               'controls', 'policy_status', 'action_type', 'action_change', 'action_department', 'action_target', 'priority', 'action_state',
               'evidence', 'approval_status', 'approver', 'approved_at', 'fingerprint']


def csv_rows(export: QdmsExport) -> list[dict]:
    """One CSV line per action (a row without an action is one line with the action columns empty)."""
    out = []
    for row in export.rows:
        base = {'row_id': row.row_id, 'source_layer': row.source_layer, 'pack': row.pack or '', 'entity_id': '; '.join(row.entity['entity_ids']),
                'entity_name': row.entity['entity_name'], 'target_level': row.entity['target_level'], 'target_id': row.entity['target_id'],
                'regulation_id': row.provision['regulation_id'], 'regulation_title': row.provision['title'],
                'binding_status': row.provision['binding_status'], 'provision_ref': row.provision['provision_ref'],
                'provision_quote': row.provision['quote'], 'applicability': row.applicability['status'],
                'applicability_basis': row.applicability['basis'], 'cardaman_decision': row.reason['decision'],
                'document_coverage': row.reason['document_coverage'], 'coverage_basis': row.reason['coverage_basis'],
                'review_reasons': '; '.join(row.reason['review_reasons']), 'model_proposal': row.reason['model_proposal'] or '',
                'statements': ' | '.join(f"{s['passage_id']} {s['relation']}: {s['quote']}" for s in row.reason['statements']),
                'departments': '; '.join(row.impacted_process['departments_tr']),
                'documents_relied_on': '; '.join(row.impacted_process['documents_relied_on']),
                'controls': '; '.join(c['control_id'] for c in row.impacted_process['controls']),
                'policy_status': row.policy_status['mapping_status'],
                'evidence': '; '.join(f"{e['evidence_id']} ({e['period']}{'' if e['in_period'] else ', dönem dışı'})" for e in row.evidence['evidence']),
                'approval_status': row.human_approval.status, 'approver': row.human_approval.approver or '',
                'approved_at': row.human_approval.decided_at.isoformat() if row.human_approval.decided_at else '', 'fingerprint': row.fingerprint}
        if not row.required_actions:
            out.append({**base, **{c: '' for c in CSV_COLUMNS if c not in base}})
        for action in row.required_actions:
            out.append({**base, 'action_type': action.qdms_type, 'action_change': action.change, 'action_department': action.department_tr,
                        'action_target': action.target or '', 'priority': action.priority, 'action_state': action.state})
    return out


REVIEW_CSV_COLUMNS = ['row_id', 'source_layer', 'packs', 'entity_id', 'entity_name', 'target_level', 'target_id', 'regulation_id',
                      'regulation_title', 'provision_ref', 'version_id', 'provision_quote', 'applicability', 'reason_codes', 'review_type',
                      'missing_facts', 'company_evidence', 'grounded', 'action_type', 'action_department', 'action_state', 'approval_status',
                      'approver', 'approved_at', 'fingerprint']


def review_csv_rows(export: QdmsExport) -> list[dict]:
    def gates(items):
        return ' | '.join(f"{g['gate']} {g.get('status', '')} required={g.get('required', '')} stated={g.get('stated', '')}"
                          f"{' complete' if g.get('complete') else ' open' if 'complete' in g else ''}"
                          f"{' conflicts=' + str(g['conflicts']) if g.get('conflicts') else ''}" for g in items)
    out = []
    for r in export.applicability_reviews:
        action = r.required_actions[0]
        out.append({'row_id': r.row_id, 'source_layer': r.source_layer, 'packs': '; '.join(r.packs), 'entity_id': '; '.join(r.entity['entity_ids']),
                    'entity_name': r.entity['entity_name'], 'target_level': r.entity['target_level'], 'target_id': r.entity['target_id'],
                    'regulation_id': r.provision['regulation_id'], 'regulation_title': r.provision['title'], 'provision_ref': r.provision['provision_ref'],
                    'version_id': r.provision['version_id'], 'provision_quote': r.provision['quote'], 'applicability': 'UNKNOWN',
                    'reason_codes': '; '.join(r.applicability['reason_codes']), 'review_type': r.uncertainty['review_type'],
                    'missing_facts': gates(r.uncertainty['missing_facts']), 'company_evidence': gates(r.uncertainty['company_evidence']),
                    'grounded': r.uncertainty['source_evidence'].get('grounded'), 'action_type': action.qdms_type,
                    'action_department': action.department_tr, 'action_state': action.state, 'approval_status': r.human_approval.status,
                    'approver': r.human_approval.approver or '',
                    'approved_at': r.human_approval.decided_at.isoformat() if r.human_approval.decided_at else '', 'fingerprint': r.fingerprint})
    return out


def write(export: QdmsExport, directory: Path) -> dict[str, Path]:
    """export.json, export.csv and applicability-reviews.csv (UTF-8 with BOM, for a spreadsheet), approvals-template.json,
    ready-actions.json."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = {'json': directory / 'export.json', 'csv': directory / 'export.csv', 'reviews_csv': directory / 'applicability-reviews.csv',
             'template': directory / 'approvals-template.json', 'ready': directory / 'ready-actions.json'}
    with paths['reviews_csv'].open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_CSV_COLUMNS, delimiter=';')
        writer.writeheader()
        writer.writerows(review_csv_rows(export))
    paths['json'].write_text(json.dumps(export.model_dump(mode='json'), ensure_ascii=False, indent=1), encoding='utf-8')
    with paths['csv'].open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, delimiter=';')
        writer.writeheader()
        writer.writerows(csv_rows(export))
    paths['template'].write_text(json.dumps(approvals_template(export), ensure_ascii=False, indent=1), encoding='utf-8')
    paths['ready'].write_text(json.dumps(ready_actions(export), ensure_ascii=False, indent=1), encoding='utf-8')
    return paths


def load_export(path: Path) -> QdmsExport:
    return QdmsExport.model_validate(json.loads(Path(path).read_text(encoding='utf-8')))


def load_approvals(path: Path) -> Approvals:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    keep = set(ApprovalEntry.model_fields)
    # entries still blank in the template are not decisions
    data['entries'] = [{k: v for k, v in e.items() if k in keep} for e in data.get('entries', []) if e.get('decision')]
    return Approvals.model_validate(data)
