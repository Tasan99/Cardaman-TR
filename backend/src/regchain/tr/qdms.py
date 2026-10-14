"""QDMS export: regulation -> Cardaman reasoning -> impacted policy / control -> human approval -> QDMS action.

The gap rows of an assessment (sector engines, and the board-decision layer) become one export row each, grouped the
way a quality and document management system (QDMS) takes a change request: the entity, the provision, the
applicability, the reason, the impacted process, the policy status, the required action, the evidence and the human
approval. Nothing is sent anywhere: the export is a pair of files (JSON, and CSV for a spreadsheet import), and no action
in it can be carried out before a person approves its row.

  export_rows       gap rows + assessments -> QdmsRow, every action DRAFT, approval PENDING
  apply_approvals   reviewers' approvals files -> rows APPROVED (actions READY_FOR_QDMS), REJECTED (CANCELLED) or
                    RETURNED (DRAFT, with the note); an approval names the row's fingerprint, and an approval of a row whose
                    reasoning changed since (another coverage, status, action, statement, entity that must act, impacted
                    process or evidence) is refused as STALE. No row, record or task of export.json is taken on trust: every
                    fingerprint is recomputed from the content beside it (an edited row is EXPORT_MODIFIED) and every
                    decision, and every earlier decision kept as its history, comes from the files given, the latest
                    holding; a file that decides a row two ways is refused whole. What no fingerprint covers is provenance
                    and decides nothing: a row's pack(s), its register period label, and the export's header and ledger
  ready_actions     only the actions of approved rows: what may be handed to the QDMS
  group_reviews     the UNKNOWN applicability records under parent review tasks: one task per (source question, cause,
                    article) or (company fact, target, missing gates); every record stays a child with its own decision
                    and evidence; the action and the approval belong to the task

The action vocabulary is generic (document change request, control definition, evidence request, review task,
nonconformity); mapping it onto a given QDMS product's import format is the integrator's step and is not done here.
"""
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, Field, PrivateAttr, ValidationError, field_validator

from ..pilot.schema import Strict
from .compare import GapReport, Register, ownership

EXPORT_FORMAT = 'cardaman-tr-qdms-export/1'
APPROVALS_FORMAT = 'cardaman-tr-qdms-approvals/1'
QDMS_RULES_VERSION = 'tr-qdms-v7'
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


class PastDecision(Strict):
    """An earlier decision on a row that a later one replaced: kept, never overwritten."""
    status: Literal['APPROVED', 'REJECTED', 'RETURNED']
    approver: str
    decided_at: datetime
    note: str = ''


class Approval(Strict):
    required: bool = True
    status: ApprovalStatus = 'PENDING'
    approver: str | None = None
    decided_at: datetime | None = None
    note: str = ''
    history: list[PastDecision] = []           # the earlier decisions of the approvals files, oldest first


class QdmsRow(Strict):
    row_id: str
    fingerprint: str                           # what an approval is bound to: who must act, the reasoning, the impacted process, the evidence, the actions
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
    required_actions: list[QdmsAction] = []    # empty once grouped: the action is the parent task's
    human_approval: Approval = Approval()
    task_id: str | None = None                 # the parent review task


CAUSE_TR = {'SOURCE_NO_ADDRESSEE_STATUTE': 'kanun hükmü kimi bağladığını söylemiyor',
            'SOURCE_NO_ADDRESSEE': 'hüküm kimi bağladığını söylemiyor',
            'SOURCE_ACTOR_WORD_WITHOUT_CLASS': 'hüküm bir taraf adlandırıyor, sözlükte sınıfı yok',
            'SOURCE_PRODUCT_SCOPE_MIXED': 'liste maddesi birden çok ürün sınıfını karıştırıyor',
            'SOURCE_NOT_GROUNDED': 'hükmün metni saklı sürümde birebir bulunamadı',
            'COMPANY_FACT_MISSING': 'şirket bilgisi eksik', 'COMPANY_FACT_CONFLICT': 'şirket bilgisi çelişkili'}


class QdmsReviewTask(Strict):
    """One question a person answers once for many applicability records: whom a clause of one article binds (Legal),
    or a missing company fact of one target (whoever keeps the profile). The children keep their own decision and
    evidence; approving the task decides them together, and only the task's action goes to the QDMS."""
    task_id: str
    fingerprint: str                           # the task's key and its children's fingerprints
    review_type: str
    cause: str
    department: str
    department_tr: str
    due_date: str | None = None                # none is stated today; tasks with different due dates are never merged
    question_tr: str
    regulation_ids: list[str]
    clauses: list[str]
    entities: list[str]
    targets: list[str]
    topics: int                                # distinct review topics (a clause, or a company fact) under the task
    child_ids: list[str]
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
    review_tasks: list[QdmsReviewTask] = []
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
    fingerprint = _row_fingerprint(entity, provision, applicability, reason, impacted, policy, evidence, actions, source_layer)
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
    rows = list(exported.values())
    items, tasks = group_reviews(list(reviews.values()))
    return QdmsExport(profile_id=profile.profile_id, register_synthetic=register.synthetic, register_period=register.period,
                      generated_at=generated_at or datetime.now().astimezone(), rows=rows, applicability_reviews=items,
                      review_tasks=tasks, ledger=ledger, summary=summarise(rows, items, ledger, tasks))


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
    action = _review_action(review.review_id, review.review_type, review.reason_codes, review.missing_facts, departments)
    fingerprint = _record_fingerprint(entity, source_layer, provision, applicability, uncertainty, action)
    return QdmsApplicabilityItem(row_id=f'APPLICABILITY:{review.review_id}', fingerprint=fingerprint, source_layer=source_layer,
                                 packs=[pack] if pack else [], entity=entity, provision=provision, applicability=applicability,
                                 uncertainty=uncertainty, required_actions=[action])


def _review_action(review_id: str, review_type: str, reason_codes: list, missing_facts: list, departments: dict) -> QdmsAction:
    qdms_type, department = REVIEW_ACTION.get(review_type, ('APPLICABILITY_REVIEW_TASK', 'COMPLIANCE'))
    return QdmsAction(action_id=f'{review_id}:applicability', qdms_type=qdms_type, change=review_type, department=department,
                      department_tr=departments.get(department, department), target=None,
                      why=[*reason_codes, *(g['gate'] for g in missing_facts)], priority='MEDIUM')


def _record_fingerprint(entity: dict, source_layer: str, provision: dict, applicability: dict, uncertainty: dict, action: QdmsAction) -> str:
    """What a review record's part of a task approval is bound to: who must act (and the target as written), the layer,
    the provision and its stored version, why the applicability is open (the facts missing and known, the source
    evidence) and the action it asks for."""
    return _digest({'entities': sorted(entity['entity_ids']), 'entity': {k: v for k, v in entity.items() if k != 'entity_ids'},
                    'layer': source_layer, 'provision': provision, 'applicability': applicability, 'uncertainty': uncertainty,
                    'action': action.model_dump(mode='json', exclude={'state'})})


def _refingerprint_record(item: QdmsApplicabilityItem, departments: dict) -> str:
    """The fingerprint of a grouped record from its own content (its action, given to the task, is rebuilt as
    applicability_item built it)."""
    action = _review_action(item.row_id.removeprefix('APPLICABILITY:'), item.uncertainty['review_type'], item.applicability['reason_codes'],
                            item.uncertainty['missing_facts'], departments)
    return _record_fingerprint(item.entity, item.source_layer, item.provision, item.applicability, item.uncertainty, action)


def review_cause(item: QdmsApplicabilityItem) -> str:
    """Why a record is undecided, from what it already carries: the review type, the clause's flags, the kind of text."""
    kind = item.uncertainty['review_type']
    if kind == 'COMPLETE_PROFILE_FACT':
        return 'COMPANY_FACT_MISSING'
    if kind == 'RESOLVE_PROFILE_CONFLICT':
        return 'COMPANY_FACT_CONFLICT'
    if kind == 'CHECK_SOURCE_GROUNDING':
        return 'SOURCE_NOT_GROUNDED'
    flags = item.uncertainty.get('source_evidence', {}).get('flags', [])
    if 'PRODUCT_SCOPE_MIXED' in flags:
        return 'SOURCE_PRODUCT_SCOPE_MIXED'
    if 'ACTOR_UNMAPPED' in flags:
        return 'SOURCE_ACTOR_WORD_WITHOUT_CLASS'
    return 'SOURCE_NO_ADDRESSEE_STATUTE' if item.provision['regulation_id'].startswith('TR:KANUN:') else 'SOURCE_NO_ADDRESSEE'


def _task_key(item: QdmsApplicabilityItem) -> tuple:
    kind, cause = item.uncertainty['review_type'], review_cause(item)
    department = REVIEW_ACTION.get(kind, ('', 'COMPLIANCE'))[1]
    if kind in ('CLARIFY_REGULATORY_SCOPE', 'CHECK_SOURCE_GROUNDING'):
        # a source question: the same for every entity; read in its article
        return (kind, cause, department, None, item.provision['provision_ref'].split('/')[0])
    gates = tuple(sorted({g['gate'] for g in item.uncertainty['missing_facts'] if g['gate'] != 'EXCEPTIONS'})) or ('EXCEPTIONS_ONLY',)
    return (kind, cause, department, None, f"{item.entity['target_level']}:{item.entity['target_id']}", gates)


def _topic_key(item: QdmsApplicabilityItem):
    if item.uncertainty['review_type'] in ('CLARIFY_REGULATORY_SCOPE', 'CHECK_SOURCE_GROUNDING'):
        return item.row_id.split(':')[1]                     # the clause (obligation id)
    return _task_key(item)


def group_reviews(items: list[QdmsApplicabilityItem], departments: dict | None = None) -> tuple[list[QdmsApplicabilityItem], list[QdmsReviewTask]]:
    """(the records, each pointing at its task and without an action of its own; the parent tasks). Records are grouped
    only when they share the review type, the cause, the department, the due date and the subject (an article for a
    source question, a target and its missing gates for a company fact). The department labels are the packaged ones
    unless given."""
    departments = ownership()['departments'] if departments is None else departments
    groups: dict[tuple, list[QdmsApplicabilityItem]] = {}
    for item in items:
        groups.setdefault(_task_key(item), []).append(item)
    tasks, task_of = [], {}
    for key, children in sorted(groups.items(), key=lambda kv: [str(k) for k in kv[0]]):
        kind, cause, department, due = key[0], key[1], key[2], key[3]
        task_id = 'TASK:' + _digest([str(k) for k in key])[:16]
        clauses = sorted({c.provision['provision_ref'] for c in children})
        entities = sorted({e for c in children for e in c.entity['entity_ids']})
        targets = sorted({c.entity['target_id'] for c in children})
        regulations = sorted({c.provision['regulation_id'] for c in children})
        if kind in ('CLARIFY_REGULATORY_SCOPE', 'CHECK_SOURCE_GROUNDING'):
            title = children[0].provision.get('title') or regulations[0]
            question = (f"{key[4]} ({title}): {len(clauses)} hüküm için {CAUSE_TR.get(cause, cause)}. Muhatabı yalnız metinden "
                        f"belirleyin; {len(targets)} hedef bu cevabı bekliyor.")
        else:
            gates = ', '.join(key[5])
            question = (f"{children[0].entity['target_name']} ({children[0].entity['target_id']}): {CAUSE_TR.get(cause, cause)} "
                        f"({gates}); {len(children)} değerlendirme bekliyor. Şirketten profil bilgisi istenir (profil anketi).")
        qdms_type = REVIEW_ACTION.get(kind, ('APPLICABILITY_REVIEW_TASK', department))[0]
        action = QdmsAction(action_id=f'{task_id}:1', qdms_type=qdms_type, change=kind, department=department,
                            department_tr=departments.get(department, department), why=[cause], priority='MEDIUM')
        fingerprint = _digest({'key': [str(k) for k in key], 'children': sorted(c.fingerprint for c in children)})
        tasks.append(QdmsReviewTask(task_id=task_id, fingerprint=fingerprint, review_type=kind, cause=cause, department=department,
                                    department_tr=departments.get(department, department), due_date=due, question_tr=question,
                                    regulation_ids=regulations, clauses=clauses, entities=entities, targets=targets,
                                    topics=len({_topic_key(c) for c in children}), child_ids=[c.row_id for c in children],
                                    required_actions=[action]))
        for child in children:
            task_of[child.row_id] = task_id
    grouped = [item.model_copy(update={'task_id': task_of[item.row_id], 'required_actions': []}) for item in items]
    return grouped, tasks


def _row_fingerprint(entity: dict, provision: dict, applicability: dict, reason: dict, impacted: dict, policy: dict, evidence: dict,
                     actions, source_layer: str) -> str:
    """What an approval of a row is bound to: who must act (the entities, and the target as written: its names, level and
    products), the layer, the provision and its stored version, the applicability, the reasoning, the impacted process
    (departments, documents, controls), the policy status, the evidence the decision rests on (not the register's period
    label) and the actions. A change of any of them makes an earlier approval stale."""
    return _digest({'entities': sorted(entity.get('entity_ids') or []), 'entity': {k: v for k, v in entity.items() if k != 'entity_ids'},
                    'layer': source_layer, 'provision': provision, 'applicability': applicability, 'reason': reason,
                    'impacted_process': impacted, 'policy': policy,
                    'evidence': {'evidence': evidence.get('evidence') or [], 'product_findings': evidence.get('product_findings') or []},
                    'actions': [a.model_dump(mode='json', exclude={'state'}) for a in actions]})


def _refingerprint(row: QdmsRow) -> str:
    return _row_fingerprint(row.entity, row.provision, row.applicability, row.reason, row.impacted_process, row.policy_status, row.evidence,
                            row.required_actions, row.source_layer)


def merge(first: QdmsExport, second: QdmsExport) -> QdmsExport:
    """Two exports of the same profile and register (the sector engines and the decision layer) as one."""
    if (first.profile_id, first.register_period) != (second.profile_id, second.register_period):
        raise ValueError('exports of different profiles or register periods are not merged')
    rows = [*first.rows, *[r for r in second.rows if r.row_id not in {x.row_id for x in first.rows}]]
    seen = {x.row_id for x in first.applicability_reviews}
    reviews, tasks = group_reviews([*first.applicability_reviews, *[r for r in second.applicability_reviews if r.row_id not in seen]])
    ledger = [*first.ledger, *second.ledger]
    return first.model_copy(update={'rows': rows, 'applicability_reviews': reviews, 'review_tasks': tasks, 'ledger': ledger,
                                    'summary': summarise(rows, reviews, ledger, tasks),
                                    'register_synthetic': first.register_synthetic or second.register_synthetic})


def summarise(rows: list[QdmsRow], reviews=(), ledger=(), tasks=()) -> dict:
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
            'review_topics': sum(t.topics for t in tasks), 'review_tasks': len(tasks),
            'review_tasks_by_cause': count(t.cause for t in tasks), 'review_tasks_by_department': count(t.department for t in tasks),
            'review_tasks_by_approval': count(t.human_approval.status for t in tasks),
            'applicability_actions_by_state': count(a.state for t in tasks for a in t.required_actions),
            'ready_for_qdms': sum(a.state == 'READY_FOR_QDMS' for a in actions)
            + sum(a.state == 'READY_FOR_QDMS' for t in tasks for a in t.required_actions),
            'ledger': count(entry['outcome'] for entry in ledger)}


DECIDED_AT_WRITTEN_OUT = re.compile(r'\d{4}-\d{2}-\d{2}[Tt _]\d{2}:\d{2}')
CLOCK_ALLOWANCE = timedelta(hours=1)           # a reviewer's clock a little ahead of the one approve runs on


class ApprovalEntry(Strict):
    row_id: str
    fingerprint: str
    decision: Literal['APPROVED', 'REJECTED', 'RETURNED']
    approver: str = Field(min_length=1)
    decided_at: AwareDatetime                  # when it was decided, with its UTC offset: decisions of several files are ordered by it
    note: str = ''

    @field_validator('decided_at', mode='before')
    @classmethod
    def _written_out(cls, value):
        """A number, or a string of digits, would be read as seconds since 1970: a spreadsheet's date serial (46307 for
        12 October 2026) would rank as 1970 and lose to every other decision. Only a date and time written out is read."""
        if isinstance(value, datetime) or (isinstance(value, str) and DECIDED_AT_WRITTEN_OUT.match(value)):
            return value
        raise ValueError(f'{value!r} is not a date and time written out; a number is not read as a date')

    @field_validator('decided_at')
    @classmethod
    def _not_after_this_run(cls, value: datetime) -> datetime:
        """A decision is dated when it was made: a time after this run (a mistyped year, a wrong UTC offset) would outrank
        every decision made later, and a later rejection would only be filed under its history."""
        now = datetime.now(timezone.utc)
        if value > now + CLOCK_ALLOWANCE:
            raise ValueError(f'{value.isoformat()} is later than this run ({now.isoformat(timespec="minutes")}); a decision is dated when it was made')
        return value


class Approvals(Strict):
    format: Literal['cardaman-tr-qdms-approvals/1']
    profile_id: str
    entries: list[ApprovalEntry]
    _source: str = PrivateAttr(default='the approvals file')   # the path it was read from, for messages


def approvals_template(export: QdmsExport) -> dict:
    """A file a reviewer fills in: one entry per row, with the fingerprint the decision will be bound to."""
    return {'format': APPROVALS_FORMAT, 'profile_id': export.profile_id,
            'entries': [{'row_id': r.row_id, 'fingerprint': r.fingerprint, 'decision': '', 'approver': '', 'decided_at': '', 'note': '',
                         'provision_ref': r.provision['provision_ref'], 'entity_ids': r.entity['entity_ids'],
                         'policy_status': r.policy_status['mapping_status'], 'decision_of_cardaman': r.reason['decision'],
                         'actions': [f'{a.qdms_type}/{a.change} -> {a.department}' for a in r.required_actions]}
                        for r in export.rows]
            + [{'row_id': t.task_id, 'fingerprint': t.fingerprint, 'decision': '', 'approver': '', 'decided_at': '', 'note': '',
                'kind': 'REVIEW_TASK', 'question': t.question_tr, 'cause': t.cause, 'clauses': len(t.clauses), 'children': len(t.child_ids),
                'actions': [f'{a.qdms_type}/{a.change} -> {a.department}' for a in t.required_actions]}
               for t in export.review_tasks]}


def _task_content(task: QdmsReviewTask) -> dict:
    """A task as grouping makes it: without its approval and the state of its action, which only approvals set."""
    content = task.model_dump(mode='json', exclude={'human_approval'})
    for action in content['required_actions']:
        action.pop('state', None)
    return content


def modified_ids(export: QdmsExport) -> set[str]:
    """The rows, records and tasks of an export whose content no longer gives what was exported. export.json is a plain
    file anyone can edit between export and approve, so no row, record or task in it is taken on trust: every row
    fingerprint is recomputed from the row, every record fingerprint from the record, and the tasks are grouped again
    from the records that hold; a task that differs from its regrouping (its question, entities, children or action), a
    record that points at another task, a row or record whose fingerprint cannot be computed from what it holds, and an
    id that appears twice are modified too. The check is the content against itself: department labels are the ones the
    export was written with (its tasks'), so a label changed in the packaged ownership data since is not an edit."""
    departments = ownership()['departments'] | {t.department: t.department_tr for t in export.review_tasks}
    modified: set[str] = set()
    for ids in ([r.row_id for r in export.rows], [t.task_id for t in export.review_tasks], [c.row_id for c in export.applicability_reviews]):
        modified |= {i for i, n in Counter(ids).items() if n > 1}
    for row in export.rows:
        try:
            fingerprint = _refingerprint(row)
        except (KeyError, TypeError, AttributeError, ValueError):
            fingerprint = None
        if fingerprint != row.fingerprint:
            modified.add(row.row_id)
    intact = []
    for record in export.applicability_reviews:
        try:
            fingerprint = _refingerprint_record(record, departments)
        except (KeyError, TypeError, AttributeError, ValueError):
            fingerprint = None
        if fingerprint != record.fingerprint or record.required_actions or record.row_id in modified:
            modified.add(record.row_id)
        else:
            intact.append(record)
    try:
        regrouped, tasks = group_reviews(intact, departments)
    except (KeyError, TypeError, AttributeError, ValueError, IndexError):
        regrouped, tasks = [], []
    expected = {t.task_id: t for t in tasks}
    task_of = {c.row_id: c.task_id for c in regrouped}
    for record in intact:
        if record.task_id != task_of.get(record.row_id):
            modified |= {record.row_id, task_of.get(record.row_id) or record.row_id}
    modified |= {t.task_id for t in export.review_tasks
                 if t.task_id not in expected or _task_content(t) != _task_content(expected[t.task_id])}
    return modified


def apply_approvals(export: QdmsExport, approvals: 'Approvals | list[Approvals]') -> tuple[QdmsExport, list[dict]]:
    """(the export with the approvals applied, problems). Decisions come only from the approvals files given (one, or
    several: every file of the row's history), never from an approval or action state already written in the export:
    every row and task starts PENDING / DRAFT again. An entry is applied only to its row, only while the row's content
    still gives its fingerprint (else EXPORT_MODIFIED) and only while that fingerprint is the one the reviewer saw (else
    STALE_APPROVAL). One file gives one decision per row: a file that decides a row two ways is refused whole. Across
    files the latest decision (decided_at) holds and the earlier ones are kept as its history; a decision written in the
    export, or an earlier one written in its history, that no file given makes is reported and leaves its row PENDING."""
    files = [approvals] if isinstance(approvals, Approvals) else list(approvals)
    for file in files:
        if file.profile_id != export.profile_id:
            raise ValueError(f'approvals for {file.profile_id}, export of {export.profile_id}')
    if export.rules_version != QDMS_RULES_VERSION:
        raise ValueError(f'the export was written under {export.rules_version} and this is {QDMS_RULES_VERSION}: its fingerprints cannot be '
                         f'checked; export again and review the new export')
    for file in files:
        decisions: dict[str, set[str]] = {}
        for entry in file.entries:
            decisions.setdefault(entry.row_id, set()).add(entry.decision)
        conflicting = sorted(i for i, d in decisions.items() if len(d) > 1)
        if conflicting:
            more = f' and {len(conflicting) - 50} more' if len(conflicting) > 50 else ''
            raise ValueError(f'{file._source} decides {len(conflicting)} row(s) or task(s) two ways (APPROVED/REJECTED/RETURNED); one file gives '
                             f'one decision per row, so nothing was applied: correct {", ".join(conflicting[:50])}{more}')
    by_id = {r.row_id: r for r in export.rows} | {t.task_id: t for t in export.review_tasks}
    children = {c.row_id: c.task_id for c in export.applicability_reviews}
    modified = modified_ids(export)
    problems = [{'row_id': i, 'problem': 'EXPORT_MODIFIED',
                 'detail': 'export.json was changed after export: this content does not give the fingerprint it carries; export again and review'}
                for i in dict.fromkeys([*(r.row_id for r in export.rows), *(c.row_id for c in export.applicability_reviews),
                                        *(t.task_id for t in export.review_tasks)]) if i in modified]
    given: dict[str, list[ApprovalEntry]] = {}
    twice: dict[str, str] = {}
    for file in files:
        seen: dict[str, ApprovalEntry] = {}
        for entry in file.entries:
            row = by_id.get(entry.row_id)
            if row is None and entry.row_id in children:
                problems.append({'row_id': entry.row_id, 'problem': 'APPROVE_THE_TASK', 'detail': f'the record belongs to {children[entry.row_id]}'})
            elif entry.row_id in modified:
                continue                                         # reported above
            elif row is None:
                problems.append({'row_id': entry.row_id, 'problem': 'UNKNOWN_ROW'})
            elif entry.fingerprint != row.fingerprint:
                problems.append({'row_id': entry.row_id, 'problem': 'STALE_APPROVAL',
                                 'detail': 'the row changed after it was reviewed; review it again'})
            elif entry.row_id in seen and seen[entry.row_id] != entry:
                twice.setdefault(entry.row_id, 'one approvals file decides the row more than once; nothing was decided')
            elif entry not in given.setdefault(entry.row_id, []):
                seen[entry.row_id] = entry
                given[entry.row_id].append(entry)
            else:
                seen[entry.row_id] = entry
    decided: dict[str, tuple[ApprovalEntry, list[ApprovalEntry]]] = {}
    for key, entries in given.items():
        entries = sorted(entries, key=lambda e: e.decided_at)
        if key not in twice and len(entries) > 1 and entries[-2].decided_at == entries[-1].decided_at:
            twice[key] = 'two decisions of the row carry the same time, so neither is the later one; nothing was decided'
        if key not in twice:
            decided[key] = (entries[-1], entries[:-1])
    problems += [{'row_id': key, 'problem': 'DECIDED_TWICE', 'detail': detail} for key, detail in twice.items()]

    def made_here(decision: 'Approval | PastDecision', key: str) -> bool:
        return any((e.decision, e.approver, e.decided_at, e.note) == (decision.status, decision.approver, decision.decided_at, decision.note)
                   for e in given.get(key, []))
    for key, item in [*((r.row_id, r) for r in export.rows), *((t.task_id, t) for t in export.review_tasks)]:
        written = item.human_approval
        if key in modified or (written.status == 'PENDING' and all(a.state == 'DRAFT' for a in item.required_actions)):
            continue
        if written.status == 'PENDING' or not made_here(written, key):
            detail = (f'export.json says {written.status} with actions {sorted({a.state for a in item.required_actions})}, and no approvals '
                      'file of this run makes that decision; give every file of the row again (or review it again). The row stays PENDING')
        else:
            lost = [h for h in written.history if not made_here(h, key)]
            if not lost:
                continue
            # the earlier decisions are the row's audit trail: a run given only the latest file must not drop them
            detail = (f"export.json keeps {len(lost)} earlier decision(s) of the row that no approvals file of this run makes "
                      f"({'; '.join(f'{h.status} by {h.approver} at {h.decided_at.isoformat()}' for h in lost)}); a decision's history is "
                      'never dropped: give every file of the row again. The row stays PENDING')
        problems.append({'row_id': key, 'problem': 'DECISION_NOT_IN_APPROVALS', 'detail': detail})
        decided.pop(key, None)
    state = {'APPROVED': 'READY_FOR_QDMS', 'REJECTED': 'CANCELLED', 'RETURNED': 'DRAFT'}

    def decide(item, key):
        if key not in decided:
            return item.model_copy(update={'human_approval': Approval(),
                                           'required_actions': [a.model_copy(update={'state': 'DRAFT'}) for a in item.required_actions]})
        latest, earlier = decided[key]
        return item.model_copy(update={
            'human_approval': Approval(status=latest.decision, approver=latest.approver, decided_at=latest.decided_at, note=latest.note,
                                       history=[PastDecision(status=e.decision, approver=e.approver, decided_at=e.decided_at, note=e.note)
                                                for e in earlier]),
            'required_actions': [a.model_copy(update={'state': state[latest.decision]}) for a in item.required_actions]})
    rows = [decide(r, r.row_id) for r in export.rows]
    tasks = [decide(t, t.task_id) for t in export.review_tasks]
    # a child takes its task's decision; its own decision and evidence are unchanged
    by_task = {t.task_id: t for t in tasks if t.task_id in decided}

    def via(child):
        task = by_task.get(child.task_id)
        if task is None or child.row_id in modified:
            return Approval()
        return task.human_approval.model_copy(update={'note': f'via {task.task_id}' + (f': {task.human_approval.note}' if task.human_approval.note else '')})
    reviews = [c.model_copy(update={'human_approval': via(c)}) for c in export.applicability_reviews]
    return export.model_copy(update={'rows': rows, 'applicability_reviews': reviews, 'review_tasks': tasks,
                                     'summary': summarise(rows, reviews, export.ledger, tasks)}), problems


def ready_actions(export: QdmsExport) -> list[dict]:
    """The actions a QDMS may open: those of approved rows only, each with the row it comes from."""
    out = []
    for row in export.rows:
        if row.human_approval.status != 'APPROVED':
            continue
        for action in row.required_actions:
            if action.state == 'READY_FOR_QDMS':
                out.append({**action.model_dump(mode='json'), 'row_id': row.row_id, 'entity_ids': row.entity['entity_ids'],
                            'provision_ref': row.provision['provision_ref'], 'regulation_id': row.provision['regulation_id'],
                            'approver': row.human_approval.approver, 'approved_at': row.human_approval.decided_at.isoformat()
                            if row.human_approval.decided_at else None})
    for task in export.review_tasks:
        if task.human_approval.status != 'APPROVED':
            continue
        for action in task.required_actions:
            if action.state == 'READY_FOR_QDMS':
                out.append({**action.model_dump(mode='json'), 'row_id': task.task_id, 'question': task.question_tr, 'entity_ids': task.entities,
                            'clauses': task.clauses, 'regulation_ids': task.regulation_ids, 'children': len(task.child_ids),
                            'approver': task.human_approval.approver, 'approved_at': task.human_approval.decided_at.isoformat()
                            if task.human_approval.decided_at else None})
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


REVIEW_CSV_COLUMNS = ['row_id', 'task_id', 'source_layer', 'packs', 'entity_id', 'entity_name', 'target_level', 'target_id', 'regulation_id',
                      'regulation_title', 'provision_ref', 'version_id', 'provision_quote', 'applicability', 'reason_codes', 'review_type',
                      'missing_facts', 'company_evidence', 'grounded', 'approval_status', 'approver', 'approved_at', 'fingerprint']
TASK_CSV_COLUMNS = ['task_id', 'review_type', 'cause', 'department', 'due_date', 'question', 'regulation_ids', 'clauses', 'entities', 'targets',
                    'topics', 'children', 'action_type', 'action_state', 'approval_status', 'approver', 'approved_at', 'fingerprint']


def task_csv_rows(export: QdmsExport) -> list[dict]:
    return [{'task_id': t.task_id, 'review_type': t.review_type, 'cause': t.cause, 'department': t.department_tr, 'due_date': t.due_date or '',
             'question': t.question_tr, 'regulation_ids': '; '.join(t.regulation_ids), 'clauses': '; '.join(t.clauses),
             'entities': '; '.join(t.entities), 'targets': '; '.join(t.targets), 'topics': t.topics, 'children': len(t.child_ids),
             'action_type': t.required_actions[0].qdms_type, 'action_state': t.required_actions[0].state,
             'approval_status': t.human_approval.status, 'approver': t.human_approval.approver or '',
             'approved_at': t.human_approval.decided_at.isoformat() if t.human_approval.decided_at else '', 'fingerprint': t.fingerprint}
            for t in export.review_tasks]


def review_csv_rows(export: QdmsExport) -> list[dict]:
    def gates(items):
        return ' | '.join(f"{g['gate']} {g.get('status', '')} required={g.get('required', '')} stated={g.get('stated', '')}"
                          f"{' complete' if g.get('complete') else ' open' if 'complete' in g else ''}"
                          f"{' conflicts=' + str(g['conflicts']) if g.get('conflicts') else ''}" for g in items)
    out = []
    for r in export.applicability_reviews:
        out.append({'row_id': r.row_id, 'task_id': r.task_id or '', 'source_layer': r.source_layer, 'packs': '; '.join(r.packs), 'entity_id': '; '.join(r.entity['entity_ids']),
                    'entity_name': r.entity['entity_name'], 'target_level': r.entity['target_level'], 'target_id': r.entity['target_id'],
                    'regulation_id': r.provision['regulation_id'], 'regulation_title': r.provision['title'], 'provision_ref': r.provision['provision_ref'],
                    'version_id': r.provision['version_id'], 'provision_quote': r.provision['quote'], 'applicability': 'UNKNOWN',
                    'reason_codes': '; '.join(r.applicability['reason_codes']), 'review_type': r.uncertainty['review_type'],
                    'missing_facts': gates(r.uncertainty['missing_facts']), 'company_evidence': gates(r.uncertainty['company_evidence']),
                    'grounded': r.uncertainty['source_evidence'].get('grounded'), 'approval_status': r.human_approval.status,
                    'approver': r.human_approval.approver or '',
                    'approved_at': r.human_approval.decided_at.isoformat() if r.human_approval.decided_at else '', 'fingerprint': r.fingerprint})
    return out


def write(export: QdmsExport, directory: Path) -> dict[str, Path]:
    """export.json; export.csv, review-tasks.csv and applicability-reviews.csv (UTF-8 with BOM, for a spreadsheet; one line
    per action, per review task, per child record); approvals-template.json; ready-actions.json."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = {'json': directory / 'export.json', 'csv': directory / 'export.csv', 'reviews_csv': directory / 'applicability-reviews.csv',
             'tasks_csv': directory / 'review-tasks.csv', 'template': directory / 'approvals-template.json', 'ready': directory / 'ready-actions.json'}
    with paths['tasks_csv'].open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=TASK_CSV_COLUMNS, delimiter=';')
        writer.writeheader()
        writer.writerows(task_csv_rows(export))
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


def _said(error: ValidationError) -> str:
    """A pydantic error as one line a reviewer can act on: field: what is wrong."""
    said = '; '.join(f"{'.'.join(str(p) for p in e['loc']) or 'file'}: {e['msg']}" for e in error.errors()[:5])
    if any('decided_at' in e['loc'] for e in error.errors()):
        said += ' (decided_at is the date and time the decision was made, with its UTC offset, e.g. 2026-10-12T10:00:00+03:00)'
    return said


def _read_json(path: Path):
    """A JSON file written by hand or re-saved by an editor: UTF-8 with or without a BOM; anything else is refused with
    a message that names the file, not a traceback."""
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except UnicodeDecodeError as error:
        raise ValueError(f'{path}: not UTF-8 text (byte {error.start}); save it as UTF-8') from None
    except json.JSONDecodeError as error:
        raise ValueError(f'{path}: not valid JSON ({error.msg}, line {error.lineno} column {error.colno})') from None


def _unreadable(export: QdmsExport) -> str | None:
    """The first row, record or task that the export's own writers cannot read (a hand edit that broke its shape: a key
    removed, a null among the ids), by its place and id; None when every one can be written."""
    for kind, items, read in (('rows', export.rows, csv_rows), ('applicability_reviews', export.applicability_reviews, review_csv_rows),
                              ('review_tasks', export.review_tasks, task_csv_rows)):
        for position, item in enumerate(items):
            try:
                read(export.model_copy(update={kind: [item]}))
            except (KeyError, TypeError, AttributeError, ValueError, IndexError) as error:
                return f"{kind}[{position}] ({getattr(item, 'row_id', None) or item.task_id}) cannot be read ({type(error).__name__}: {error})"
    return None


def load_export(path: Path) -> QdmsExport:
    """An export.json as written by `qdms export` (or by `qdms approve`). One whose rows, records or tasks its own writers
    cannot read is refused whole, as one of another shape is: export again."""
    path = Path(path)
    try:
        export = QdmsExport.model_validate(_read_json(path))
    except ValidationError as error:
        raise ValueError(f'{path}: not a QDMS export: {_said(error)}') from None
    unreadable = _unreadable(export)
    if unreadable:
        raise ValueError(f'{path}: not a QDMS export as written: {unreadable}; export again')
    return export


def load_approvals(path: Path) -> Approvals:
    """A filled approvals file (approvals-template.json with decisions). Entries still blank in the template are not
    decisions; a bad entry is named by its place in the file (entries[N], as an editor counts the template's entries)
    and its row."""
    path = Path(path)
    data = _read_json(path)
    if not isinstance(data, dict) or not isinstance(data.get('entries', []), list):
        raise ValueError(f'{path}: not an approvals file (an object with format, profile_id and a list of entries)')
    keep = set(ApprovalEntry.model_fields)
    entries = []
    for position, raw in enumerate(data.get('entries', [])):
        if not isinstance(raw, dict):
            raise ValueError(f'{path}: entries[{position}] is not an entry object')
        if not raw.get('decision'):
            continue
        try:
            entries.append(ApprovalEntry.model_validate({k: v for k, v in raw.items() if k in keep}))
        except ValidationError as error:
            raise ValueError(f"{path}: entries[{position}] ({raw.get('row_id', 'no row_id')}): {_said(error)}") from None
    try:
        approvals = Approvals.model_validate({**data, 'entries': []})
    except ValidationError as error:
        raise ValueError(f'{path}: not an approvals file: {_said(error)}') from None
    approvals = approvals.model_copy(update={'entries': entries})
    approvals._source = str(path)
    return approvals
