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
    entity: dict
    provision: dict
    applicability: dict
    reason: dict
    impacted_process: dict
    policy_status: dict
    required_actions: list[QdmsAction]
    evidence: dict
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
    rows = []
    heads: dict[str, str | None] = {}
    for pack, report in reports.items():
        for row, assessment in zip(report.rows, assessments[pack]):
            if not include_covered and row.mapping.status == 'COVERED' and assessment.decision == 'AUTO' and not row.actions:
                continue
            qrow = export_row(row, assessment, profile, register, registry, pack, source_layer)
            if store is not None:
                if row.regulation_id not in heads:
                    head = store.head(row.regulation_id)
                    heads[row.regulation_id] = head.version_id if head is not None else None
                qrow.provision['version_id'] = heads[row.regulation_id]
                qrow = qrow.model_copy(update={'fingerprint': _refingerprint(qrow)})
            rows.append(qrow)
    return QdmsExport(profile_id=profile.profile_id, register_synthetic=register.synthetic, register_period=register.period,
                      generated_at=generated_at or datetime.now().astimezone(), rows=rows, summary=summarise(rows))


def _refingerprint(row: QdmsRow) -> str:
    return _digest({'provision': row.provision, 'applicability': row.applicability, 'reason': row.reason, 'policy': row.policy_status,
                    'actions': [a.model_dump(mode='json', exclude={'state'}) for a in row.required_actions]})


def merge(first: QdmsExport, second: QdmsExport) -> QdmsExport:
    """Two exports of the same profile and register (the sector engines and the decision layer) as one."""
    if (first.profile_id, first.register_period) != (second.profile_id, second.register_period):
        raise ValueError('exports of different profiles or register periods are not merged')
    rows = [*first.rows, *[r for r in second.rows if r.row_id not in {x.row_id for x in first.rows}]]
    return first.model_copy(update={'rows': rows, 'summary': summarise(rows),
                                    'register_synthetic': first.register_synthetic or second.register_synthetic})


def summarise(rows: list[QdmsRow]) -> dict:
    def count(values):
        out: dict[str, int] = {}
        for value in values:
            out[value] = out.get(value, 0) + 1
        return dict(sorted(out.items()))
    actions = [a for r in rows for a in r.required_actions]
    return {'rows': len(rows), 'by_layer': count(r.source_layer for r in rows), 'by_policy_status': count(r.policy_status['mapping_status'] for r in rows),
            'by_decision': count(r.reason['decision'] for r in rows), 'by_approval': count(r.human_approval.status for r in rows),
            'actions': len(actions), 'actions_by_type': count(a.qdms_type for a in actions), 'actions_by_state': count(a.state for a in actions),
            'actions_by_department': count(a.department for a in actions), 'ready_for_qdms': sum(a.state == 'READY_FOR_QDMS' for a in actions)}


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
                        for r in export.rows]}


def apply_approvals(export: QdmsExport, approvals: Approvals) -> tuple[QdmsExport, list[dict]]:
    """(the export with the approvals applied, problems). An entry is applied only to its row and only while the row's
    fingerprint is the one the reviewer saw; anything else is reported and leaves the row PENDING."""
    if approvals.profile_id != export.profile_id:
        raise ValueError(f'approvals for {approvals.profile_id}, export of {export.profile_id}')
    by_id = {r.row_id: r for r in export.rows}
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
    rows = []
    for row in export.rows:
        entry = decided.get(row.row_id)
        if entry is None:
            rows.append(row)
            continue
        state = {'APPROVED': 'READY_FOR_QDMS', 'REJECTED': 'CANCELLED', 'RETURNED': 'DRAFT'}[entry.decision]
        rows.append(row.model_copy(update={
            'human_approval': Approval(status=entry.decision, approver=entry.approver, decided_at=entry.decided_at, note=entry.note),
            'required_actions': [a.model_copy(update={'state': state}) for a in row.required_actions]}))
    return export.model_copy(update={'rows': rows, 'summary': summarise(rows)}), problems


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


def write(export: QdmsExport, directory: Path) -> dict[str, Path]:
    """export.json, export.csv (UTF-8 with BOM, for a spreadsheet), approvals-template.json, ready-actions.json."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = {'json': directory / 'export.json', 'csv': directory / 'export.csv', 'template': directory / 'approvals-template.json',
             'ready': directory / 'ready-actions.json'}
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
