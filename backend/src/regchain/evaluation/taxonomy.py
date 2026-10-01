"""Error taxonomy (v0.18): every wrong decision of a run, classified by fixed rules.

A metric says how often the analysis is wrong; this module says in which way. It reads a
finished or partial run (results.json, with results.rescored.json laid over it case by case
when present, and packets/*.json) and never re-runs a model. With a dataset file the saved
packets are re-paired with that dataset's labels in memory instead (nothing is written but the
errors files). It reads v0.17 packets (no gate chain in the trace) and v0.18 packets alike:
every v0.18 trace key is optional, and the v0.17 fields (applicability_scope, applicability_rule,
diagnostics, basis) are the fallback.

Applicability errors (a required expectation not extracted, or a three-class state that
differs from the label) take the first rule that holds, in this order; every other rule that
holds is listed under ``also``:

 1. PARSER_EXTRACTION_ERROR  a required expectation no extracted duty matched.
 2. LABEL_AMBIGUITY          the row is in DISPUTED_LABELS (the labels left open by the
                             23 September review, docs/PHASE17.md "Etiket incelemesi"); since
                             v0.19 only for a row of that dataset (DISPUTED_BY_DATASET).
 3. A deterministic gate ruled out a duty the label keeps (predicted DOES_NOT_APPLY, expected
    APPLIES or UNKNOWN): SUBJECT_SCOPE_GATE / JURISDICTION_GATE -> GLOBAL_SCOPE_MISMATCH,
    ADDRESSEE_GATE -> SUBJECT_ENTITY_MISMATCH, ENTITY_GATE -> CUSTOMER_ENTITY_MISMATCH when a
    ruled-out entity is a counterparty, else SUBJECT_ENTITY_MISMATCH; EXEMPTION_GATE -> OTHER.
 4. The label rules the duty out and the run did not (expected DOES_NOT_APPLY, predicted
    APPLIES or UNKNOWN):
    a. GLOBAL_SCOPE_MISMATCH    the company is not an obliged party at all: the subject or
                                jurisdiction gate says MISMATCH, or (v0.17 packets) the label's
                                note says "yükümlü listesinde yer almayan".
    b. SUBJECT_ENTITY_MISMATCH  the duty is addressed to someone else: the note names customs /
                                an authority ("gümrük", "ödev yüklemez"), the addressee gate says
                                MISMATCH, or the duty's subject is an authority.
    c. the label expects the entity gate to say MISMATCH: CUSTOMER_ENTITY_MISMATCH when the
       packet read a counterparty requirement (and did not rule it out), SUBJECT_ENTITY_MISMATCH
       for an obliged-party requirement; PARENT_CHILD_LEAKAGE when the packet read no entity for
       the clause and the state came from the provision-level judgement; otherwise the note
       decides (a customer family named -> CUSTOMER_ENTITY_MISMATCH, else SUBJECT_ENTITY_MISMATCH).
    d. EXEMPTION_MISSED         exclusion wording in the clause ("hariç", "istisna", "muaf",
                                "uygulanmaz" ...), explicit exclusion evidence in the trace, a
                                BASIS_NO_EXCLUSIONARY diagnostic or a v0.18 exemption gate reading.
 5. PARENT_CHILD_LEAKAGE     predicted APPLIES, expected otherwise, the state was shared from the
                             provision (APPLICABILITY_SHARED) and the clause itself names an entity
                             the gate did not settle (neither MATCH nor MISMATCH).
 6. RULE_MODEL_DISAGREEMENT  expected APPLIES, predicted UNKNOWN, and either the verified basis
                             contradicted the model (DOWNGRADED_BASIS_INCONSISTENT) or the entity
                             gate / every v0.18 gate was a clear MATCH while the model hedged
                             (model_decision UNKNOWN or POSSIBLY_APPLIES); or expected APPLIES,
                             predicted DOES_NOT_APPLY by the model against a gate MATCH (review
                             flag RULE_MODEL_DISAGREEMENT).
 7. PROFILE_TOO_AMBIGUOUS    a profile field the rules read is None, the clause gate is
                             UNDETERMINED, or v0.18 decided_by is PROFILE_INCOMPLETE /
                             PROFILE_AMBIGUOUS.
 8. MODEL_HALLUCINATION      the model's answer carried a basis or scope quote that did not verify
                             (BASIS_DROPPED / SCOPE_QUOTE_DROPPED) and the state it supported is
                             wrong.
 9. OTHER.

Coverage and conflict errors (a separate list; a row whose coverage or conflict flag differs
from the label):

 1. LABEL_AMBIGUITY          the row's coverage label is disputed, or its expected evidence text
                             occurs in no policy passage of the packet.
 2. the applicability of the row is wrong too: the coverage error follows from it and takes the
    applicability error's taxonomy.
 3. RETRIEVAL_FAILURE        an expected evidence passage exists but was not retrieved within
                             rank 10.
 4. MODEL_HALLUCINATION      the judge claimed more than the label: a relation where the label
                             expects NO_EVIDENCE, or a conflict where none is expected.
 5. OTHER                    the expected passage was retrieved and judged otherwise (an
                             under-call), or the judgement failed.

The rules are heuristics over recorded fields; a rationale names the fields each decision
rests on, so a reviewer can disagree with a classification, not guess at it.
"""
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from regchain.pilot.policies import fold
from .identifiers import clause_id

TAXONOMY = ('SUBJECT_ENTITY_MISMATCH', 'CUSTOMER_ENTITY_MISMATCH', 'GLOBAL_SCOPE_MISMATCH', 'EXEMPTION_MISSED', 'PARENT_CHILD_LEAKAGE',
            'PROFILE_TOO_AMBIGUOUS', 'RULE_MODEL_DISAGREEMENT', 'MODEL_HALLUCINATION', 'RETRIEVAL_FAILURE', 'PARSER_EXTRACTION_ERROR',
            'LABEL_AMBIGUITY', 'OTHER')
# (case_id, article, clause or '*') -> (the label dimension in dispute, why). Left open by the label review of
# 23 September 2026 (docs/PHASE17.md, "Tartışmalı bırakılanlar"); the labels keep the generator rule's value.
DISPUTED_LABELS = {
    ('C09', '6', '1'): ('applicability', 'Labelled DOES_NOT_APPLY (entity gate MISMATCH) for a payment institution with legal-entity customers '
                                         'only; APPLIES is defensible because md. 14(1)(b), 7(3) and 17(2) send the identification of the '
                                         "representative to md. 6."),
    ('C01', '8', '*'): ('applicability', "DemoPay's 'üye işyerleri' (merchants) may include associations; UNKNOWN is defensible (labelled "
                                         'DOES_NOT_APPLY / MISMATCH).'),
    ('C16', '46', '1'): ('coverage', 'aml-mixed.md: whether the retention paragraph is COVERS_TEXT or PARTIAL is disputed.'),
    ('C18', '46', '1'): ('coverage', 'The CONFLICT comes from the control register (kontrol-kaydi-zayif.csv RET-01, five years), not the '
                                     'policy; the register is counted apart from written coverage (v0.16), so where the conflict belongs '
                                     'is disputed.'),
    ('C12', '5', '2'): ('coverage', 'aml-partial.md: 5(2) is labelled PARTIAL and 5(3) is unasserted; the line between the two is disputed.'),
    ('C12', '5', '3'): ('coverage', 'aml-partial.md: 5(2) is labelled PARTIAL and 5(3) is unasserted; the line between the two is disputed.'),
    ('C19', '46', '1'): ('coverage', 'kontrol-kaydi-uygun.csv: NO_EVIDENCE is expected while an evidence substring ("sekiz yıl saklanır") '
                                     'is also given; a control row counted as evidence but not as written coverage reads as contradictory.')}
# v0.19 (B5): the disputes belong to the dataset whose labels they question. Case ids are not unique across
# datasets (an independent dataset may well have a C01 or a C12), so a row is only ever looked up in the table
# of its own dataset; a dataset without an entry here has no disputed label.
LEGACY_DATASET = 'tr-aml-v1'
DISPUTED_BY_DATASET = {LEGACY_DATASET: DISPUTED_LABELS}
NOT_OBLIGED_NOTE = 'yükümlü listesinde yer almayan'
AUTHORITY_NOTE = re.compile(r'gümrük|idaresinin işlemi|ödev yüklemez|yükümlülerin ödevi değil')
AUTHORITY_SUBJECT = re.compile(r'^\s*(?:gümrük|başkanlık|bakanlık|denetim eleman|kamu kurum|yolcu)')
CUSTOMER_NOTE = re.compile(r'müşteri')
# Exclusion wording (folded text); "dışında" only when it is not "yurt dışında".
EXCLUSION = re.compile(r'(?<![a-zçğıöşü])(?:uygulanmaz|hariç(?:tir)?|(?<!yurt )dışında(?:dır)?|istisna(?:dır|sıdır)?|kapsam dışı(?:dır|nda(?:dır)?)?|muaf(?:tır)?|'
                       r'except|exempt(?:ed)?)(?![a-zçğıöşü])')
PROFILE_FIELDS = ('jurisdictions', 'activities', 'licences', 'products', 'customer_types')
GATE_RULES = {'SUBJECT_SCOPE_GATE': 'GLOBAL_SCOPE_MISMATCH', 'JURISDICTION_GATE': 'GLOBAL_SCOPE_MISMATCH', 'ADDRESSEE_GATE': 'SUBJECT_ENTITY_MISMATCH'}
HEDGED = ('UNKNOWN', 'POSSIBLY_APPLIES')
RETRIEVAL_DEPTH = 10


def clip(text, size=300):
    text = ' '.join(str(text or '').split())
    return text if len(text) <= size else text[:size - 1] + '…'


def disputed(case_id, article, clause, dataset_id=LEGACY_DATASET):
    """(dimension, reason) when the label of this row is in dispute, else None.

    ``dataset_id`` names the dataset the row belongs to; callers that know it must pass it. It defaults
    to tr-aml-v1, the only dataset whose labels were ever disputed, for callers written before v0.19.
    """
    table = DISPUTED_BY_DATASET.get(dataset_id) or {}
    return table.get((case_id, str(article), clause_id(clause))) or table.get((case_id, str(article), '*'))


class Signals:
    """What a packet obligation recorded about its applicability, read defensively (v0.17 or v0.18)."""

    def __init__(self, obligation=None, prediction=None, company=None):
        obligation, prediction = obligation or {}, prediction or {}
        proposal = obligation.get('proposal') or {}
        self.proposal, self.company = proposal, company or {}
        self.candidate = obligation.get('candidate') or {}
        self.source_label = obligation.get('source_label') or prediction.get('source_label') or ''
        self.scope = proposal.get('applicability_scope') or {}
        self.trace = proposal.get('trace') if isinstance(proposal.get('trace'), dict) else {}
        self.gates = {g['gate']: g for g in self.trace.get('gates') or [] if isinstance(g, dict) and g.get('gate')}
        self.subject_gate = self.trace.get('subject_gate') or self.gates.get('REGULATION_SUBJECT_SCOPE') or {}
        self.diagnostics = obligation.get('diagnostics') or []
        self.codes = {d.get('code') for d in self.diagnostics if isinstance(d, dict) and d.get('code')}
        self.entities = [e for e in self.scope.get('required_entities') or self.trace.get('target_entity') or [] if isinstance(e, dict)]
        self.roles = {e.get('role') for e in self.entities}
        self.entity_match = self.scope.get('match') or prediction.get('entity_gate')
        self.rule = proposal.get('applicability_rule') or prediction.get('applicability_rule')
        self.decided_by = self.trace.get('decided_by') or prediction.get('decided_by')
        model_gate = self.trace.get('model_gate') or self.gates.get('MODEL') or {}
        self.model_decision = self.trace.get('model_decision') or prediction.get('model_decision') or (
            model_gate.get('status') if model_gate.get('status') not in (None, 'NOT_ASKED', 'FAILED') else None)
        self.review_flags = list(proposal.get('review_flags') or prediction.get('review_flags') or [])
        self.shared = 'APPLICABILITY_SHARED' in self.codes
        # v0.17 decides a clause either by the entity gate or by the provision-level judgement (rule PROVISION_LEVEL).
        self.provision_level = self.shared or self.scope.get('rule') == 'PROVISION_LEVEL' or (self.decided_by or self.rule) in (
            'MODEL', 'UPGRADED_FROM_UNKNOWN_BY_BASIS', 'DOWNGRADED_BASIS_INCONSISTENT')

    def gate_status(self, name):
        return (self.gates.get(name) or {}).get('status')

    def gates_clear_match(self):
        """v0.17: the clause gate read MATCH; v0.18: every deterministic gate is a clear MATCH / NOT_RESTRICTED / NONE."""
        deterministic = [g for name, g in self.gates.items() if name not in ('MODEL', 'FINAL_AGGREGATOR')]
        if deterministic:
            return all(g.get('clear') and g.get('status') in ('MATCH', 'NOT_RESTRICTED', 'NONE') for g in deterministic)
        return self.entity_match == 'MATCH'

    def clause_text(self):
        return self.candidate.get('source_quote') or self.scope.get('child_excerpt') or ''

    def missing_profile(self):
        return [key for key in PROFILE_FIELDS if key in self.company and self.company.get(key) is None]

    def exclusion(self):
        """Why an exclusion might have applied, or None."""
        wording = EXCLUSION.search(fold(self.clause_text()))
        if wording:
            return f'the clause says "{wording.group(0)}"'
        if self.trace.get('explicit_exclusion_evidence'):
            return 'the trace records explicit exclusion evidence'
        if 'BASIS_NO_EXCLUSIONARY' in self.codes:
            return 'an exclusionary basis condition failed (BASIS_NO_EXCLUSIONARY)'
        if self.gate_status('EXEMPTION') in ('EXEMPT', 'POSSIBLE') or 'EXEMPTION_POSSIBLE' in self.review_flags:
            return f'the exemption gate read {self.gate_status("EXEMPTION") or "POSSIBLE"}'
        return None


def applicability_rules(case_id, expected, actual, sig, dataset_id=LEGACY_DATASET):
    """[(taxonomy, rationale)] of every applicability rule that holds, in rule order."""
    found = []
    notes = expected.get('notes') or ''
    where = sig.decided_by or sig.rule
    dispute = disputed(case_id, expected['article'], expected.get('clause'), dataset_id)
    if dispute and dispute[0] == 'applicability':
        found.append(('LABEL_AMBIGUITY', 'Disputed label: ' + dispute[1]))
    if actual == 'DOES_NOT_APPLY' and expected['applicability'] != 'DOES_NOT_APPLY':
        if where in GATE_RULES:
            found.append((GATE_RULES[where], f'{where} ruled the duty out; the label keeps it {expected["applicability"]}.'))
        elif where == 'ENTITY_GATE':
            ruled_out = [e for e in sig.entities if e.get('match') == 'MISMATCH']
            counterparty = any(e.get('role') == 'counterparty' for e in ruled_out)
            named = ', '.join(f'{e.get("role")} {e.get("type")} ("{e.get("text")}")' for e in ruled_out) or 'an entity'
            found.append(('CUSTOMER_ENTITY_MISMATCH' if counterparty else 'SUBJECT_ENTITY_MISMATCH',
                          f'The entity gate ruled out {named}; the label keeps the clause {expected["applicability"]}.'))
        elif where == 'EXEMPTION_GATE':
            found.append(('OTHER', 'The exemption gate ruled out a clause the label keeps.'))
    if expected['applicability'] == 'DOES_NOT_APPLY' and actual != 'DOES_NOT_APPLY':
        subject, jurisdiction = sig.subject_gate.get('status'), sig.gate_status('JURISDICTION')
        if subject == 'MISMATCH' or jurisdiction == 'MISMATCH' or NOT_OBLIGED_NOTE in notes:
            basis = ('the subject gate read MISMATCH' if subject == 'MISMATCH' else 'the jurisdiction gate read MISMATCH'
                     if jurisdiction == 'MISMATCH' else 'the label: the company is not in the obliged-party list (md. 4)')
            found.append(('GLOBAL_SCOPE_MISMATCH', f'Not an obliged party ({basis}), yet {actual}.'))
        subject_text = fold(sig.candidate.get('subject') or '')
        if AUTHORITY_NOTE.search(fold(notes)) or sig.gate_status('COMPANY_ENTITY') == 'MISMATCH' or AUTHORITY_SUBJECT.search(subject_text):
            found.append(('SUBJECT_ENTITY_MISMATCH', f'The duty is addressed to an authority or another actor (subject "{clip(sig.candidate.get("subject"), 60)}"), '
                                                     f'not the company; predicted {actual}.'))
        if expected.get('entity_gate') == 'MISMATCH':
            if 'counterparty' in sig.roles:
                found.append(('CUSTOMER_ENTITY_MISMATCH', f'The clause concerns a customer family the profile lacks; the gate read the counterparty '
                                                          f'requirement as {sig.entity_match}, not MISMATCH.'))
            elif 'obliged_party' in sig.roles:
                found.append(('SUBJECT_ENTITY_MISMATCH', f'The clause binds a kind of obliged party the company is not; the gate read it as {sig.entity_match}.'))
            elif sig.provision_level:
                found.append(('PARENT_CHILD_LEAKAGE', 'The label rules the clause out by its entity, the packet read no entity for the clause, '
                                                      f'and the provision-level state ({actual}) was inherited.'))
            else:
                found.append(('CUSTOMER_ENTITY_MISMATCH' if CUSTOMER_NOTE.search(fold(notes)) else 'SUBJECT_ENTITY_MISMATCH',
                              'The label expects the entity gate to rule the clause out; the packet recorded no entity requirement.'))
        reason = sig.exclusion()
        if reason:
            found.append(('EXEMPTION_MISSED', f'Exclusion signal ({reason}), yet {actual}.'))
    if actual == 'APPLIES' and expected['applicability'] != 'APPLIES' and sig.shared:
        loose = [e for e in sig.entities if e.get('source') == 'clause' and e.get('match') not in ('MATCH', 'MISMATCH')]
        if loose:
            found.append(('PARENT_CHILD_LEAKAGE', f'APPLIES was shared from the provision while the clause names {loose[0].get("type")} '
                                                  f'("{loose[0].get("text")}"), read {loose[0].get("match")}.'))
    if expected['applicability'] == 'APPLIES' and actual == 'UNKNOWN':
        downgraded = 'DOWNGRADED_BASIS_INCONSISTENT' in (sig.rule, sig.decided_by) or 'DOWNGRADED_BASIS_INCONSISTENT' in sig.codes
        if downgraded:
            found.append(('RULE_MODEL_DISAGREEMENT', 'The model said DOES_NOT_APPLY against its own verified basis (downgraded to UNKNOWN); '
                                                     f'entity gate {sig.entity_match}.'))
        elif sig.gates_clear_match() and sig.model_decision in HEDGED:
            found.append(('RULE_MODEL_DISAGREEMENT', f'The gates read a clear MATCH and the model hedged ({sig.model_decision}).'))
    if expected['applicability'] == 'APPLIES' and actual == 'DOES_NOT_APPLY' and where not in (*GATE_RULES, 'ENTITY_GATE', 'EXEMPTION_GATE'):
        if 'RULE_MODEL_DISAGREEMENT' in sig.review_flags or (sig.gates_clear_match() and sig.model_decision == 'DOES_NOT_APPLY'):
            found.append(('RULE_MODEL_DISAGREEMENT', 'The model said DOES_NOT_APPLY against a gate MATCH.'))
    missing = sig.missing_profile()
    undetermined = sig.entity_match == 'UNDETERMINED' or 'UNDETERMINED' in (sig.gate_status('CHILD_CLAUSE'), sig.gate_status('CUSTOMER_ENTITY'))
    if missing or undetermined or where in ('PROFILE_INCOMPLETE', 'PROFILE_AMBIGUOUS'):
        why = (f'profile fields not stated: {", ".join(missing)}' if missing else 'the clause gate read UNDETERMINED' if undetermined
               else f'decided by {where}')
        found.append(('PROFILE_TOO_AMBIGUOUS', f'The profile does not settle the clause ({why}); predicted {actual}, labelled {expected["applicability"]}.'))
    # A dropped basis or quote is itself proof that a model answered (the rules cite nothing to drop).
    dropped = sorted(sig.codes & {'BASIS_DROPPED', 'SCOPE_QUOTE_DROPPED'})
    if dropped:
        found.append(('MODEL_HALLUCINATION', f'The model cited basis or scope text that did not verify ({", ".join(dropped)}) and the state is wrong.'))
    if not found:
        gates = ', '.join(f'{name} {status}' for name, status in ((n, g.get('status')) for n, g in sig.gates.items()) if name != 'FINAL_AGGREGATOR')
        model = sig.gate_status('MODEL') or ('answered' if sig.model_decision else 'no answer recorded')
        found.append(('OTHER', f'No rule matched: decided by {where}, model {sig.model_decision or model}, '
                               + (f'gates {gates}.' if gates else f'entity gate {sig.entity_match}.')))
    return found


def coverage_rules(case_id, item, sig, applicability_record=None, dataset_id=LEGACY_DATASET):
    """[(taxonomy, rationale)] of every coverage/conflict rule that holds, in rule order."""
    found = []
    expected = item['expected']
    dispute = disputed(case_id, expected['article'], expected.get('clause'), dataset_id)
    if dispute and dispute[0] == 'coverage':
        found.append(('LABEL_AMBIGUITY', 'Disputed label: ' + dispute[1]))
    retrieval = item.get('retrieval')
    if retrieval and expected.get('evidence') and not retrieval.get('expected_passages'):
        found.append(('LABEL_AMBIGUITY', 'The expected evidence text occurs in no policy passage of the packet.'))
    if applicability_record is not None:
        found.append((applicability_record['taxonomy'], f'Follows from the applicability error ({applicability_record["expected"]} -> '
                                                        f'{applicability_record["actual"]}).'))
    coverage = item.get('coverage')
    conflict = item.get('conflict')
    if retrieval and retrieval.get('expected_passages') and (retrieval.get('rank') is None or retrieval['rank'] > RETRIEVAL_DEPTH):
        found.append(('RETRIEVAL_FAILURE', 'The expected passage was ' + ('not retrieved' if retrieval.get('rank') is None else
                                                                          f'retrieved at rank {retrieval["rank"]}') + f' (depth {RETRIEVAL_DEPTH}).'))
    over = (coverage and coverage[0] == 'NO_EVIDENCE' and coverage[1] in ('COVERS_TEXT', 'PARTIAL', 'CONFLICT'))
    if over or (conflict and conflict[1] and not conflict[0]):
        found.append(('MODEL_HALLUCINATION', 'The judge claimed ' + (f'{coverage[1]} where the label expects NO_EVIDENCE' if over else
                                                                     'a conflict the label does not have') + '.'))
    failed = [n.get('code') for d in sig.diagnostics for r in d.get('results') or [] for n in r.get('notes') or []
              if n.get('code') in ('ProviderFailure', 'JUDGEMENT_INVALID')]
    if failed:
        found.append(('OTHER', f'A passage judgement failed ({", ".join(sorted(set(failed)))}).'))
    elif retrieval and retrieval.get('rank') is not None and retrieval['rank'] <= RETRIEVAL_DEPTH:
        found.append(('OTHER', f'The expected passage was retrieved at rank {retrieval["rank"]} and judged '
                               f'{coverage[1] if coverage else "otherwise"}' + ('' if retrieval.get('hit') else ' without being cited') + '.'))
    if not found:
        found.append(('OTHER', f'No rule matched: judged {coverage[1] if coverage else "—"}, conflict {conflict[1] if conflict else "—"}; '
                               'the label names no evidence passage to check retrieval against.'))
    return found


def company_view(company):
    company = company or {}
    return {key: company.get(key) for key in ('id', 'name', 'licences', 'activities', 'customer_types')}


def regulation_basis(sig):
    subject = sig.subject_gate
    evidence = subject.get('evidence') or {}
    return {'scope_quotes': [clip(q.get('quote')) for q in (sig.proposal.get('scope_evidence') or [])[:4] if isinstance(q, dict)],
            'subject_gate': {'status': subject.get('status'), 'reason': clip(subject.get('reason')),
                             'list_source_label': subject.get('list_source_label') or evidence.get('source_label'),
                             'list_item': clip(evidence.get('list_item'), 200) or None,
                             'matched_items': subject.get('matched_items') or evidence.get('matched_items')} if subject else None,
            'basis': [{'condition': clip(b.get('regulatory_condition'), 200), 'company_fact': clip(b.get('company_fact'), 120),
                       'match': b.get('match'), 'exclusionary': b.get('exclusionary')}
                      for b in (sig.proposal.get('basis') or [])[:6] if isinstance(b, dict)]}


def rule_view(sig):
    return {'rule_decision': sig.trace.get('rule_decision'), 'applicability_rule': sig.rule, 'decided_by': sig.decided_by,
            'entity_gate': sig.entity_match,
            'required_entities': [f'{e.get("role")} {e.get("type")} ("{e.get("text")}", {e.get("source")}): {e.get("match")}' for e in sig.entities],
            'gates': {name: g.get('status') for name, g in sig.gates.items()}, 'review_flags': sig.review_flags}


def record(case_id, item, sig, prediction, rules, kind):
    expected = item['expected']
    taxonomy, rationale = rules[0] if rules else ('OTHER', 'No rule matched.')
    also = list(dict.fromkeys(t for t, _ in rules[1:] if t != taxonomy))
    row = {'case_id': case_id, 'where': f'{case_id} md.{expected["article"]}{expected.get("clause") or ""}',
           'obligation_id': (prediction or {}).get('obligation_id'), 'obligation_key': (prediction or {}).get('key') or item.get('prediction'),
           'expectation': {k: expected.get(k) for k in ('article', 'clause', 'action_keywords', 'entity_gate', 'required', 'notes')},
           'company_profile': company_view(sig.company),
           'regulation_basis': regulation_basis(sig),
           'child_clause': {'provision': sig.source_label or None, 'clause': sig.scope.get('child_clause'), 'excerpt': clip(sig.clause_text()) or None,
                            'subject': sig.candidate.get('subject')},
           'rule_decision': rule_view(sig),
           'model_decision': {'state': sig.model_decision, 'provision_state': sig.scope.get('provision_state'),
                              'mode': (sig.trace.get('model_gate') or {}).get('mode')},
           'taxonomy': taxonomy, 'rationale': rationale, 'also': also}
    if kind == 'applicability':
        row['expected'] = expected['applicability']
        row['actual'] = item['applicability'][1] if item.get('matched') else 'NOT_EXTRACTED'
        row['evidence'] = {'positive': [clip(e.get('regulation_text'), 160) for e in (sig.trace.get('positive_evidence') or [])[:3] if isinstance(e, dict)],
                           'negative': [clip(e.get('regulation_text') or e.get('quote'), 160) for e in (sig.trace.get('negative_evidence') or [])[:3]
                                        if isinstance(e, dict)],
                           'reason': clip(sig.proposal.get('applicability_reason'), 400) or None}
    else:
        coverage, conflict = item.get('coverage'), item.get('conflict')
        row['expected'] = {'coverage': coverage[0] if coverage else expected.get('coverage'), 'conflict': conflict[0] if conflict else expected.get('conflict')}
        row['actual'] = {'coverage': coverage[1] if coverage else (prediction or {}).get('coverage'), 'conflict': conflict[1] if conflict else (prediction or {}).get('conflict')}
        retrieval = item.get('retrieval') or {}
        row['evidence'] = {'expected_evidence': expected.get('evidence') or [], 'expected_passages': retrieval.get('expected_passages'),
                           'expected_rank': retrieval.get('rank'), 'cited_expected': retrieval.get('hit'),
                           'cited': [{'source_id': q.get('source_id'), 'quote': clip(q.get('quote'), 200)}
                                     for q in (sig.proposal.get('policy_evidence') or [])[:4] if isinstance(q, dict)],
                           'judged': [{'source_id': c.get('source_id'), 'relation': c.get('relation')} for c in (prediction or {}).get('policy_checks') or []][:8],
                           'reason': clip(sig.proposal.get('coverage_reason'), 400) or None}
    return row


def applicability_error(case_id, item, prediction=None, obligation=None, company=None, dataset_id=LEGACY_DATASET):
    """The taxonomy record of one wrong (or unextracted) applicability row."""
    sig = Signals(obligation, prediction, company)
    if not item.get('matched'):
        rules = [('PARSER_EXTRACTION_ERROR', 'No extracted duty matched this required expectation (article, clause and action keywords).')]
    else:
        rules = applicability_rules(case_id, item['expected'], item['applicability'][1], sig, dataset_id)
    return record(case_id, item, sig, prediction, rules, 'applicability')


def coverage_error(case_id, item, prediction=None, obligation=None, company=None, applicability_record=None, dataset_id=LEGACY_DATASET):
    """The taxonomy record of one row whose coverage or conflict flag differs from the label."""
    sig = Signals(obligation, prediction, company)
    return record(case_id, item, sig, prediction, coverage_rules(case_id, item, sig, applicability_record, dataset_id), 'coverage')


def classify_results(results, packets, dataset_id=LEGACY_DATASET):
    """results: harness case results (scored); packets: {case_id: payload}; dataset_id: the dataset the labels
    belong to (it selects the disputed labels). -> the errors report body."""
    applicability, coverage, case_errors, unexpected = [], [], [], 0
    for result in results:
        case_id = result['case_id']
        if result.get('error'):
            case_errors.append({'case_id': case_id, 'error': result['error']})
            continue
        payload = packets.get(case_id) or {}
        obligations = {o.get('id'): o for o in payload.get('obligations') or []}
        by_key = {}
        for prediction in result.get('predictions') or []:
            by_key.setdefault(prediction['key'], prediction)
        company = payload.get('company') or {}
        unexpected += len(result.get('extras') or [])
        for item in result.get('scored') or []:
            prediction = by_key.get(item.get('prediction')) if item.get('matched') else None
            obligation = obligations.get((prediction or {}).get('obligation_id'))
            if not item.get('matched'):
                if item['expected'].get('required'):
                    applicability.append(applicability_error(case_id, item, None, None, company, dataset_id))
                continue
            app_record = None
            if item['applicability'][0] != item['applicability'][1]:
                app_record = applicability_error(case_id, item, prediction, obligation, company, dataset_id)
                applicability.append(app_record)
            coverage_wrong = 'coverage' in item and item['coverage'][0] != item['coverage'][1]
            conflict_wrong = 'conflict' in item and item['conflict'][0] != item['conflict'][1]
            if coverage_wrong or conflict_wrong:
                coverage.append(coverage_error(case_id, item, prediction, obligation, company, app_record, dataset_id))
    return {'counts': {'applicability': dict(Counter(r['taxonomy'] for r in applicability)),
                       'coverage': dict(Counter(r['taxonomy'] for r in coverage))},
            'applicability_errors': applicability, 'coverage_errors': coverage, 'case_errors': case_errors,
            'unexpected_candidates': unexpected}


def load_results(run_dir: Path):
    """results.json with results.rescored.json laid over it case by case -> (results, {case_id: file})."""
    run_dir = Path(run_dir)
    results = json.loads((run_dir / 'results.json').read_text(encoding='utf-8'))
    sources = {r['case_id']: 'results.json' for r in results}
    rescored_path = run_dir / 'results.rescored.json'
    if rescored_path.is_file():
        # A rescore made while the run was still going covers only the cases done by then; the later ones
        # stay as the run scored them.
        rescored = {r['case_id']: r for r in json.loads(rescored_path.read_text(encoding='utf-8'))}
        for index, result in enumerate(results):
            if result['case_id'] in rescored:
                results[index] = rescored[result['case_id']]
                sources[result['case_id']] = 'results.rescored.json'
    return results, sources


def load_packets(run_dir: Path, case_ids):
    packets = {}
    for case_id in case_ids:
        path = Path(run_dir) / 'packets' / f'{case_id}.json'
        if path.is_file():
            packets[case_id] = json.loads(path.read_text(encoding='utf-8'))['events'][0]['payload']
    return packets


def run_results(run_dir: Path, dataset_path=None):
    """(results, {case_id: source}, packets, scored against) of a run directory.

    With a dataset the saved packets are re-paired with its labels in memory (what
    `report --rescore` does, without writing results.rescored.json).
    """
    from .harness import load_dataset, prediction_rows, score_case
    run_dir = Path(run_dir)
    results, sources = load_results(run_dir)
    packets = load_packets(run_dir, [r['case_id'] for r in results])
    if not dataset_path:
        return results, sources, packets, 'the labels the results were scored with'
    dataset = load_dataset(Path(dataset_path))
    cases = {c.case_id: c for c in dataset.cases}
    for index, result in enumerate(results):
        case, payload = cases.get(result['case_id']), packets.get(result['case_id'])
        if result.get('error') or case is None or payload is None:
            continue
        predictions = prediction_rows(payload, case.regulation_id)
        scored, extras = score_case(case, payload, predictions)
        results[index] = {**result, 'predictions': predictions, 'scored': scored, 'extras': extras}
        sources[result['case_id']] = 'packets re-paired with the dataset'
    return results, sources, packets, str(Path(dataset_path))


def classify_run(run_dir: Path, dataset_path=None):
    """The errors report of a run directory (nothing is written)."""
    run_dir = Path(run_dir)
    results, sources, packets, scored_against = run_results(run_dir, dataset_path)
    manifest_path = run_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}
    if dataset_path:
        from .harness import load_dataset
        dataset_id = load_dataset(Path(dataset_path)).dataset_id
    else:
        dataset_id = (manifest.get('dataset') or {}).get('id') or LEGACY_DATASET
    body = classify_results(results, packets, dataset_id)
    return {'format': 'cardaman-error-taxonomy-v1', 'created_at': datetime.now(timezone.utc).isoformat(), 'run': str(run_dir),
            'dataset_id': dataset_id, 'manifest_sha256': manifest.get('manifest_sha256'), 'scored_against': scored_against, 'sources': sources,
            'cases': len(results), 'packets': len(packets), 'taxonomy': list(TAXONOMY),
            'rules': 'regchain.evaluation.taxonomy (module docstring)', **body}


def errors_markdown(report):
    lines = [f'# Error taxonomy — `{report["run"]}`', '',
             f'{report["cases"]} cases ({report["packets"]} packets), scored against {report["scored_against"]}. '
             f'{len(report["applicability_errors"])} applicability errors, {len(report["coverage_errors"])} coverage/conflict errors, '
             f'{len(report["case_errors"])} failed cases, {report["unexpected_candidates"]} unexpected candidates (not classified).', '',
             'Rules: the docstring of `regchain.evaluation.taxonomy`; the first rule that holds names the error, the others are listed under *also*.', '',
             '| Taxonomy | Applicability | Coverage / conflict |', '|---|---|---|']
    for name in TAXONOMY:
        a, c = report['counts']['applicability'].get(name, 0), report['counts']['coverage'].get(name, 0)
        if a or c:
            lines.append(f'| {name} | {a} | {c} |')
    lines.append(f'| **total** | {len(report["applicability_errors"])} | {len(report["coverage_errors"])} |')
    for title, rows, show in (('Applicability errors', report['applicability_errors'], lambda r: f'{r["expected"]} → {r["actual"]}'),
                              ('Coverage and conflict errors', report['coverage_errors'],
                               lambda r: f'{r["expected"]["coverage"]} → {r["actual"]["coverage"]}'
                                         + (f', conflict {r["expected"]["conflict"]} → {r["actual"]["conflict"]}'
                                            if r['expected']['conflict'] != r['actual']['conflict'] and r['expected']['conflict'] is not None else ''))):
        lines += ['', f'## {title}']
        groups = {}
        for row in rows:
            groups.setdefault(row['taxonomy'], []).append(row)
        if not groups:
            lines += ['', 'None.']
        for name in TAXONOMY:
            if name not in groups:
                continue
            lines += ['', f'### {name} ({len(groups[name])})', '', '| Where | Exp → actual | Rationale | Also |', '|---|---|---|---|']
            for row in groups[name]:
                lines.append(f'| {row["where"]} | {show(row)} | {row["rationale"].replace("|", "/")} | {", ".join(row["also"]) or ""} |')
    if report['case_errors']:
        lines += ['', '## Failed cases', ''] + [f'- {e["case_id"]}: {e["error"]}' for e in report['case_errors']]
    return '\n'.join(lines) + '\n'


def write_errors(run_dir: Path, dataset_path=None, out_dir=None):
    """errors.json and errors.md in the run directory (or out_dir); returns (report, json path, md path)."""
    report = classify_run(run_dir, dataset_path)
    target = Path(out_dir) if out_dir else Path(run_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / 'errors.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    (target / 'errors.md').write_text(errors_markdown(report), encoding='utf-8')
    return report, target / 'errors.json', target / 'errors.md'
