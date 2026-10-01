"""Candidate statements, escalation, adjudication and the decision (no model, no network).

The strong model is a scripted provider here: what is tested is what the pipeline does with an answer - which rows are
asked about at all, that a quote is checked against its statement, that a record replays, and that no answer of a model
turns a decision of the rule comparer around. What the real model answers is measured on BEVERAGE_TR_DEV_V2
(test_tr_devset.py replays the recorded run).
"""
import json
import tempfile
import unittest
from pathlib import Path

from regchain.tr.adjudicate import (Adjudicator, ClauseAdjudicator, assess_obligation, assess_profile, clause_escalation, clause_text,
                                    clause_verdict, disagreement, escalate, load_adjudications, load_clause_adjudications,
                                    model_coverage, validate)
from regchain.tr.compare import coverage_of, documents_in_force, load_register, relate
from regchain.tr.corpus import CorpusStore
from regchain.tr.extraction import extract_regulation, route
from regchain.tr.packs import Registry
from regchain.tr.profile import load_pilot_profiles
from regchain.tr.semantic import SimilarityTable, candidates, duty_text, lexical_similarity, opposes

REGISTRY = Registry.load()
STORE = CorpusStore()
PROFILES = load_pilot_profiles(REGISTRY.vocabulary)
BREWER = PROFILES['ALCOHOL_GROUP_EFES_TYPE']
BOTTLER = PROFILES['NON_ALCOHOL_GROUP_COCA_COLA_TYPE']
BREWER_REGISTER = load_register(BREWER.profile_id)
BOTTLER_REGISTER = load_register(BOTTLER.profile_id)
SALES_BYLAW = 'TR:YONETMELIK:ALKOLLU_ICKI_SATIS_SUNUM'
ENERGY = 'TR:TEBLIG:TGK_ENERJI_ICECEKLERI'
LICENCE = 'Yönetmelik 14646 md. 6/f.1/b.e'          # "Satış belgesi ... asılır": the register says it in its own words
PLACES = 'Tebliğ 23706 md. 11/f.2'                  # no energy drinks in sports facilities, school canteens, hospitals
_EXTRACTED = {}


def extracted(regulation_id):
    if regulation_id not in _EXTRACTED:
        _EXTRACTED[regulation_id] = extract_regulation(regulation_id, REGISTRY, STORE)
    return _EXTRACTED[regulation_id]


def duty(regulation_id, ref, profile, target_id):
    """(obligation, decision) of the one obligation of `ref` that applies on the target."""
    for item in extracted(regulation_id)[1]:
        if item.provision_ref == ref:
            for decision in route(item, profile, REGISTRY, STORE)[0]:
                if decision.target_id == target_id and decision.status in ('APPLIES', 'PARTIAL'):
                    return item, decision
    raise AssertionError((ref, target_id))


def in_force(register, profile, decision):
    documents = {d.document_id for d in documents_in_force(register, profile, decision)}
    return [p for p in register.passages if p.document_id in documents]


def rule_reading(item, passages):
    related = [r for r in (relate(item, p, REGISTRY.vocabulary) for p in passages) if r.relation != 'UNRELATED']
    return related, coverage_of(related)


class Scripted:
    """A provider that answers from a script."""
    name, model_version, base_url = 'scripted', 'scripted@test', 'http://localhost:11434'

    def __init__(self, *answers):
        self.answers, self.requests = list(answers), []

    def generate_structured(self, prompt, payload, schema):
        self.requests.append(payload)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return json.dumps(answer, ensure_ascii=False)


class Embedder:
    """Unit vectors by keyword: texts about hanging something up point one way, everything else another."""

    def embed(self, texts):
        return [[1.0, 0.0] if 'asılır' in text else [0.0, 1.0] for text in texts]

    def manifest(self):
        return {'model_version': 'keyword@test'}


def judgement(number, relation, quote='', missing=()):
    return {'id': f'S{number}', 'relation': relation, 'quote': quote, 'missing': list(missing), 'reason': 'scripted'}


class CandidateTests(unittest.TestCase):
    def test_a_word_changed_at_its_end_keeps_its_body(self):
        licence = 'Satış belgesi işyerinde görünür bir yere asılır.'
        self.assertGreater(lexical_similarity(licence, 'Satış belgeleri tüketicilerin görebileceği yerde asılı bulundurulur.'), 0.30)
        self.assertLess(lexical_similarity(licence, 'Ambalaj atıkları kaynağında ayrı biriktirilir.'), 0.10)

    def test_a_statement_in_the_companys_own_words_is_a_candidate_and_nothing_more(self):
        item, decision = duty(SALES_BYLAW, LICENCE, BREWER, 'ALC-INT-SALES')
        passages = in_force(BREWER_REGISTER, BREWER, decision)
        related, (coverage, reasons) = rule_reading(item, passages)
        self.assertEqual((related, coverage, reasons), ([], 'NO_EVIDENCE', ['NO_RELATED_STATEMENT']))
        found = candidates(item, passages, REGISTRY.vocabulary)
        self.assertEqual([(c.passage_id, c.channels, c.semantic) for c in found], [('SOP-SALES-02#4', ['LEXICAL'], None)])
        self.assertIn('asılır', found[0].quote)

    def test_recorded_similarities_are_replayed_without_an_embedder(self):
        item, decision = duty(SALES_BYLAW, LICENCE, BREWER, 'ALC-INT-SALES')
        passages = in_force(BREWER_REGISTER, BREWER, decision)
        table = SimilarityTable().add([duty_text(item)], [p.text for p in passages], Embedder())
        with tempfile.TemporaryDirectory() as folder:
            table.save(Path(folder) / 'similarities.json')
            replayed = SimilarityTable.load(Path(folder) / 'similarities.json')
        self.assertEqual((replayed.model_version, replayed.pairs), ('keyword@test', table.pairs))
        found = candidates(item, passages, REGISTRY.vocabulary, replayed, lexical_floor=9.0)       # the semantic channel alone
        self.assertEqual([(c.passage_id, c.channels, c.semantic) for c in found], [('SOP-SALES-02#4', ['SEMANTIC'], 1.0)])
        self.assertIsNone(replayed.get(duty_text(item), 'a statement that was never paired'))

    def test_a_permission_next_to_a_prohibition_is_kept_whatever_its_rank(self):
        item, decision = duty(ENERGY, PLACES, BOTTLER, 'NONALC-BOTTLING')
        passages = in_force(BOTTLER_REGISTER, BOTTLER, decision)
        related, _ = rule_reading(item, passages)
        found = candidates(item, passages, REGISTRY.vocabulary, exclude={r.passage_id for r in related}, top_k=1)
        self.assertEqual(len(found), 2)
        samples = found[1]                 # "Üniversite kampüslerindeki etkinliklerde enerji içeceği numune dağıtımı yapılabilir."
        self.assertEqual(samples.passage_id, 'POL-MKT-02#3')
        self.assertLessEqual({'OPPOSED_POLARITY', 'STATEMENT_PERMITS'}, set(samples.signals))
        self.assertTrue(opposes(samples, item))
        self.assertFalse(opposes(found[0], item))


class EscalationTests(unittest.TestCase):
    def test_nothing_related_and_no_candidate_is_decided_without_a_model(self):
        item, decision = duty('TR:KANUN:4250', 'Kanun 4250 md. 6/f.5/c.1', BREWER, 'ALC-INT-SALES')
        passages = in_force(BREWER_REGISTER, BREWER, decision)
        related, (coverage, reasons) = rule_reading(item, passages)
        found = candidates(item, passages, REGISTRY.vocabulary, exclude={r.passage_id for r in related})
        self.assertEqual((coverage, found), ('PARTIAL', []))
        self.assertIsNone(escalate(item, coverage, reasons, related, found))
        self.assertIsNone(escalate(item, 'NO_EVIDENCE', ['NO_RELATED_STATEMENT'], [], []))

    def test_a_candidate_the_rules_could_not_relate_is_read_as_a_possible_paraphrase(self):
        item, decision = duty(SALES_BYLAW, LICENCE, BREWER, 'ALC-INT-SALES')
        passages = in_force(BREWER_REGISTER, BREWER, decision)
        found = candidates(item, passages, REGISTRY.vocabulary)
        escalation = escalate(item, 'NO_EVIDENCE', ['NO_RELATED_STATEMENT'], [], found)
        self.assertEqual((escalation.reasons, escalation.statements), (['SEMANTIC_PARAPHRASE'], ['SOP-SALES-02#4']))

    def test_an_element_check_of_the_rules_sends_nothing_to_the_model_but_a_possible_conflict(self):
        item, decision = duty(ENERGY, PLACES, BOTTLER, 'NONALC-BOTTLING')
        passages = in_force(BOTTLER_REGISTER, BOTTLER, decision)
        related, (coverage, reasons) = rule_reading(item, passages)
        self.assertEqual((coverage, reasons), ('PARTIAL', ['PLACE_MISSING:HEALTH_FACILITY']))
        found = candidates(item, passages, REGISTRY.vocabulary, exclude={r.passage_id for r in related})
        harmless = [c for c in found if not opposes(c, item)]
        self.assertTrue(harmless)
        self.assertIsNone(escalate(item, coverage, reasons, related, harmless))        # similar statements alone: the rule decision stands
        escalation = escalate(item, coverage, reasons, related, found)
        # the permission first, then the statement the rules related, so that the model reads both
        self.assertEqual((escalation.reasons, escalation.statements), (['POSSIBLE_CONFLICT'], ['POL-MKT-02#3', 'POL-MKT-02#1']))
        self.assertIsNone(escalate(item, 'CONFLICT', ['PERMITS_PROHIBITED_ACT'], related, found))   # the rules already see a conflict

    def test_an_open_coverage_and_a_contested_reading_are_escalated(self):
        item, decision = duty(SALES_BYLAW, LICENCE, BREWER, 'ALC-INT-SALES')
        found = candidates(item, in_force(BREWER_REGISTER, BREWER, decision), REGISTRY.vocabulary)
        self.assertEqual(escalate(item, 'UNKNOWN', ['UNCLEAR_STATEMENT'], [], []).reasons, ['UNKNOWN_COVERAGE'])
        self.assertEqual(escalate(item, 'NO_EVIDENCE', ['NO_RELATED_STATEMENT'], [], found, contested=True).reasons,
                         ['SEMANTIC_PARAPHRASE', 'READER_DISAGREEMENT'])
        self.assertIsNone(escalate(item, 'NO_EVIDENCE', ['NO_RELATED_STATEMENT'], [], [], contested=True))   # nothing to read it against


class AdjudicatorTests(unittest.TestCase):
    def setUp(self):
        self.item, decision = duty(SALES_BYLAW, LICENCE, BREWER, 'ALC-INT-SALES')
        by_id = {p.passage_id: p for p in in_force(BREWER_REGISTER, BREWER, decision)}
        self.passages = [by_id['SOP-SALES-02#4'], by_id['SOP-SALES-02#1']]

    def test_a_quote_that_is_not_in_its_statement_is_no_judgement(self):
        answer = {'judgements': [judgement(1, 'STATES_DUTY', 'Satış belgesi her zaman kapıda durur'),
                                 judgement(2, 'CONTRADICTS', 'on sekiz yaşını doldurmamış kişilere satılamaz')]}
        first, second = validate(answer, self.passages)
        self.assertEqual((first['grounded'], first['rule_relation']), (False, 'UNCLEAR'))
        self.assertEqual((second['grounded'], second['rule_relation']), (True, 'CONFLICTS'))
        # the check is exact: the right words with other capitals are no quote (counted as a grounding loss, not waved through)
        loose = validate({'judgements': [judgement(1, 'STATES_DUTY', 'satış belgesi, Satış noktasının içinde'), judgement(2, 'UNRELATED')]}, self.passages)
        self.assertEqual((loose[0]['grounded'], loose[0]['rule_relation']), (False, 'UNCLEAR'))
        # "states the duty" that names something missing is a part, by its own account
        part = validate({'judgements': [judgement(1, 'STATES_DUTY', 'bir yere asılır', missing=['okunabilecek']), judgement(2, 'UNRELATED')]}, self.passages)
        self.assertEqual((part[0]['relation'], part[0]['rule_relation'], part[0]['missing']), ('STATES_DUTY', 'PARTIAL', ['okunabilecek']))
        skipped = validate({'judgements': [judgement(1, 'UNRELATED')]}, self.passages)
        self.assertEqual([j['rule_relation'] for j in skipped], ['UNRELATED', 'UNCLEAR'])
        self.assertEqual(skipped[1]['reason'], 'not answered')

    def test_the_models_coverage_is_conflict_first_and_unknown_when_nothing_is_grounded(self):
        rows = lambda *relations: [{'rule_relation': r} for r in relations]
        self.assertEqual(model_coverage(rows('SUPPORTS', 'CONFLICTS')), 'CONFLICT')
        self.assertEqual(model_coverage(rows('PARTIAL', 'SUPPORTS', 'UNRELATED')), 'COVERS_TEXT')
        self.assertEqual(model_coverage(rows('UNRELATED', 'PARTIAL')), 'PARTIAL')
        self.assertEqual(model_coverage(rows('UNRELATED', 'UNCLEAR')), 'NO_EVIDENCE')
        self.assertEqual(model_coverage(rows('UNCLEAR', 'UNCLEAR')), 'UNKNOWN')

    def test_an_adjudication_is_recorded_and_replayed_without_a_model(self):
        provider = Scripted({'judgements': [judgement(1, 'STATES_DUTY', 'tüketicilerin görebileceği bir yere asılır'), judgement(2, 'UNRELATED')]})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'adjudications.jsonl'
            live = Adjudicator(provider, path)
            record, elapsed = live(self.item, self.passages, ['SEMANTIC_PARAPHRASE'])
            again, again_elapsed = live(self.item, self.passages, ['SEMANTIC_PARAPHRASE'])       # the same duty and statements: no second call
            header, recorded = load_adjudications(path)
        self.assertEqual((record['coverage'], record['status'], live.calls, again is record, again_elapsed), ('COVERS_TEXT', 'OK', 1, True, 0))
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(len(provider.requests), 1)
        request = provider.requests[0]
        self.assertEqual([s['id'] for s in request['statements']], ['S1', 'S2'])
        self.assertTrue(request['duty']['text'].endswith('asılır.'), request['duty']['text'])
        self.assertEqual((header['format'], header['runtime']['model_version']), ('cardaman-tr-adjudications/1', 'scripted@test'))
        replay = Adjudicator(recorded=recorded)
        replayed, replay_elapsed = replay(self.item, self.passages, ['SEMANTIC_PARAPHRASE'])
        self.assertEqual((replayed['judgements'], replay_elapsed, replay.calls), (record['judgements'], 0, 0))
        self.assertEqual(replay(self.item, self.passages[:1], ['SEMANTIC_PARAPHRASE']), (None, 0))   # other statements: not in the record

    def test_a_thinking_model_that_runs_out_is_asked_once_more_without_thinking(self):
        thinking = Scripted(RuntimeError('Provider output incomplete or truncated'))
        thinking.quick = Scripted({'judgements': [judgement(1, 'STATES_DUTY', 'bir yere asılır'), judgement(2, 'UNRELATED')]})
        record, _ = Adjudicator(thinking)(self.item, self.passages, ['SEMANTIC_PARAPHRASE'])
        self.assertEqual(record['coverage'], 'COVERS_TEXT')
        self.assertTrue(record['status'].startswith('OK_WITHOUT_THINKING after FAILED: RuntimeError'), record['status'])
        self.assertEqual((len(thinking.requests), len(thinking.quick.requests)), (1, 1))

    def test_the_lead_in_of_a_list_is_context_not_part_of_the_duty(self):
        # "Aşağıda yer alan hükümler ... işyerlerinin tümünü kapsar:" opens the list; the item has its own duty word.
        request = Scripted({'judgements': [judgement(1, 'UNRELATED'), judgement(2, 'UNRELATED')]})
        Adjudicator(request)(self.item, self.passages, ['SEMANTIC_PARAPHRASE'])
        sent = request.requests[0]['duty']
        self.assertTrue(sent['text'].startswith('Satış belgeleri, işyerlerinin içerisinde'), sent['text'])
        self.assertTrue(sent['lead_in'].startswith('Aşağıda yer alan hükümler'), sent['lead_in'])
        # an item without a predicate of its own is read with the lead-in that carries its duty word
        stadium, _ = duty(SALES_BYLAW, 'Yönetmelik 14646 md. 10/f.3/b.c', BREWER, 'ALC-INT-SALES')
        self.assertTrue(duty_text(stadium).startswith('Aşağıda sayılan yerlerde faaliyet gösteren işyerlerinde alkollü içki satışı'), duty_text(stadium))

    def test_a_failed_call_is_a_recorded_outcome_that_covers_nothing(self):
        record, _ = Adjudicator(Scripted(RuntimeError('model not loaded')))(self.item, self.passages, ['SEMANTIC_PARAPHRASE'])
        self.assertTrue(record['status'].startswith('FAILED: RuntimeError'))
        self.assertEqual((record['coverage'], {j['rule_relation'] for j in record['judgements']}), ('UNKNOWN', {'UNCLEAR'}))


class DecisionTests(unittest.TestCase):
    def test_disagreement_counts_only_against_a_decision_of_the_rules(self):
        self.assertEqual(disagreement('COVERS_TEXT', ['SUPPORTING_STATEMENT'], 'CONFLICT'), 3)
        self.assertEqual(disagreement('NO_EVIDENCE', ['NO_RELATED_STATEMENT'], 'CONFLICT'), 3)
        self.assertEqual(disagreement('CONFLICT', ['LIMIT_WEAKER'], 'COVERS_TEXT'), 3)
        self.assertEqual(disagreement('COVERS_TEXT', ['SUPPORTING_STATEMENT'], 'NO_EVIDENCE'), 2)
        self.assertEqual(disagreement('COVERS_TEXT', ['SUPPORTING_STATEMENT'], 'PARTIAL'), 1)
        self.assertEqual(disagreement('PARTIAL', ['PLACE_MISSING:HEALTH_FACILITY'], 'COVERS_TEXT'), 1)
        # where the rules found nothing, a model that finds the statement fills a gap and disagrees with nobody
        self.assertEqual(disagreement('NO_EVIDENCE', ['NO_RELATED_STATEMENT'], 'COVERS_TEXT'), 0)
        self.assertEqual(disagreement('PARTIAL', ['WORDING_PARTLY_MATCHED'], 'COVERS_TEXT'), 0)
        self.assertEqual(disagreement('PARTIAL', ['PLACE_MISSING:HEALTH_FACILITY'], None), 0)

    def test_a_covered_only_the_model_reads_is_a_proposal_and_no_decision(self):
        item, decision = duty(SALES_BYLAW, LICENCE, BREWER, 'ALC-INT-SALES')
        provider = Scripted({'judgements': [judgement(1, 'STATES_DUTY', 'Satış belgesi, satış noktasının içinde tüketicilerin görebileceği bir yere asılır.')]})
        row, assessment = assess_obligation(item, decision, BREWER, BREWER_REGISTER, REGISTRY, adjudicator=Adjudicator(provider), store=STORE)
        self.assertEqual((assessment.rule_coverage, assessment.model_coverage, assessment.coverage, assessment.basis, assessment.disagreement),
                         ('NO_EVIDENCE', 'COVERS_TEXT', 'NO_EVIDENCE', 'MODEL_PROPOSES_COVERED', 0))
        self.assertEqual(assessment.escalation.reasons, ['SEMANTIC_PARAPHRASE'])
        self.assertEqual((assessment.decision, assessment.review_reasons, assessment.proposal, assessment.adjudication_status),
                         ('REVIEW_REQUIRED', ['MODEL_PROPOSES_COVERED'], 'COVERS_TEXT', 'OK'))
        self.assertEqual((row.document_coverage, row.mapping.status, row.review_required), ('NO_EVIDENCE', 'NOT_COVERED', True))
        # the statement the model quoted is on the row for the person who confirms
        self.assertIn(('SOP-SALES-02#4', 'SUPPORTS', ['MODEL_READING']), [(r.passage_id, r.relation, r.reasons) for r in row.readings])

    def test_a_part_only_the_model_reads_is_a_proposal_too(self):
        item, decision = duty(SALES_BYLAW, LICENCE, BREWER, 'ALC-INT-SALES')
        provider = Scripted({'judgements': [judgement(1, 'STATES_PART', 'tüketicilerin görebileceği bir yere asılır', missing=['okunabilecek'])]})
        row, assessment = assess_obligation(item, decision, BREWER, BREWER_REGISTER, REGISTRY, adjudicator=Adjudicator(provider), store=STORE)
        self.assertEqual((assessment.coverage, assessment.basis, assessment.decision, assessment.review_reasons, assessment.proposal),
                         ('NO_EVIDENCE', 'MODEL_PROPOSES_PARTIAL', 'REVIEW_REQUIRED', ['MODEL_PROPOSES_PARTIAL'], 'PARTIAL'))
        self.assertEqual((row.document_coverage, row.mapping.status, row.review_required), ('NO_EVIDENCE', 'NOT_COVERED', True))

    def test_a_call_that_fails_leaves_the_row_open_and_is_counted(self):
        item, decision = duty(SALES_BYLAW, LICENCE, BREWER, 'ALC-INT-SALES')
        adjudicator = Adjudicator(Scripted(TimeoutError('timed out')))
        row, assessment = assess_obligation(item, decision, BREWER, BREWER_REGISTER, REGISTRY, adjudicator=adjudicator, store=STORE)
        self.assertEqual((assessment.coverage, assessment.model_coverage, assessment.decision, assessment.review_reasons),
                         ('NO_EVIDENCE', 'UNKNOWN', 'REVIEW_REQUIRED', ['ADJUDICATION_UNRESOLVED']))
        self.assertTrue(assessment.adjudication_status.startswith('FAILED: TimeoutError'))
        self.assertTrue(row.review_required)

    def test_what_a_rule_decision_rests_on_is_checked_again(self):
        from regchain.tr.adjudicate import verify
        item, decision = duty('TR:KANUN:4250', 'Kanun 4250 md. 6/f.1/c.1', BREWER, 'ALC-INT-ACT-BEER-ADS')
        row, assessment = assess_obligation(item, decision, BREWER, BREWER_REGISTER, REGISTRY, store=STORE)
        self.assertEqual((assessment.coverage, assessment.basis, assessment.decision, assessment.review_reasons),
                         ('COVERS_TEXT', 'RULE_ONLY', 'AUTO', []))
        self.assertEqual(verify(item, decision, row, BREWER_REGISTER, STORE), [])
        # a quote that is not the statement's, a document that is not in force for the target, a clause that moved
        forged = row.model_copy(update={'readings': [row.readings[0].model_copy(update={'quote': 'Alkollü içkilerin reklamı serbesttir.'})]})
        self.assertEqual(verify(item, decision, forged, BREWER_REGISTER, STORE), ['STATEMENT_QUOTE_NOT_EXACT'])
        elsewhere = row.model_copy(update={'documents_in_force': []})
        self.assertEqual(verify(item, decision, elsewhere, BREWER_REGISTER, STORE), ['STATEMENT_NOT_IN_FORCE'])
        moved = item.model_copy(update={'frame': item.frame.model_copy(update={'start': item.frame.start + 1})})
        self.assertEqual(verify(moved, decision, row, BREWER_REGISTER, STORE), ['CLAUSE_NOT_GROUNDED'])

    def test_a_conflict_only_the_model_sees_changes_no_decision_and_waits_for_a_person(self):
        item, decision = duty(ENERGY, PLACES, BOTTLER, 'NONALC-BOTTLING')
        provider = Scripted({'judgements': [judgement(1, 'CONTRADICTS', 'enerji içeceği numune dağıtımı yapılabilir'),
                                            judgement(2, 'STATES_DUTY', 'Okul kantinlerinde ve spor tesislerinde')]})
        row, assessment = assess_obligation(item, decision, BOTTLER, BOTTLER_REGISTER, REGISTRY, adjudicator=Adjudicator(provider))
        self.assertIn('numune dağıtımı yapılabilir', provider.requests[0]['statements'][0]['text'])
        self.assertEqual((assessment.rule_coverage, assessment.model_coverage, assessment.coverage, assessment.basis),
                         ('PARTIAL', 'CONFLICT', 'PARTIAL', 'MODEL_CONFLICT_UNCONFIRMED'))
        self.assertEqual((assessment.disagreement, assessment.review_required, row.review_required), (3, True, True))
        self.assertEqual((assessment.decision, assessment.review_reasons, assessment.proposal),
                         ('REVIEW_REQUIRED', ['MODEL_CONFLICT_UNCONFIRMED'], 'CONFLICT'))
        self.assertNotEqual(row.mapping.status, 'CONTRADICTED')

    def test_a_model_that_finds_nothing_leaves_the_rules_word(self):
        item, decision = duty(SALES_BYLAW, LICENCE, BREWER, 'ALC-INT-SALES')
        provider = Scripted({'judgements': [judgement(1, 'UNRELATED')]})
        row, assessment = assess_obligation(item, decision, BREWER, BREWER_REGISTER, REGISTRY, adjudicator=Adjudicator(provider))
        self.assertEqual((assessment.coverage, assessment.basis, assessment.review_required, row.mapping.status),
                         ('NO_EVIDENCE', 'BOTH_READINGS', False, 'NOT_COVERED'))

    def test_without_an_adjudicator_an_escalated_row_keeps_the_rules_word(self):
        item, decision = duty(SALES_BYLAW, LICENCE, BREWER, 'ALC-INT-SALES')
        row, assessment = assess_obligation(item, decision, BREWER, BREWER_REGISTER, REGISTRY)
        self.assertEqual((assessment.coverage, assessment.basis, assessment.model_coverage, assessment.adjudication_key, assessment.elapsed_ms),
                         ('NO_EVIDENCE', 'RULE_ONLY', None, None, 0))
        self.assertEqual(assessment.escalation.reasons, ['SEMANTIC_PARAPHRASE'])
        self.assertEqual(row.mapping.status, 'NOT_COVERED')

    def test_the_statistics_say_how_much_reached_the_model_and_what_it_cost(self):
        obligations = extract_regulation(SALES_BYLAW, REGISTRY, STORE, articles=['6'])[1]
        provider = Scripted(*[{'judgements': [judgement(1, 'UNRELATED')]}] * 40)
        adjudicator = Adjudicator(provider)
        report, assessments, stats = assess_profile(BREWER, obligations, BREWER_REGISTER, REGISTRY, STORE, adjudicator=adjudicator)
        self.assertEqual((stats['rows'], stats['obligations']), (len(report.rows), len({o.obligation_id for o in obligations})))
        self.assertEqual(stats['escalated_rows'], sum(a.escalation is not None for a in assessments))
        self.assertGreater(stats['escalated_rows'], 0)
        self.assertLess(stats['escalated_rows'], stats['rows'])                       # the model is not asked about every row
        self.assertEqual(stats['adjudications'], adjudicator.calls)
        self.assertLessEqual(stats['adjudications'], stats['escalated_rows'])         # rows of one duty share an answer
        self.assertEqual(stats['model_ms'], adjudicator.elapsed_ms)
        self.assertEqual(sum(stats['by_basis'].values()), stats['rows'])
        self.assertEqual((stats['calls_failed'], stats['escalated_unresolved_rows']), (0, 0))
        self.assertEqual(stats['decision_review_rows'], sum(a.decision == 'REVIEW_REQUIRED' for a in assessments))
        rule_only = assess_profile(BREWER, obligations, BREWER_REGISTER, REGISTRY, STORE)[2]
        self.assertEqual((rule_only['adjudications'], rule_only['model_ms'], rule_only['escalated_rows']), (0, 0, stats['escalated_rows']))


class ClauseTests(unittest.TestCase):
    def frame(self, regulation_id, ref):
        return next(f for f in extracted(regulation_id)[0] if f.ref == ref)

    def duties(self, regulation_id, ref):
        return [o for o in extracted(regulation_id)[1] if o.provision_ref == ref]

    def test_what_makes_a_clause_hard_to_read_is_named(self):
        cases = {('TR:KANUN:4250', 'Kanun 4250 md. 6/f.1/c.1'): ['NESTED_EXCEPTION'],          # "Ancak, münhasıran ..." the rules could not place
                 ('TR:KANUN:4250', 'Kanun 4250 md. 6/f.2'): ['MULTIPLE_ACTORS'],               # üretenler, ithal edenler ve pazarlayanlar
                 ('TR:KANUN:4250', 'Kanun 4250 md. 6/f.1/c.7'): ['UNCLEAR_ADDRESSEE'],         # "... görüntülere yer verilemez": by whom?
                 (SALES_BYLAW, 'Yönetmelik 14646 md. 10/f.6/c.1'): ['CROSS_REFERENCE'],
                 (SALES_BYLAW, LICENCE): []}
        for (regulation_id, ref), expected in cases.items():
            with self.subTest(ref=ref):
                self.assertEqual(clause_escalation(self.frame(regulation_id, ref), self.duties(regulation_id, ref)), expected)
        plain = (self.frame(SALES_BYLAW, LICENCE), self.duties(SALES_BYLAW, LICENCE))
        self.assertEqual(clause_escalation(*plain, support='CONTESTED'), ['READER_DISAGREEMENT'])
        self.assertEqual(clause_escalation(*plain, support='MODEL_ONLY'), ['READER_DISAGREEMENT'])
        self.assertEqual(clause_escalation(*plain, support='CONFIRMED'), [])

    def test_the_strong_model_marks_a_clause_for_review_and_never_changes_its_kind(self):
        self.assertEqual(clause_verdict('OBLIGATION', None), {'kind': 'OBLIGATION', 'model': None, 'agree': None, 'review': False})
        self.assertEqual(clause_verdict('OBLIGATION', {'kind': 'OBLIGATION'})['review'], False)
        self.assertEqual(clause_verdict('OBLIGATION', {'kind': 'NONE'}), {'kind': 'OBLIGATION', 'model': 'NONE', 'agree': False, 'review': True})
        self.assertEqual(clause_verdict('NONE', {'kind': 'PROHIBITION'}), {'kind': 'NONE', 'model': 'PROHIBITION', 'agree': False, 'review': True})
        self.assertEqual(clause_verdict('PROHIBITION', {'kind': 'BOTH'})['review'], False)     # one of the two it names
        self.assertEqual(clause_verdict('PROHIBITION', {'kind': None})['review'], True)        # an ungrounded answer is no agreement

    def test_a_marker_that_is_not_in_the_clause_is_no_reading(self):
        frame = self.frame('TR:KANUN:4250', 'Kanun 4250 md. 6/f.2')
        self.assertTrue(clause_text(frame).endswith('alkollü içki dağıtamazlar.'))
        answers = ({'kind': 'PROHIBITION', 'marker': 'alkollü içki dağıtamazlar', 'addressee': 'üretenler', 'exceptions': [], 'reason': 'x'},)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'clauses.jsonl'
            live = ClauseAdjudicator(Scripted(*answers), path)
            record = live(frame, ['MULTIPLE_ACTORS'])
            self.assertIs(live(frame, ['MULTIPLE_ACTORS']), record)
            header, recorded = load_clause_adjudications(path)
        self.assertEqual((record['kind'], record['grounded'], live.calls), ('PROHIBITION', True, 1))
        self.assertEqual((header['format'], list(recorded)), ('cardaman-tr-clause-adjudications/1', [record['key']]))
        self.assertEqual(ClauseAdjudicator(recorded=recorded)(frame, ['MULTIPLE_ACTORS'])['kind'], 'PROHIBITION')
        invented = ClauseAdjudicator(Scripted({'kind': 'DUTY', 'marker': 'bildirmek zorundadır', 'addressee': '', 'exceptions': [], 'reason': 'x'}))
        self.assertEqual([invented(frame, [])[k] for k in ('kind', 'grounded')], [None, False])
        self.assertIsNone(ClauseAdjudicator()(frame, []))                                       # neither a record nor a model


if __name__ == '__main__':
    unittest.main()
