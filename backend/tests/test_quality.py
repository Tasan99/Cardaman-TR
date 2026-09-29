import json
import unittest
from hashlib import sha256

from regchain.ingestion.pdf_structure import segment_pages
from regchain.extraction.retrieval import retrieve
from regchain.extraction.pipeline import extract
from regchain.extraction.providers import RulesProvider


def section(sid,text,label=None,version='v1',**values):
    return dict(id=sid,version_id=version,text=text,printed_label=label,section_number='CONC 7',
        paragraph_number=label or sid,heading_path=['CONC 7'],content_hash=sha256(text.encode()).hexdigest(),**values)


class ContextProvider:
    name='fixture'
    model_version='fixture'
    def __init__(self,supported=True,evidence=True):
        self.supported=supported
        self.evidence=evidence
    def generate_with_context(self,text,context):
        data=json.loads(RulesProvider().generate(text))
        if self.evidence:
            for candidate in data['obligations']:
                candidate['exceptions']=[context.items[0]['text']]
                candidate['supporting_evidence']=[dict(section_id=context.items[0]['section_id'],role='exception',quote=context.items[0]['text'])]
        return json.dumps(data)
    def review(self,text,output,context):
        return self.supported


class QualityTests(unittest.TestCase):
    def test_numbered_paragraph_keeps_cross_page_exception_and_subclauses(self):
        paragraphs=segment_pages([['Chapter 7','7.1 A firm must retain records','(1) of consent'],['2','unless exempt.','7.2 A firm must review controls.']])
        p=next(p for p in paragraphs if p.printed_label=='7.1')
        self.assertEqual(p.text,'7.1 A firm must retain records (1) of consent unless exempt.')
        self.assertEqual((p.page,p.page_end),(1,2))
        self.assertEqual([s['page'] for s in p.source_spans],[1,2])

    def test_table_of_contents_does_not_mark_entire_document_amendment(self):
        paragraphs=segment_pages([['Appendix 1','Made rules (legal instrument)'],['Chapter 1','1.1 Firms should review records.'],
            ['Annex A','In this Annex, underlining indicates new text and striking through indicates deleted text.','7.1 A firm must act.']],commentary=True)
        self.assertEqual(next(p for p in paragraphs if p.printed_label=='1.1').source_kind,'COMMENTARY')
        self.assertEqual(next(p for p in paragraphs if p.printed_label=='7.1').source_kind,'AMENDMENT')

    def test_redline_is_never_extracted_as_operative_law(self):
        target=section('a','A firm must retain records.',source_kind='AMENDMENT')
        self.assertEqual(extract(target['text'],RulesProvider(),retrieve(target,[target])).reason,'AMENDMENT_REQUIRES_CONSOLIDATED_SOURCE')

    def test_printed_labels_do_not_discard_original_text(self):
        lines=['7.1 A firm must act.','(1) unless exempt.','7.2 A firm should review.']
        paragraphs=segment_pages([lines])
        self.assertEqual(' '.join(p.text for p in paragraphs),' '.join(lines))

    def test_omitted_text_and_wrapped_labels_require_review(self):
        for flag,reason in [('OMITTED_TEXT_MARKER','SOURCE_CONTAINS_OMITTED_TEXT'),('POSSIBLE_WRAPPED_LABEL','AMBIGUOUS_PRINTED_LABEL')]:
            target=section('a','A firm must act.',quality_flags=[flag])
            self.assertEqual(extract(target['text'],RulesProvider(),retrieve(target,[target])).reason,reason)

    def test_reference_uses_only_same_version(self):
        target=section('a','A firm must act under CONC 7.2.','7.1')
        other=section('b','Unless exempt.','7.2',version='v2')
        packet=retrieve(target,[target,other])
        self.assertEqual(packet.items,())
        self.assertTrue(packet.unresolved)

    def test_ambiguous_reference_is_not_guessed(self):
        target=section('a','A firm must act under CONC 7.2R.','7.1')
        packet=retrieve(target,[target,section('b','First.','7.2'),section('c','Second.','7.2R')])
        self.assertTrue(packet.unresolved)

    def test_suffix_is_not_removed_from_source_label(self):
        target=section('a','A firm must act under CONC 7.2.','7.1')
        self.assertTrue(retrieve(target,[target,section('b','Not the same label.','7.2G')]).unresolved)

    def test_reference_cycle_is_bounded(self):
        target=section('a','A firm must act under CONC 7.2.','7.1')
        other=section('b','See CONC 7.1.','7.2')
        packet=retrieve(target,[target,other])
        self.assertFalse(packet.unresolved)
        self.assertEqual(len(packet.items),1)

    def test_retrieved_neighbor_missing_reference_is_recorded_for_review(self):
        # A reference owed by retrieved context is reported rather than silently
        # dropped, but only the paragraph's own references gate its extraction.
        target=section('a','A firm must act.','7.1')
        packet=retrieve(target,[target,section('b','Unless exempt under CONC 7.9.','7.2')])
        self.assertFalse(packet.unresolved)
        self.assertIn('CONC 7.9',packet.dependency_gaps)
        self.assertIn('CONC 7.9',packet.manifest()['dependency_gaps'])

    def test_one_exact_reference_cannot_hide_an_unidentified_reference(self):
        target=section('a','A firm must act under CONC 7.2 and in accordance with external standards.','7.1')
        packet=retrieve(target,[target,section('b','Keep records.','7.2')])
        self.assertTrue(packet.unresolved)

    def test_budget_exhaustion_is_explicit_and_abstains(self):
        target=section('a','A firm must act under CONC 7.2.','7.1')
        packet=retrieve(target,[target,section('b','x'*100,'7.2')],max_chars=40)
        self.assertTrue(packet.truncated)
        self.assertEqual(extract(target['text'],RulesProvider(),packet).output.obligations,[])

    def test_rule_provider_does_not_claim_to_resolve_context(self):
        target=section('a','A firm must act under CONC 7.2.','7.1')
        packet=retrieve(target,[target,section('b','Unless the account is exempt.','7.2')])
        self.assertFalse(packet.unresolved)
        self.assertEqual(extract(target['text'],RulesProvider(),packet).reason,'CONTEXT_REQUIRES_LLM')

    def test_context_exception_requires_exact_attributed_evidence(self):
        target=section('a','A firm must retain records.','7.1')
        packet=retrieve(target,[target,section('b','Unless the account is exempt.','7.2')])
        good=extract(target['text'],ContextProvider(),packet)
        self.assertEqual(good.output.status,'EXTRACTED')
        self.assertEqual(good.output.obligations[0].supporting_evidence[0].section_id,'b')
        self.assertEqual(extract(target['text'],ContextProvider(evidence=False),packet).reason,'GROUNDING_REJECTED')

    def test_second_pass_can_reject_a_verbatim_but_uncertain_candidate(self):
        target=section('a','A firm must retain records.','7.1')
        packet=retrieve(target,[target,section('b','Unless exempt.','7.2')])
        result=extract(target['text'],ContextProvider(supported=False),packet)
        self.assertEqual(result.reason,'SECOND_PASS_UNCERTAIN')
        self.assertEqual(result.output.obligations,[])

    def test_fabricated_context_id_is_rejected(self):
        class Forged(ContextProvider):
            def generate_with_context(self,text,context):
                data=json.loads(super().generate_with_context(text,context))
                data['obligations'][0]['supporting_evidence'][0]['section_id']='not-retrieved'
                return json.dumps(data)
        target=section('a','A firm must retain records.','7.1')
        packet=retrieve(target,[target,section('b','Unless exempt.','7.2')])
        self.assertEqual(extract(target['text'],Forged(),packet).reason,'GROUNDING_REJECTED')

    def test_new_normative_forms_and_actor_boundaries(self):
        for text,mode,actor in [('A firm is required to retain records.','MUST','A firm'),
             ('A firm is prohibited from deleting records.','MUST_NOT','A firm'),
             ('7.1 R A firm must retain records.','MUST','A firm'),
             ('If fees change, a firm must notify customers.','MUST','a firm')]:
            output=extract(text,RulesProvider()).output
            self.assertEqual(output.status,'EXTRACTED')
            self.assertEqual((output.obligations[0].subject,output.obligations[0].modality),(actor,mode))

    def test_regulator_intentions_and_epistemic_may_are_not_duties(self):
        for text in ['We must publish the consultation.','The firm may be in difficulty.','In May 2024, new rules were published.']:
            self.assertEqual(extract(text,RulesProvider()).output.obligations,[])


if __name__=='__main__': unittest.main()
