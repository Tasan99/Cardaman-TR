"""Four reading fixes found on the UNKNOWN review load of 3 October 2026 (evaluation/reports/verification-20261003 §5):
an authority's own task is not a company's duty; a sentence whose subject is the consumer binds no company; a list's
closing line that holds a second sentence gives the items only its first sentence, and the second is a clause of its own;
a change to the rule files needs a new rule version label."""
import hashlib
import json
import unittest
from pathlib import Path

from regchain.tr.clauses import Clause, split_clauses
from regchain.tr.corpus import CorpusStore
from regchain.tr.extraction import extract_regulation
from regchain.tr.frames import frame_of
from regchain.tr.packs import Registry

REGISTRY = Registry.load()
STORE = CorpusStore()


def read(text: str):
    clause = Clause(ref='K/f.1', label='K', fikra=1, start=0, end=len(text), text=text, reading=text)
    frame = frame_of(clause, 'TR:KANUN:5996')
    return frame.kind, frame.modality


def obligations(regulation_id, article):
    return {o.provision_ref: o for o in extract_regulation(regulation_id, REGISTRY, STORE, articles=[article])[1]}


class AuthorityTaskTests(unittest.TestCase):
    def test_an_authority_that_is_the_subject_of_a_duty_does_its_own_task(self):
        self.assertEqual(read('a) Bakanlık, ihbarı mecburî bir hastalığın varlığı veya şüphesi ya da yeni bir salgın durumunda, inceleme yapmak, '
                              'teşhis etmek, gerekli kontrol ve koruma tedbirlerini almakla yükümlüdür.')[0], 'DELEGATION')
        self.assertEqual(read('(8) İl özel idareleri ve belediyeler, hayvan hastalıkları ile mücadele ve kontrollerde Bakanlığa yardımcı '
                              'olmakla yükümlüdür.')[0], 'DELEGATION')

    def test_a_company_duty_that_names_an_authority_stays_a_duty(self):
        # the authority is the recipient, the agent of a participle, or whose conditions are to be met
        for text in ('(6) Bir yerde bulaşıcı hayvan hastalığından haberdar olan ilgililer, durumu Bakanlığa ihbar etmekle yükümlüdür.',
                     'Kurum tarafından adına dağıtım yetki belgesi düzenlenen firmalar, satış raporunu Kuruma vermekle yükümlüdür.',
                     'Bu hususlarla ilgili Bakanlıkça belirlenen şartlara uyulması zorunludur.',
                     'Üretici ve ithalatçı firmalar aylık satış raporunu ayı takip eden ayın en geç 20 nci günü Kuruma intikal ettirmek zorundadır.'):
            self.assertEqual(read(text)[0], 'OBLIGATION', text)

    def test_on_the_corpus(self):
        found = obligations('TR:KANUN:5996', '4')
        self.assertNotIn('Kanun 5996 md. 4/f.1/b.a', found)
        self.assertNotIn('Kanun 5996 md. 4/f.8', found)
        self.assertIn('Kanun 5996 md. 4/f.6', found)                       # "... haberdar olan ilgililer ... ihbar etmekle yükümlüdür"


class ConsumerSubjectTests(unittest.TestCase):
    def test_a_sentence_whose_subject_is_the_consumer_binds_no_company(self):
        kind, _ = read('Tüketici, bildirim tarihinden itibaren en geç altmış gün içinde borcun tamamını ödediği ve kredi kullanmaya son '
                       'verdiği takdirde faiz artışından etkilenmez.')
        self.assertNotIn(kind, ('OBLIGATION', 'PROHIBITION'))

    def test_a_consumer_informed_by_the_seller_is_the_seller_s_duty(self):
        self.assertEqual(read('(2) Tüketici, mesafeli sözleşmeyi ya da buna karşılık gelen herhangi bir teklifi kabul etmeden önce '
                              'ayrıntıları yönetmelikte belirlenen hususlarda satıcı veya sağlayıcı tarafından bilgilendirilir.')[0], 'OBLIGATION')
        # a consumer-credit contract is no consumer subject
        self.assertNotEqual(read('(3) Tüketici kredisi sözleşmesi yazılı olarak kurulur.')[0], 'OTHER')

    def test_on_the_corpus(self):
        self.assertNotIn('Kanun 6502 md. 26/f.2/c.4', obligations('TR:KANUN:6502', '26'))
        self.assertIn('Kanun 6502 md. 48/f.2/c.1', obligations('TR:KANUN:6502', '48'))


class ListClosingTests(unittest.TestCase):
    def test_a_closing_line_with_two_sentences_gives_the_items_only_the_first(self):
        section = STORE.section('TR:KANUN:6502', 'Kanun 6502 md. 11')
        clauses = {c.ref: c for c in split_clauses(section)}
        for letter in ('a', 'b', 'c', 'ç'):
            item = clauses[f'Kanun 6502 md. 11/f.1/b.{letter}']
            self.assertEqual(item.closing, 'seçimlik haklarından birini kullanabilir.')
        second = clauses['Kanun 6502 md. 11/f.1/c.2']
        self.assertEqual(second.text, 'Satıcı, tüketicinin tercih ettiği bu talebi yerine getirmekle yükümlüdür.')
        self.assertEqual(section['text'][second.start:second.end], second.text)       # an exact span of the stored text

    def test_the_items_are_rights_and_the_seller_is_bound_by_the_second_sentence(self):
        found = obligations('TR:KANUN:6502', '11')
        for letter in ('a', 'b', 'c', 'ç'):
            self.assertNotIn(f'Kanun 6502 md. 11/f.1/b.{letter}', found)
        seller = found['Kanun 6502 md. 11/f.1/c.2']
        self.assertEqual((seller.basis, seller.scope.scope_status), ('ACTOR_EXPLICIT', 'DEFINED'))
        self.assertIn('SELLER', [a.id for a in seller.frame.actors])


class RuleVersionTests(unittest.TestCase):
    """Each rule version label names one content of the files that carry its rules (data/rule_versions.json). A change to
    those files with the same label is a change nobody can see in the outputs: the label must change with it."""

    def test_every_label_names_the_content_of_its_files(self):
        from regchain.tr import rule_versions
        recorded = json.loads(rule_versions.RECORD.read_text(encoding='utf-8'))
        for name, (label, files) in rule_versions.current().items():
            digest = rule_versions.fingerprint(files)
            self.assertIn(label, recorded['labels'], f'{name}: label {label} is not recorded (scripts/tr_rule_versions.py record)')
            self.assertEqual(recorded['labels'][label]['sha256'], digest,
                             f'{name}: the files of {label} changed; give the rules a new label and record it')

    def test_the_labels_of_the_rules_changed_since_1_october_are_new(self):
        from regchain.tr import extraction, frames
        self.assertNotEqual(frames.FRAME_RULES_VERSION, 'tr-frames-v1')
        self.assertNotEqual(extraction.EXTRACTION_RULES_VERSION, 'tr-scope-rules-v1')


if __name__ == '__main__':
    unittest.main()
