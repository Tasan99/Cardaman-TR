"""The independent evaluation set (evaluation/independent/independent-v1.json) is not the generator's output.

tr-aml-v1 was written by evaluation/build_dataset.py: its applicability labels come from a rule
function (applicability()), its coverage labels from designed tables (COVERAGE, CONTROL_COVERAGE),
its policies and profiles from the same file. A score against those labels measures agreement with
the rules the pipeline was tuned against. The independent set is only worth something if nothing of
that leaked into it, so these tests read build_dataset.py (imported from its path: importing it builds
nothing) and fail on any shared profile, policy file, policy passage, rule note or evidence tuple, on
a label that resolves to RULE_DERIVED, and on a fixture or evidence string that is not where the
dataset says. Agreement of an independent label with the rule is not tested: a correct label may
legitimately agree with it.

The independent set is built by a separate step; until its file exists the tests that read it skip.
"""
import csv
import importlib.util
import io
import unittest
from hashlib import sha256
from pathlib import Path

from regchain.evaluation.harness import load_dataset
from regchain.pilot.policies import read_policy

REPO = Path(__file__).resolve().parents[2]
BUILDER = REPO / 'evaluation' / 'build_dataset.py'
INDEPENDENT_DIR = REPO / 'evaluation' / 'independent'
INDEPENDENT = INDEPENDENT_DIR / 'independent-v1.json'
LABEL_FIELDS = ('applicability', 'coverage', 'conflict', 'entity_gate')
MIN_PASSAGE = 60          # a generated sentence this long appearing verbatim is a copied passage, not a shared phrase


def squash(text) -> str:
    return ' '.join(str(text or '').split())


def load_builder():
    spec = importlib.util.spec_from_file_location('cardaman_build_dataset_for_isolation', BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generated_texts(builder) -> dict:
    """{basename: text} of every policy and control file build_dataset.py writes, rendered as build() writes them."""
    texts = dict(builder.POLICIES)
    for name, rows in builder.CONTROLS.items():
        buffer = io.StringIO()
        csv.writer(buffer, lineterminator='\n').writerows(rows)
        texts[name] = buffer.getvalue()
    return texts


class BuilderTests(unittest.TestCase):
    def test_build_dataset_never_writes_or_mentions_the_independent_set(self):
        self.assertNotIn('independent', BUILDER.read_text(encoding='utf-8').lower())
        builder = load_builder()
        for constant in ('DATASET', 'LABELS', 'POL_DIR', 'REG_DIR'):
            path = Path(getattr(builder, constant)).resolve()
            with self.subTest(constant):
                self.assertFalse(path.is_relative_to(INDEPENDENT_DIR.resolve()), f'{constant} points into evaluation/independent')


@unittest.skipUnless(INDEPENDENT.is_file(), 'evaluation/independent/independent-v1.json is not built yet (a separate step builds it)')
class IsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = load_builder()
        cls.dataset = load_dataset(INDEPENDENT)
        cls.generated = generated_texts(cls.builder)
        cls.policies = {}                          # (case_id, relative path) -> resolved path
        for case in cls.dataset.cases:
            for relative in [*case.policy_documents, *case.optional_control_records]:
                cls.policies[(case.case_id, relative)] = (INDEPENDENT_DIR / relative).resolve()

    def expectations(self):
        return [(case, expected) for case in self.dataset.cases for expected in case.expected_obligations]

    def test_it_is_not_tr_aml_v1(self):
        self.assertNotEqual(self.dataset.dataset_id, 'tr-aml-v1')

    def test_no_company_profile_of_the_generator(self):
        ids = {profile['id'] for profile in self.builder.COMPANIES.values()}
        names = {squash(profile['name']).casefold() for profile in self.builder.COMPANIES.values()}
        for case in self.dataset.cases:
            with self.subTest(case.case_id):
                self.assertNotIn(case.company_profile.get('id'), ids)
                self.assertNotIn(squash(case.company_profile.get('name')).casefold(), names)

    def test_every_fixture_exists(self):
        for case in self.dataset.cases:
            with self.subTest(case.case_id):
                snapshot = (INDEPENDENT_DIR / case.regulation_fixture).resolve()
                self.assertTrue((snapshot / 'snapshot.json').is_file(), f'{case.regulation_fixture} is not a retained snapshot')
        for (case_id, relative), path in self.policies.items():
            with self.subTest(case_id, policy=relative):
                self.assertTrue(path.is_file(), f'{relative} does not exist')

    def test_no_policy_file_name_or_text_of_the_generator(self):
        generated_sha = {sha256(text.encode('utf-8')).hexdigest() for text in self.generated.values()}
        generated_squashed = {squash(text) for text in self.generated.values()}
        header = squash(self.builder.HEADER)
        passages = {squash(part) for text in self.generated.values() for part in text.split('\n\n')}
        passages |= {squash(cell) for rows in self.builder.CONTROLS.values() for row in rows for cell in row}
        passages = {p for p in passages if len(p) >= MIN_PASSAGE and p != header}
        for case in self.dataset.cases:
            names = [Path(p).name for p in [*case.policy_documents, *case.optional_control_records]]
            with self.subTest(case.case_id, check='unique basenames'):
                self.assertEqual(len(names), len(set(names)), 'the harness copies policies by basename; two would overwrite each other')
        for (case_id, relative), path in self.policies.items():
            with self.subTest(case_id, policy=relative):
                self.assertNotIn(path.name, self.generated)
                self.assertNotIn(sha256(path.read_bytes()).hexdigest(), generated_sha)
                text = squash(' '.join(chunk['text'] for chunk in read_policy(path)['chunks']))
                self.assertNotIn(text, generated_squashed)
                copied = [p for p in passages if p in text]
                self.assertEqual(copied, [], 'passages copied from a generated tr-aml-v1 policy')

    def test_no_note_of_the_rule_function_or_the_duty_table(self):
        builder = self.builder
        notes = {spec.get('notes', '') for specs in builder.DUTIES.values() for spec in specs}
        notes |= {builder.applicability(profile, article, spec['clause'])[2]
                  for profile in builder.COMPANIES for article, specs in builder.DUTIES.items() for spec in specs}
        notes = {squash(note) for note in notes if squash(note)}
        for case, expected in self.expectations():
            with self.subTest(case.case_id, article=expected.article, clause=expected.clause):
                self.assertEqual([n for n in notes if n in squash(expected.notes)], [])

    def test_no_evidence_tuple_of_the_coverage_tables(self):
        tables = [self.builder.COVERAGE, self.builder.CONTROL_COVERAGE]
        tuples = {(coverage, conflict, tuple(evidence)) for table in tables for per_policy in table.values()
                  for coverage, conflict, evidence in per_policy.values() if evidence}
        for case, expected in self.expectations():
            if expected.evidence:
                with self.subTest(case.case_id, article=expected.article, clause=expected.clause):
                    self.assertNotIn((expected.coverage, expected.conflict, tuple(expected.evidence)), tuples)

    def test_every_label_resolves_to_a_source_that_is_not_rule_derived(self):
        for case, expected in self.expectations():
            overall = self.dataset.label_source_of(expected)
            with self.subTest(case.case_id, article=expected.article, clause=expected.clause):
                self.assertNotEqual(overall, 'RULE_DERIVED')
                for field in LABEL_FIELDS:
                    self.assertNotEqual(expected.label_sources.get(field) or overall, 'RULE_DERIVED', field)

    def test_every_evidence_string_is_in_a_passage_of_its_policy_files(self):
        """As the harness scores retrieval (harness.evidence_passages): a whitespace-squashed substring of one passage."""
        chunks = {key: [squash(chunk['text']) for chunk in read_policy(path)['chunks']] for key, path in self.policies.items() if path.is_file()}
        for case, expected in self.expectations():
            passages = [text for (case_id, _), texts in chunks.items() if case_id == case.case_id for text in texts]
            for evidence in expected.evidence:
                with self.subTest(case.case_id, article=expected.article, clause=expected.clause, evidence=evidence[:40]):
                    self.assertTrue(any(squash(evidence) in passage for passage in passages))


if __name__ == '__main__':
    unittest.main()
