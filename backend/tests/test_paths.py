import os
import shutil
import tempfile
import unittest
from pathlib import Path

from regchain.pilot.paths import PREFIX, extended
from regchain.pilot.sources import load_sources
from test_workspace import source_fixture


class ExtendedPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_idempotent_and_absolute(self):
        once = extended(self.root/'a'/'..'/'b')
        self.assertEqual(extended(once), once)
        self.assertTrue(once.is_absolute())
        self.assertEqual(once.name, 'b')

    def test_containment_holds_inside_one_namespace(self):
        root = extended(self.root)
        self.assertTrue((root/'runs'/'x').resolve().is_relative_to(root))
        self.assertFalse(extended(self.root.parent/'elsewhere').is_relative_to(root))

    @unittest.skipUnless(os.name == 'nt', 'The extended namespace exists only on Windows')
    def test_windows_prefix_and_unc_form(self):
        self.assertTrue(str(extended(self.root)).startswith(PREFIX))
        self.assertEqual(str(extended(Path('\\\\server\\share\\folder'))), PREFIX+'UNC\\server\\share\\folder')

    @unittest.skipUnless(os.name == 'nt', 'MAX_PATH only limits Windows')
    def test_hash_named_sources_survive_a_directory_beyond_max_path(self):
        deep = self.root/('d'*110)/('e'*110)
        # Registered after the TemporaryDirectory cleanup, so it runs before it: the
        # plain-path rmtree of tempfile cannot delete what it cannot address.
        self.addCleanup(shutil.rmtree, extended(self.root/('d'*110)), True)
        bundle = source_fixture(deep/'sources')
        name = bundle['sources'][0]['raw_file']
        self.assertGreater(len(str(deep/'sources'/name)), 260)
        _, sections = load_sources(deep/'sources')
        self.assertEqual([s['printed_label'] for s in sections], ['CONC 7.1.1', 'CONC 7.3.4'])


if __name__ == '__main__':
    unittest.main()
