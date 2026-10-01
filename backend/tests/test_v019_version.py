"""v0.19 Phase 17: one version everywhere, a launcher that cannot silently run old settings.

The exe of 22 September 2026 was two days older than its source: it lacked the v0.18 keys and ran
the v0.17 pipeline without saying so, and nothing showed a version (the UI still said "Pilot v0.16",
pyproject said 0.15.0). These tests pin the version string and its copies, /api/state and the page,
the server's own product defaults, the 503 for a model failure on an on-demand question or draft,
and the launcher source and build script (the exe itself is built and checked by hand:
`Cardaman.exe --version`).
"""
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import tomllib
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from regchain import __version__, display_version
from regchain.extraction.providers import ProviderFailure, failure
from regchain.pilot import workspace as workspace_module
from regchain.pilot.workspace import PRODUCT_DEFAULTS, Workspace, create_app, product_defaults
from test_pilot import FixtureProvider
from test_workspace import request_input, source_fixture

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
LAUNCHER = ROOT/'scripts'/'launcher'/'CardamanLauncher.cs'
BUILD = ROOT/'scripts'/'Build-Launcher.ps1'
RUN_WORKSPACE = ROOT/'scripts'/'Run-Workspace.ps1'
NEW_KEYS = ('COVERAGE_PIPELINE', 'JUDGE_CTX_MODE', 'JUDGE_NUM_CTX_SMALL')


def launcher_source():
    return LAUNCHER.read_text(encoding='utf-8')


def launcher_keys():
    block = re.search(r'EnvKeys\s*=\s*\{(.*?)\};', launcher_source(), re.S).group(1)
    return re.findall(r'"([A-Z_]+)"', block)


def launcher_link_pattern():
    """The regex the launcher reads the session link with, taken from its source (.NET and Python agree on it)."""
    return re.search(r'Regex\.Match\(e\.Data, @"(\^Cardaman:[^"]+)"\)', launcher_source()).group(1)


class VersionStringTests(unittest.TestCase):
    def test_the_version_and_how_people_read_it(self):
        self.assertEqual(__version__, '0.19.0.dev0')
        self.assertEqual(display_version(), 'v0.19.0-dev')
        self.assertEqual(display_version('0.19.0'), 'v0.19.0')
        self.assertEqual(display_version('0.20.1.dev3'), 'v0.20.1-dev')

    def test_pyproject_carries_the_same_version(self):
        project = tomllib.loads((BACKEND/'pyproject.toml').read_text(encoding='utf-8'))['project']
        self.assertEqual(project['version'], __version__)


class Failing(FixtureProvider):
    """The local judge whose every call fails as Ollama does when it is down or times out."""
    def __init__(self, exc):
        super().__init__()
        self.exc = exc

    def _chat(self, prompt, payload, schema):
        raise self.exc


class WorkspaceVersionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Workspace(Path(self.temp.name)/'workspace', provider_factory=lambda _: FixtureProvider(),
                                   source_fetcher=source_fixture, judge_factory=lambda provider: provider)
        self.client = TestClient(create_app(self.workspace, 'test-only-token'), base_url='http://127.0.0.1:8767', client=('127.0.0.1', 51000),
                                 headers={'Authorization': 'Bearer test-only-token'})
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def finish(self, **changes):
        run_id = self.client.post('/api/runs', json=request_input(**changes)).json()['id']
        for _ in range(400):
            row = self.workspace.metadata(run_id)
            if row['state'] in ('COMPLETED', 'FAILED') and not self.workspace.active:
                return row
            time.sleep(.01)
        self.fail('Local fixture job did not finish')

    def state(self, **env):
        with patch.dict(os.environ, env):
            response = self.client.get('/api/state')
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_state_names_the_versions_and_the_pipeline_in_effect(self):
        state = self.state(CARDAMAN_LAUNCHER_VERSION='', RELEVANCE_SCREEN='on', APPLICABILITY_CLEAR_MATCH='rule')
        self.assertEqual((state['version'], state['version_display']), (__version__, display_version()))
        # No launcher version (python -m, Run-Workspace.ps1, an exe older than v0.19) is not a mismatch.
        self.assertEqual((state['launcher_version'], state['version_mismatch']), ('', False))
        self.assertEqual(state['pipeline_settings']['relevance_screen'], 'on')
        self.assertEqual(state['pipeline_settings']['applicability_clear_match'], 'rule')
        self.assertIn('rerank_model', state)                                  # the v0.16 keys stay
        same = self.state(CARDAMAN_LAUNCHER_VERSION=__version__)
        self.assertEqual((same['launcher_version'], same['version_mismatch']), (__version__, False))
        old = self.state(CARDAMAN_LAUNCHER_VERSION='0.18.0')
        self.assertEqual((old['launcher_version'], old['version_mismatch']), ('0.18.0', True))
        # An invalid switch is reported, it does not take the whole page down.
        broken = self.state(RELEVANCE_SCREEN='maybe')
        self.assertIn('RELEVANCE_SCREEN', broken['pipeline_settings']['error'])
        v19 = self.state(COVERAGE_PIPELINE='v19')['pipeline_settings']
        if 'coverage_pipeline' in v19:                                        # recorded once the engine has the v0.19 switch
            self.assertEqual(v19['coverage_pipeline'], 'v19')

    def test_the_page_shows_the_version_and_a_mismatch_banner(self):
        page = self.client.get('/').text
        for needle in ('id="appVersion"', 'id="versionBanner"', 'function renderVersion', 'state.version_mismatch',
                       'state.version_display', 'Build-Launcher.ps1', 'renderVersion();renderList()'):
            self.assertIn(needle, page)
        self.assertIn('Pilot v0.16', page)                                    # pinned by test_phase16

    def test_a_model_failure_on_a_question_or_draft_is_a_503_that_says_so(self):
        row = self.finish(provider='ollama')
        self.assertEqual(row['state'], 'COMPLETED', row['error'])
        run = '/api/runs/'+row['id']
        obligation_id = self.client.get(run+'/packet').json()['events'][0]['payload']['obligations'][0]['id']
        timeout = ProviderFailure('Ollama request timed out after 900 s')
        timeout.code, timeout.kind = 'TIMEOUT', 'transport'
        self.workspace.provider_factory = lambda _: Failing(timeout)
        for path, body in (('/ask', {'question': 'Kayıtlar ne kadar saklanır?'}), ('/draft', {'obligation_id': obligation_id})):
            response = self.client.post(run+path, json=body)
            self.assertEqual(response.status_code, 503, response.text)
            self.assertEqual(response.json()['failure_code'], 'TIMEOUT')
            self.assertIn('Yerel AI modeli yanıt veremedi (TIMEOUT', response.json()['detail'])
            self.assertIn('Kayıtlı analizler etkilenmedi', response.json()['detail'])
        # A failure without a code (a provider older than v0.19) still names what it knows.
        self.workspace.provider_factory = lambda _: Failing(failure('Provider output incomplete or truncated', 'output_truncated'))
        truncated = self.client.post(run+'/draft', json={'obligation_id': obligation_id})
        self.assertEqual(truncated.status_code, 503, truncated.text)
        self.assertTrue(truncated.json()['failure_code'])
        # Input errors keep their own status, and the analysis is untouched.
        self.assertEqual(self.client.post(run+'/ask', json={'question': 'ab'}).status_code, 422)
        self.assertEqual(self.client.get(run+'/verify').status_code, 200)
        self.assertFalse((self.workspace.result(row['id'])/'questions.jsonl').exists())


class ServerMainTests(unittest.TestCase):
    """workspace.main() with the server itself replaced: arguments, defaults, output, marker."""

    def run_main(self, env):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        seen, out = {}, io.StringIO()

        def fake_app(workspace, token, port):
            seen['workspace'] = workspace
            return object()

        def fake_run(app, **_):
            seen['marker'] = json.loads((seen['workspace'].root/'server.json').read_text(encoding='utf-8'))
            seen['env'] = {key: os.environ.get(key) for key in PRODUCT_DEFAULTS}

        before = {key: os.environ.get(key) for key in PRODUCT_DEFAULTS}
        with patch.dict(os.environ, env), redirect_stdout(out), \
                patch('sys.argv', ['workspace', '--root', str(Path(temp.name)/'ws'), '--port', '8799']), \
                patch.object(workspace_module, 'create_app', fake_app), patch('uvicorn.run', fake_run):
            for key in PRODUCT_DEFAULTS:
                if key not in env:
                    os.environ.pop(key, None)
            workspace_module.main()
        seen['workspace'].close()
        # The defaults belong to the server process; the rest of the suite keeps the code defaults.
        self.assertEqual({key: os.environ.get(key) for key in PRODUCT_DEFAULTS}, before)
        self.assertFalse((seen['workspace'].root/'server.json').exists())
        seen['lines'] = out.getvalue().splitlines()
        return seen

    def test_main_sets_the_product_pipeline_and_prints_the_version_before_the_link(self):
        seen = self.run_main({})
        self.assertEqual(seen['env'], {'APPLICABILITY_CLEAR_MATCH': 'rule', 'RELEVANCE_SCREEN': 'on', 'COVERAGE_PIPELINE': 'v19'})
        lines, link = seen['lines'], re.compile(launcher_link_pattern())
        self.assertEqual(lines[0], f'Cardaman version: {__version__} ({display_version()})')
        self.assertIsNone(link.match(lines[0]))                               # an old exe never takes it for the link
        linked = [i for i, line in enumerate(lines) if link.match(line)]
        self.assertEqual(len(linked), 1)
        self.assertGreater(linked[0], 0)
        self.assertEqual(seen['marker']['version'], __version__)
        self.assertNotIn('token', json.dumps(seen['marker']))

    def test_explicit_values_are_never_overridden(self):
        explicit = {'APPLICABILITY_CLEAR_MATCH': 'model', 'RELEVANCE_SCREEN': 'off', 'COVERAGE_PIPELINE': 'v18'}
        self.assertEqual(self.run_main(explicit)['env'], explicit)
        partly = self.run_main({'COVERAGE_PIPELINE': 'v18'})['env']
        self.assertEqual(partly, {'APPLICABILITY_CLEAR_MATCH': 'rule', 'RELEVANCE_SCREEN': 'on', 'COVERAGE_PIPELINE': 'v18'})

    def test_an_empty_value_counts_as_unset(self):
        environ = {'RELEVANCE_SCREEN': '', 'APPLICABILITY_CLEAR_MATCH': '  ', 'COVERAGE_PIPELINE': 'v18'}
        self.assertEqual(product_defaults(environ), ['APPLICABILITY_CLEAR_MATCH', 'RELEVANCE_SCREEN'])
        self.assertEqual(environ, {'RELEVANCE_SCREEN': 'on', 'APPLICABILITY_CLEAR_MATCH': 'rule', 'COVERAGE_PIPELINE': 'v18'})


class LauncherSourceTests(unittest.TestCase):
    def test_the_allowlist_has_the_v019_keys_in_every_place_that_reads_env(self):
        keys = launcher_keys()
        for key in NEW_KEYS:
            self.assertIn(key, keys)
        # Same allowlist as scripts/Run-Workspace.ps1 (the launcher says so), and every key is documented.
        ps_keys = re.search(r"\$line -match '\^\(([A-Z_|]+)\)=", RUN_WORKSPACE.read_text(encoding='utf-8')).group(1).split('|')
        self.assertEqual(sorted(keys), sorted(ps_keys))
        example = (ROOT/'.env.example').read_text(encoding='utf-8')
        for key in keys:
            self.assertRegex(example, rf'(?m)^{key}=', key)
        self.assertIn('Default(env, "COVERAGE_PIPELINE", "v19");', launcher_source())
        self.assertIn("$env:COVERAGE_PIPELINE = 'v19'", RUN_WORKSPACE.read_text(encoding='utf-8'))

    def test_the_exe_checks_the_backend_version_before_it_touches_anything(self):
        source = launcher_source()
        for needle in ('BuildInfo.Version', 'BuildInfo.SourceSha256', 'BuildInfo.BuiltAt', 'regchain\\\\__init__.py',
                       '"--allow-version-mismatch"', 'scripts\\\\Build-Launcher.ps1', 'info.EnvironmentVariables["CARDAMAN_LAUNCHER_VERSION"] = BuildInfo.Version',
                       '@"^Cardaman version:\\s+(\\S+)"'):
            self.assertIn(needle, source)
        check = source.index('if (backend != BuildInfo.Version)')
        self.assertLess(check, source.index('if (!ReleaseWorkspace(workspace))'))    # nothing killed on a refusal
        self.assertLess(check, source.index('ConfigureModels(env);'))              # nor Ollama started
        self.assertIn('if (!allowMismatch)', source[check:check + 1200])
        # The session link pattern is unchanged, so v0.19 servers still start from older launchers too.
        self.assertEqual(launcher_link_pattern(), r'^Cardaman:\s+(http://127\.0\.0\.1:\d+/#token=\S+)')

    def test_version_prints_and_exits_before_the_single_instance_signal(self):
        source = launcher_source()
        self.assertIn('arg == "--version"', source)
        show = source.index('if (showVersion) return PrintVersion(root, source);')
        self.assertLess(show, source.index('new EventWaitHandle('))
        body = source[source.index('static int PrintVersion('):source.index('static string BackendVersion(')]
        for needle in ('BuildInfo.Version', 'BuildInfo.SourceSha256', 'BuildInfo.BuiltAt', 'BackendVersion(source)', 'MATCH', 'MISMATCH', 'return 0;'):
            self.assertIn(needle, body)
        for forbidden in ('Process.Start', 'StartOllama', 'InstalledModels', 'Fail('):
            self.assertNotIn(forbidden, body)

    def test_the_source_stays_within_csharp_5(self):
        # csc 4.8 is the C# 5 compiler that ships with Windows: no interpolation, no ?. , no nameof.
        source = launcher_source()
        self.assertIsNone(re.search(r'[\s(=,+]\$"', source))
        self.assertIsNone(re.search(r'\w\?\.\w', source))
        self.assertNotIn('nameof(', source)


class BuildScriptTests(unittest.TestCase):
    def test_the_build_script_generates_and_compiles_build_info(self):
        script = BUILD.read_text(encoding='utf-8')
        for needle in ('BuildInfo.cs', 'public const string Version', 'public const string SourceSha256',
                       'public const string BuiltAt', 'backend\\src\\regchain\\__init__.py', 'Get-FileHash -Algorithm SHA256',
                       '$launcherSource $BuildInfoPath'):
            self.assertIn(needle, script)

    @unittest.skipUnless(os.name == 'nt' and shutil.which('powershell'), 'Windows PowerShell builds the launcher')
    def test_build_info_carries_the_backend_version_and_the_source_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'BuildInfo.cs'
            done = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(BUILD),
                                   '-InfoOnly', '-BuildInfoPath', str(target)], capture_output=True, text=True, timeout=120)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            info = target.read_text(encoding='utf-8')
        values = dict(re.findall(r'public const string (\w+) = "([^"]*)";', info))
        self.assertEqual(values['Version'], __version__)
        self.assertEqual(values['SourceSha256'], hashlib.sha256(LAUNCHER.read_bytes()).hexdigest())
        self.assertRegex(values['BuiltAt'], r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$')

    @unittest.skipUnless((LAUNCHER.parent/'BuildInfo.cs').exists(), 'no launcher built in this tree')
    def test_the_last_build_was_for_this_backend_version(self):
        info = (LAUNCHER.parent/'BuildInfo.cs').read_text(encoding='utf-8')
        self.assertIn(f'public const string Version = "{__version__}";', info,
                      'Cardaman.exe was built for another version: run scripts\\Build-Launcher.ps1')


if __name__ == '__main__':
    unittest.main()
