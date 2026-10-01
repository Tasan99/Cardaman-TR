"""An immutable description of what produced an evaluation run, so two runs can be compared."""
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from regchain import __version__
from regchain.evidence import digest
from regchain.extraction.contract import CONTRACT_VERSION
from regchain.extraction.retrieval import RETRIEVER_VERSION
from regchain.extraction.service import PROMPT_HASH as EXTRACTION_PROMPT_HASH
from regchain.ingestion.mevzuat import MEVZUAT_PARSER_VERSION
from regchain.ingestion.models import PARSER_VERSION
from regchain.pilot.engine import PROMPT_HASH as PILOT_PROMPT_HASH, VERSION as PILOT_VERSION, pipeline_settings, prompt_registry
from regchain.pilot.policies import PARSER as POLICY_READER
from regchain.pilot.rerank import VERSION as RERANK_VERSION
from regchain.pilot.semantic import VERSION as RETRIEVAL_VERSION

# Only these settings are recorded; nothing else from the environment is read.
ENV_KEYS = ('LLM_MODEL', 'LLM_MODEL_DIGEST', 'LLM_THINKING', 'LLM_TIMEOUT_SECONDS', 'EMBED_MODEL', 'EMBED_MODEL_DIGEST', 'JUDGE_MODEL',
            'JUDGE_MODEL_DIGEST', 'JUDGE_THINKING', 'JUDGE_TIMEOUT_SECONDS', 'JUDGE_NUM_PREDICT', 'JUDGE_NUM_CTX', 'RERANK_MODEL', 'RERANK_DEVICE',
            'EVIDENCE_MIN_SIMILARITY', 'AI_CACHE_DIR', 'OLLAMA_BASE_URL', 'APPLICABILITY_CLEAR_MATCH', 'RELEVANCE_SCREEN',
            'COVERAGE_PIPELINE', 'JUDGE_CTX_MODE', 'JUDGE_NUM_CTX_SMALL',
            # v0.19 context addendum: the adaptive fallback window and the fast model (scratchpad run legs export all three).
            'JUDGE_NUM_CTX_LARGE', 'FAST_MODEL', 'FAST_MODEL_DIGEST')
# v0.19 label provenance keys (schema.py). Left unset they are dropped before the dataset digest, so a dataset
# written before v0.19 keeps its digest (tr-aml-v1: 4ab81127...) and the gate's frozen baselines stay comparable.
PROVENANCE_KEYS = ('label_source', 'label_sources', 'default_label_source')
# Outside manifest_sha256: when the manifest was written and where the dataset file happened to be.
UNHASHED = ('created_at', 'dataset_path')


def without_provenance_defaults(value):
    """The dataset dump with every unset (None or empty) provenance key removed, at every level."""
    if isinstance(value, dict):
        return {k: without_provenance_defaults(v) for k, v in value.items() if not (k in PROVENANCE_KEYS and v in (None, {}, []))}
    if isinstance(value, list):
        return [without_provenance_defaults(v) for v in value]
    return value


def dataset_digest(dataset) -> str:
    """The canonical digest of a dataset's content (manifest['dataset']['sha256'])."""
    return digest(without_provenance_defaults(json.loads(dataset.model_dump_json())))


def source_tree_hash(root: Path) -> str:
    """sha256 over the package's own source files, in path order: the substitute for a commit hash."""
    hasher = hashlib.sha256()
    for path in sorted(root.rglob('*')):
        if path.suffix in ('.py', '.html', '.json') and '__pycache__' not in path.parts and path.is_file():
            hasher.update(str(path.relative_to(root)).replace('\\', '/').encode('utf-8'))
            hasher.update(path.read_bytes())
    return hasher.hexdigest()


def git_commit(root: Path) -> str:
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(root), capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else 'not-a-git-repository'
    except (OSError, subprocess.SubprocessError):
        return 'not-a-git-repository'


def snapshot_fingerprint(sections) -> str:
    """sha256 over the sorted (printed_label, content_hash) pairs of a parsed regulation snapshot.

    Two runs that read the same snapshot with the same parser agree on it; a parser change that
    moves a label or a paragraph boundary, or a re-ingested snapshot, does not.
    """
    return digest(sorted([s['printed_label'], s['content_hash']] for s in sections))


def run_prompts() -> tuple:
    """(prompt registry, coverage prompts sha256 or None) of the pipeline the environment selects.

    The engine's registry without settings is the v18 one; a v19 run asks judge.fast and judge.verify, and
    its packets name them and the sha256 of the v0.19 coverage prompts. The manifest names the same, so two
    v19 runs before and after a prompt change do not show identical versions. A setting the engine refuses
    (the run then fails in analyze anyway) records the plain registry.
    """
    try:
        settings = pipeline_settings()
    except ValueError:
        return prompt_registry(), None
    return prompt_registry(settings), settings.get('coverage_prompts_sha256')


def build_manifest(dataset, mode: dict, package_root: Path | None = None, snapshots: dict | None = None, dataset_path=None) -> dict:
    root = package_root or Path(__file__).resolve().parents[1]
    prompts, coverage_prompts = run_prompts()
    manifest = {'format': 'cardaman-evaluation-manifest-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
                'cardaman_version': __version__, 'git_commit': git_commit(root.parents[1]), 'source_tree_sha256': source_tree_hash(root),
                'dataset': {'id': dataset.dataset_id, 'version': dataset.version, 'cases': len(dataset.cases),
                            'sha256': dataset_digest(dataset)},
                'mode': mode,
                'models': {'extraction': os.getenv('LLM_MODEL', ''), 'extraction_digest': os.getenv('LLM_MODEL_DIGEST', ''),
                           'judge': os.getenv('JUDGE_MODEL', '') or os.getenv('LLM_MODEL', ''), 'judge_digest': os.getenv('JUDGE_MODEL_DIGEST', ''),
                           'judge_thinking': os.getenv('JUDGE_THINKING', ''), 'embedding': os.getenv('EMBED_MODEL', ''),
                           'embedding_digest': os.getenv('EMBED_MODEL_DIGEST', ''), 'reranker': os.getenv('RERANK_MODEL', '') or 'off'},
                'versions': {'pilot': PILOT_VERSION, 'pilot_prompt_hash': PILOT_PROMPT_HASH, 'extraction_prompt_hash': EXTRACTION_PROMPT_HASH,
                             'prompts': prompts, 'contract': CONTRACT_VERSION, 'retriever': RETRIEVER_VERSION,
                             'policy_retrieval': RETRIEVAL_VERSION, 'reranker': RERANK_VERSION, 'fca_parser': PARSER_VERSION,
                             'mevzuat_parser': MEVZUAT_PARSER_VERSION, 'policy_reader': POLICY_READER,
                             # v19 only, so a v18 manifest keeps its v0.18 shape and hash.
                             **({'coverage_prompts_sha256': coverage_prompts} if coverage_prompts else {})},
                'environment': {key: os.getenv(key, '') for key in ENV_KEYS},
                'platform': {'python': platform.python_version(), 'system': platform.system(), 'machine': platform.machine()}}
    if snapshots is not None:
        # v0.18: {regulation_fixture: snapshot_fingerprint}; the gate's parser snapshot compatibility check.
        manifest['snapshots'] = snapshots
    if dataset_path is not None:
        manifest['dataset_path'] = str(dataset_path)
    manifest['manifest_sha256'] = digest({k: v for k, v in manifest.items() if k not in UNHASHED})
    return manifest


def comparable(a: dict, b: dict) -> tuple[bool, list[str]]:
    """Whether two manifests describe the same configuration, and what differs when they do not."""
    differences = []
    for key in ('source_tree_sha256', 'dataset', 'mode', 'models', 'versions'):
        if a.get(key) != b.get(key):
            differences.append(key)
    return not differences, differences
