"""The record of one measurement: code version, input files and their hashes, source versions, settings, similarity table.

    PYTHONPATH=<tree>/backend/src python backend/scripts/tr_run_manifest.py <out.json> [label]

Inputs are the ones every TR measurement script and `python -m regchain.tr assess|qdms` read with the recorded run
replayed: the packaged corpus (head versions), the pilot profiles, the synthetic policy registers, the lexicon and the
catalogue, the recorded similarity table and adjudications of 20261001. No model is called.
"""
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import regchain
from regchain.tr import adjudicate, compare, engines, extraction, frames, semantic

try:
    from regchain.tr import qdms
except ImportError:                                   # a commit before the QDMS export
    qdms = None
from regchain.tr.corpus import CorpusStore
from regchain.tr.packs import Registry
from regchain.tr.semantic import SimilarityTable

src = Path(regchain.__file__).resolve().parents[1]
data = src / 'regchain' / 'tr' / 'data'
rec = data / 'evaluation' / 'recorded' / '20261001'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args) -> str:
    return subprocess.run(['git', '-C', str(src), *args], capture_output=True, text=True).stdout.strip()


files = [rec / 'similarities.json', *sorted(rec.glob('adjudicate-*.jsonl')), rec / 'FROZEN.json', data / 'lexicon.json',
         data / 'regulations.json', data / 'scopes.json', data / 'vocabulary.json', data / 'ownership.json',
         *sorted((data / 'pilot_profiles').glob('*.json')), *sorted((data / 'pilot_policies').rglob('*.*')),
         *sorted((data / 'packs').glob('*.json'))]
registry, store = Registry.load(), CorpusStore()
table = SimilarityTable.load(rec / 'similarities.json')
try:                                                   # every label of the code, and whether its files match its record
    from regchain.tr import rule_versions
    rule_status = {f: {'label': s['label'], 'matches_record': s['matches']} for f, s in rule_versions.status().items()}
except ImportError:                                    # a commit before data/rule_versions.json
    rule_status = None
manifest = {
    'label': sys.argv[2] if len(sys.argv) > 2 else '', 'recorded_at': time.strftime('%Y-%m-%d %H:%M:%S %z'),
    'code': {'commit': git('rev-parse', 'HEAD'), 'short': git('rev-parse', '--short', 'HEAD'), 'dirty': bool(git('status', '--porcelain', '--', '.')),
             'src': str(src), 'python': platform.python_version()},
    'model': 'none: recorded adjudications replayed (qwen3:8b records of run 8); no embedder call (recorded table)',
    'similarity_table': {'file': str(rec / 'similarities.json'), 'sha256': sha(rec / 'similarities.json'), 'model_version': table.model_version,
                         'duties': len(table.pairs)},
    'settings': {'LEXICAL_FLOOR': semantic.LEXICAL_FLOOR, 'SEMANTIC_FLOOR': semantic.SEMANTIC_FLOOR, 'ANCHOR_FLOOR': semantic.ANCHOR_FLOOR,
                 'ANCHOR_STEMS': semantic.ANCHOR_STEMS, 'TOP_K': semantic.TOP_K, 'SEMANTIC_RELEVANCE': compare.SEMANTIC_RELEVANCE},
    'rule_versions': {'frames': frames.FRAME_RULES_VERSION, 'extraction': extraction.EXTRACTION_RULES_VERSION,
                      'compare': compare.COMPARE_RULES_VERSION, 'candidates': semantic.CANDIDATE_RULES_VERSION,
                      'adjudication': adjudicate.ADJUDICATION_RULES_VERSION, 'engines': engines.ENGINES_VERSION,
                      'qdms': getattr(qdms, 'QDMS_RULES_VERSION', None)},
    'rule_version_record': rule_status,
    'inputs': {str(p.relative_to(src)): sha(p) for p in files if p.is_file()},
    'source_versions': {rid: {'version_id': v.version_id, 'parsed_hash': v.parsed_hash} if (v := store.head(rid)) else None
                        for rid in sorted(registry.regulations)},
}
Path(sys.argv[1]).write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
print('manifest', manifest['code']['short'], 'dirty' if manifest['code']['dirty'] else 'clean', len(manifest['inputs']), 'inputs',
      sum(1 for v in manifest['source_versions'].values() if v), 'stored sources')
