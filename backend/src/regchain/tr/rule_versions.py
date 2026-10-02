"""Rule version labels and the content they name.

Every output carries rule version labels (Frame.rules_version, ExtractedObligation.rules_version, the comparer's,
the adjudication's, the QDMS export's). A label is useful only if it changes when the rules change. data/rule_versions.json
records, for every label, the SHA-256 of the files that carry its rules; tests/test_tr_reading_fixes.py fails when a
label's files no longer have that content, so a change to the rules needs a new label. Labels are append-only:
scripts/tr_rule_versions.py records a new label and refuses to re-record an old one with other content.
"""
import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
RECORD = PACKAGE / 'data' / 'rule_versions.json'


def current() -> dict[str, tuple[str, list[str]]]:
    """{rule family: (its current label, the files of the tr package that carry its rules)}."""
    from . import adjudicate, compare, decisions, extraction, frames, qdms, routing, semantic
    return {
        'frames': (frames.FRAME_RULES_VERSION, ['frames.py', 'clauses.py', 'data/lexicon.json']),
        'scope': (extraction.EXTRACTION_RULES_VERSION, ['extraction.py', 'data/lexicon.json']),
        'routing': (routing.ROUTING_RULES_VERSION, ['routing.py']),
        'compare': (compare.COMPARE_RULES_VERSION, ['compare.py']),
        'candidates': (semantic.CANDIDATE_RULES_VERSION, ['semantic.py']),
        'adjudication': (adjudicate.ADJUDICATION_RULES_VERSION, ['adjudicate.py']),
        'decisions': (decisions.DECISION_RULES_VERSION, ['decisions.py']),
        'qdms': (qdms.QDMS_RULES_VERSION, ['qdms.py']),
    }


def fingerprint(files: list[str], root: Path = PACKAGE) -> str:
    """SHA-256 over the files' names and contents, line endings normalised (a checkout's CRLF is no rule change)."""
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode('utf-8') + b'\0')
        digest.update((root / name).read_bytes().replace(b'\r\n', b'\n') + b'\0')
    return digest.hexdigest()


def load() -> dict:
    if RECORD.exists():
        return json.loads(RECORD.read_text(encoding='utf-8'))
    return {'format': 'cardaman-tr-rule-versions/1', 'labels': {}}


def status() -> dict[str, dict]:
    """{family: {'label', 'recorded', 'matches'}} for the current code."""
    record = load()['labels']
    out = {}
    for family, (label, files) in current().items():
        digest = fingerprint(files)
        entry = record.get(label)
        out[family] = {'label': label, 'files': files, 'sha256': digest, 'recorded': entry is not None,
                       'matches': entry is not None and entry['sha256'] == digest}
    return out
