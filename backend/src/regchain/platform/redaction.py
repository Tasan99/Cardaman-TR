"""What an AI call record may still say once it leaves the operator's machine.

The provider never records a prompt, payload or answer (``OllamaProvider.call_log`` holds
hashes, token counts and timings), so redaction here is an allowlist over that record, not a
search for text. Three levels, by audience:

* ``minimal``  the local operator debugging a run: every known field.
* ``standard`` a tenant export or a reviewer: drops the tuning internals (thinking flag,
  generation budget, request bytes, schema hash, done reason). ``request_bytes`` and the
  schema hash say nothing a reviewer decides on, and the byte count is a length side channel
  on the document.
* ``strict``   a cross-tenant operations view (SRE metrics): also drops the links to the
  analysis (provision id, obligation id, evidence ids) and the payload hash, which lets a
  holder of a document confirm it was processed.

``error`` survives every level, truncated to 120 characters and passed through the secret
redactor: the provider already reduces errors to status meanings and exception names.

v0.19 fields: the failure code, the runtime's timings and throughput and the processor split
(``gpu_fraction``, ``processor``) are what an operations view measures, so they survive every
level like ``elapsed_ms``; the context window and what fit_call did with it (``window_action``, the
estimate before a payload was compressed or trimmed) are tuning internals, and a compression or a
fallback to the wide window says the document was long, so they stay at the operator's level.
"""
import json
import re

from .secrets import redact_secrets

COMMON = ('at', 'task', 'stage', 'provider', 'model', 'model_version', 'prompt_sha256', 'cache_hit',
          'retry_count', 'status', 'elapsed_ms', 'prompt_tokens', 'output_tokens', 'error',
          # v0.19: numbers and codes only.
          'failure_code', 'load_duration_ms', 'eval_duration_ms', 'prompt_eval_duration_ms', 'tokens_per_second',
          'prompt_tokens_per_second', 'gpu_fraction', 'processor')
LINKS = ('provision_id', 'obligation_id', 'evidence_ids', 'payload_sha256')
# v0.18 adds the admission estimate and the reserved generation budget: numbers, like num_predict,
# and a length side channel like request_bytes, so they stay at the operator's level.
TUNING = ('thinking', 'num_predict', 'num_predict_reduced', 'request_bytes', 'done_reason', 'schema_sha256',
          'estimated_prompt_tokens', 'reserved_output',
          # v0.19: the window and what fit_call did with it.
          'num_ctx', 'window_action', 'window_estimate_before',
          # v0.19 round 5: a count and a code of what fit_call did (the compressed evidence, why the wide window).
          'compressed_evidence_count', '16k_fallback_reason',
          # v0.19 t7: the replay identity of a call (hashes of the request sent and of the answer, the parse outcome) and the
          # sampling settings sent. A hash over the payload or the answer confirms a document to its holder, like
          # payload_sha256, so they stay at the operator's level; the answer text itself is never a record field.
          'request_hash', 'temperature', 'seed', 'raw_output_sha256', 'parse_ok', 'parsed_sha256', 'parse_failure')

ALLOWED_CALL_FIELDS = {
    'minimal': frozenset(COMMON + LINKS + TUNING),
    'standard': frozenset(COMMON + LINKS),
    'strict': frozenset(COMMON),
}
# Names a text-bearing field could have; the allowlists must never contain one.
FORBIDDEN_CALL_FIELDS = frozenset({'prompt', 'payload', 'response', 'answer', 'content', 'messages',
                                   'text', 'source_paragraph', 'context', 'raw'})
assert all(fields.isdisjoint(FORBIDDEN_CALL_FIELDS) for fields in ALLOWED_CALL_FIELDS.values())
ERROR_LIMIT = 120
PASSAGE_CHARS = 200
PASSAGE_WORDS = 20
# Structural delimiters of a log line; the text between two of them is one segment.
SEGMENT = re.compile(r'([\r\n\t"“”«»\[\]{}=])')


class RedactionPolicy:
    def __init__(self, level: str = 'standard'):
        if level not in ALLOWED_CALL_FIELDS:
            raise ValueError("level must be 'minimal', 'standard' or 'strict'")
        self.level = level
        self.allowed = ALLOWED_CALL_FIELDS[level]

    def __repr__(self):
        return f'RedactionPolicy({self.level!r})'


def redact_call(entry: dict, policy: RedactionPolicy) -> dict:
    """The allowed fields of one ai-calls.jsonl entry, in the entry's own order; ``error`` is
    truncated and secret-masked, lists are copied so the caller's record is not shared."""
    if not isinstance(entry, dict):
        raise ValueError('an AI call entry is a dict')
    kept = {}
    for key, value in entry.items():
        if key not in policy.allowed or key in FORBIDDEN_CALL_FIELDS:
            continue
        if key == 'error' and value is not None:
            value = redact_secrets(str(value))[:ERROR_LIMIT]
        elif isinstance(value, list):
            value = list(value)
        kept[key] = value
    return kept


def redact_call_log(text: str, policy: RedactionPolicy) -> str:
    """An ai-calls.jsonl body at ``policy``'s level: one redacted entry per line, blank lines
    dropped; the bytes the single-run export writes (``Workspace.redacted_calls_text``). The
    tenant export writes its copy through this, so what leaves the machine does not depend on
    which export was used (v0.17 copied the raw log into the tenant zip)."""
    return ''.join(json.dumps(redact_call(json.loads(line), policy), ensure_ascii=False) + '\n'
                   for line in text.splitlines() if line.strip())


def _looks_like_passage(segment: str) -> bool:
    text = segment.strip()
    if len(text) <= PASSAGE_CHARS:
        return False
    words = text.split()
    # Prose (many short words) or one unbroken blob (a base64 or URL-encoded document); a
    # comma-separated list of a few 64-character hashes is neither and stays readable.
    return len(words) >= PASSAGE_WORDS or len(words) == 1


def scrub_text(text) -> str:
    """Replace any segment of a free-text log line that looks like a policy passage (more than
    200 characters of prose between structural delimiters) with ``[redacted N chars]``.

    A best-effort last line of defence for lines written by hand (``logging`` calls, error
    messages): the primary control is that no retained record holds document text at all.
    """
    text = text if isinstance(text, str) else str(text)
    pieces = SEGMENT.split(text)
    for index in range(0, len(pieces), 2):
        if _looks_like_passage(pieces[index]):
            pieces[index] = f'[redacted {len(pieces[index])} chars]'
    return ''.join(pieces)
