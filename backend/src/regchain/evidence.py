"""Versioned UTF-8 canonical evidence format, not a claim to implement RFC 8785.

Floats are forbidden: persist confidence as decimal strings. Keys and strings
remain byte-exact Unicode. Every verifier must implement this same format.
"""
import hashlib
import json
import re
from typing import Any

FORMAT = "regchain-evidence-v1"
GENESIS = "0" * 64

def _validate(value: Any) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if isinstance(value, list):
        for item in value:
            _validate(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _validate(item)
        return
    raise ValueError("Evidence supports only null, strings, booleans, integers, lists and string-keyed objects")

def canonical_bytes(value: Any) -> bytes:
    _validate(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()

def make_event(payload: dict, previous_hash: str = GENESIS) -> dict:
    if not re.fullmatch(r"[0-9a-f]{64}", previous_hash):
        raise ValueError("Invalid previous hash")
    # Snapshot prevents later caller mutation from changing a constructed event.
    body = json.loads(canonical_bytes({"format": FORMAT, "previous_event_hash": previous_hash, "payload": payload}))
    return {**body, "event_hash": digest(body)}

def verify_chain(events: list[dict], expected_head: str, expected_count: int) -> bool:
    """A trusted head AND count are required to detect tail deletion."""
    previous = GENESIS
    try:
        if len(events) != expected_count:
            return False
        for event in events:
            if set(event) != {"format", "previous_event_hash", "payload", "event_hash"}:
                return False
            if event["format"] != FORMAT or event["previous_event_hash"] != previous:
                return False
            body = {key: value for key, value in event.items() if key != "event_hash"}
            if event["event_hash"] != digest(body):
                return False
            previous = event["event_hash"]
        return previous == expected_head
    except (KeyError, ValueError, TypeError, UnicodeError):
        return False
