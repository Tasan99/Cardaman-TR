"""Windows extended-length paths for retained evidence.

Evidence files are named by their SHA-256. A run directory only a few folders deep
therefore exceeds MAX_PATH (260), and publication fails after the model has already
run. The ``\\\\?\\`` namespace lifts that limit for absolute paths without needing the
machine-wide LongPathsEnabled policy. ``is_relative_to`` is only meaningful between
paths that BOTH went through ``extended``: a prefixed path is never relative to a
plain one, so containment checks must not mix the two forms.
"""
import os
from pathlib import Path

PREFIX = '\\\\?\\'


def extended(path) -> Path:
    resolved = Path(path).resolve()
    text = str(resolved)
    if os.name != 'nt' or text.startswith(PREFIX):
        return resolved
    if text.startswith('\\\\'):
        return Path(PREFIX + 'UNC\\' + text[2:])
    return Path(PREFIX + text)
