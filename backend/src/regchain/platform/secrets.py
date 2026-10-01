"""Secrets come from an allowlist of environment variables and never appear in a log line.

An allowlist rather than ``os.getenv(anything)``: a code path that asks for a name outside the
list is a bug worth failing on, and the list doubles as the inventory of what a deployment must
provide. ``redact_secrets`` masks the configured values (and the password inside a URL-shaped
value) so a stack trace or a debug line that interpolated one does not ship it.
"""
import os
from urllib.parse import urlsplit

ALLOWED = ('LLM_API_KEY', 'DATABASE_URL', 'APP_DB_PASSWORD', 'POSTGRES_PASSWORD', 'EVM_RPC_URL')
MASK = '[redacted {name}]'


class SecretUnavailable(LookupError):
    """The name is not allowlisted, or the variable is unset or empty."""


def get_secret(name: str, environ=None) -> str:
    environ = os.environ if environ is None else environ
    if name not in ALLOWED:
        raise SecretUnavailable(f'{name} is not an allowlisted secret')
    value = environ.get(name, '')
    if not value or not value.strip():
        raise SecretUnavailable(f'{name} is not set')
    return value


def configured_secrets(environ=None) -> dict:
    """name -> value for every allowlisted secret that is set; used by the redactor."""
    environ = os.environ if environ is None else environ
    return {name: environ[name] for name in ALLOWED if environ.get(name, '').strip()}


def _pieces(name: str, value: str):
    """The value itself, and the password embedded in a URL-shaped value (postgres://u:pw@host),
    which log lines tend to carry on its own."""
    yield value
    try:
        password = urlsplit(value).password
    except ValueError:
        password = None
    if password:
        yield password


def redact_secrets(text, environ=None) -> str:
    """Mask every configured secret value inside ``text``. Longer values go first so a whole
    URL is masked before the password inside it, leaving no partial value behind."""
    text = text if isinstance(text, str) else str(text)
    replacements = []
    for name, value in configured_secrets(environ).items():
        for piece in _pieces(name, value):
            replacements.append((piece, MASK.format(name=name)))
    for piece, mask in sorted(replacements, key=lambda item: -len(item[0])):
        if piece in text:
            text = text.replace(piece, mask)
    return text
