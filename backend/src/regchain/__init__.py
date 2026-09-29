__version__ = "0.19.0.dev0"

# The default sector knowledge (regchain.compat) is registered before any engine module reads it.
from . import compat as _compat  # noqa: E402,F401


def display_version(version: str = __version__) -> str:
    """The version as people read it: '0.19.0.dev0' -> 'v0.19.0-dev', '0.19.0' -> 'v0.19.0'.

    __version__ stays the PEP 440 string because backend/pyproject.toml and Cardaman.exe (its
    BuildInfo.Version, written by scripts/Build-Launcher.ps1) compare it byte for byte."""
    release, dev, _ = version.partition('.dev')
    return 'v' + release + ('-dev' if dev else '')
