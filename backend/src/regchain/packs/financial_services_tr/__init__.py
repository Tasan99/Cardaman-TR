"""FINANCIAL_SERVICES_TR: the Turkish AML / financial-services sector knowledge the pilot engine was built on."""
from pathlib import Path

from .knowledge import KNOWLEDGE, PACK_ID

DATA_ROOT = Path(__file__).resolve().parent / 'data'

__all__ = ['DATA_ROOT', 'KNOWLEDGE', 'PACK_ID']
