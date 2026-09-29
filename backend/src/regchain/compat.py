"""Compatibility layer: the default sector knowledge.

Every analysis Cardaman has run so far read the Turkish AML tables; this module keeps that the default
by registering the FINANCIAL_SERVICES_TR knowledge when the regchain package is imported. It is the
one place outside the pack that names the pack. An installation that wants another default replaces
this registration before the engine modules are imported.
"""
from . import sector
from .packs.financial_services_tr import DATA_ROOT, KNOWLEDGE

DEFAULT_PACK_ID = KNOWLEDGE.pack_id

if sector.default_pack_id() is None:
    sector.register(KNOWLEDGE, default=True, data_root=DATA_ROOT)
