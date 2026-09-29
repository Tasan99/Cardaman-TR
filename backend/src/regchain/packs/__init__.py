"""Sector packs: the domain knowledge the shared engine reads through regchain.sector.

A pack here is a Python module of raw tables (regchain.packs.<pack>.knowledge) registered with
regchain.sector, alongside its data root under the pack registry (regchain.tr.packs). Nothing in
regchain.pilot, regchain.extraction, regchain.ingestion or regchain.evaluation imports this package.
"""
