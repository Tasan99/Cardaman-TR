"""Curated official sources. Nothing here is crawled or discovered automatically.

A policy statement and its amendment instrument say how the law changed. The operative
text lives in the consolidated Handbook, so the consolidated chapters cited by the
starter publication are listed explicitly and ingested as their own documents.
"""
import re

HANDBOOK_ORIGIN = "https://www.handbook.fca.org.uk"
CHAPTER = re.compile(r"^([A-Z]{2,6})\s+([0-9][0-9A-Z]*)\.([0-9][0-9A-Z]*)$")

PUBLICATIONS = (
    ("FCA-PS24-2-PAGE", "https://www.fca.org.uk/publications/policy-statements/ps24-2-strengthening-protections-borrowers-financial-difficulty"),
    ("FCA-PS24-2-PDF", "https://www.fca.org.uk/publication/policy/ps24-2.pdf"),
)

# Consolidated chapters cited by PS24/2 and by the chapters it amends. This list is the
# ingestion worklist; `python -m regchain.extraction.coverage` reports what is still
# missing after a run rather than leaving references silently unresolved.
CONSOLIDATED_CHAPTERS = (
    "CONC 2.10", "CONC 4.6", "CONC 5D.1", "CONC 5D.2", "CONC 5D.3", "CONC 5D.4",
    "CONC 6.7", "CONC 7.1", "CONC 7.2", "CONC 7.3", "CONC 7.4", "CONC 7.5",
    "CONC 7.6", "CONC 7.7", "CONC 7.15", "CONC 8.2",
    # MCOB 13.4 is cited by older material but the FCA now serves MCOB 13.5 for it;
    # it is left out deliberately and reported by the coverage report instead.
    "MCOB 13.3", "MCOB 13.5", "MCOB 13.8",
    "GEN 2.2", "GEN 2.3", "PRIN 2A.2", "PRIN 2A.6", "PERG 12.6",
)


def chapter_url(label: str) -> str | None:
    """Map a printed chapter locator such as 'CONC 7.3' to its Handbook URL."""
    match = CHAPTER.match(label.strip().upper())
    if not match:
        return None
    return f"{HANDBOOK_ORIGIN}/handbook/{match[1]}/{match[2]}/{match[3]}.html"


def chapter_key(label: str) -> str | None:
    match = CHAPTER.match(label.strip().upper())
    return f"FCA-HB-{match[1]}-{match[2]}-{match[3]}" if match else None


def chapter_of(locator: str) -> str | None:
    """Reduce a provision locator such as 'CONC 7.3.5D' to its chapter 'CONC 7.3'."""
    parts = locator.strip().upper().split()
    if len(parts) != 2:
        return None
    numbers = parts[1].split(".")
    return f"{parts[0]} {numbers[0]}.{numbers[1]}" if len(numbers) >= 2 else None


def consolidated_sources() -> tuple[tuple[str, str], ...]:
    return tuple((chapter_key(label), chapter_url(label)) for label in CONSOLIDATED_CHAPTERS)


def glossary_source(glossary_id: str) -> tuple[str, str]:
    identifier = glossary_id.strip()
    if not re.fullmatch(r"G[0-9A-Za-z]{1,64}", identifier):
        raise ValueError("A glossary identifier looks like G252; it is read from the source, not guessed")
    return f"FCA-HB-GLOSSARY-{identifier}", f"{HANDBOOK_ORIGIN}/glossary/{identifier}"
