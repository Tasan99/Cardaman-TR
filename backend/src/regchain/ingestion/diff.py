from dataclasses import dataclass
from difflib import SequenceMatcher, unified_diff
from itertools import zip_longest

from .models import Paragraph


@dataclass(frozen=True)
class Change:
    kind: str
    old_index: int | None
    new_index: int | None
    diff_text: str


def paragraph_diff(old: tuple[Paragraph, ...], new: tuple[Paragraph, ...]) -> list[Change]:
    """Align unchanged paragraphs before pairing edits, avoiding insertion cascades.

    Matching is textual/structural, not a legal equivalence judgment. Ambiguous splits
    and merges remain human-review candidates.
    """
    matcher = SequenceMatcher(a=[(p.section, p.text) for p in old],
                              b=[(p.section, p.text) for p in new], autojunk=False)
    changes = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        for i, j in zip_longest(range(i1, i2), range(j1, j2)):
            left, right = old[i] if i is not None else None, new[j] if j is not None else None
            kind = "MODIFIED" if left and right else "REMOVED" if left else "ADDED"
            diff = "\n".join(unified_diff(
                left.text.splitlines() if left else [], right.text.splitlines() if right else [],
                fromfile=f"old/{left.section}/{left.number}" if left else "old/absent",
                tofile=f"new/{right.section}/{right.number}" if right else "new/absent", lineterm=""))
            if not diff:  # heading-only move remains visible
                diff = f"Section moved: {left.section if left else ''} -> {right.section if right else ''}"
            changes.append(Change(kind, i, j, diff))
    return changes
