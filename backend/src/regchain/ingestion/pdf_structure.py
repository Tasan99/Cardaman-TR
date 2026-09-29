"""Printed paragraph boundaries and page provenance, never redline consolidation."""
import re
from collections import Counter

from .models import Paragraph

LABEL = re.compile(r'^(\d+[A-Z]?(?:\.\d+[A-Z]*){1,4})\s+(?:(R|G|D|E)\s+)?(?=\S)')
RUNNING = re.compile(r'^(?:\d{1,4}|Page\s+\d+\s+of\s+\d+|FCA\s+\d{4}/\d+)$',re.I)
AMENDMENT = re.compile(r'underlin(?:ing|ed).*?(?:new|insert)|strik(?:ing|e|en).*?(?:delet|through)',re.I)
SECTION_HEADING = re.compile(r'^(?:Chapter\s+\d+|Annex\s+[A-Z0-9]+|Appendix\s+\d+|Part\s+\d+)\b',re.I)


def segment_pages(pages: list[list[str]], *, commentary: bool=False) -> tuple[Paragraph,...]:
    """Keep numeric subclauses/lists with their parent; continue blocks across pages.

    All non-running text is preserved in source_spans. A printed number is only a
    locator; it is not classified as an operative consolidated rule.
    """
    blocks = []
    current: list[tuple[int,str]] = []
    current_label = None
    current_kind = 'COMMENTARY' if commentary else 'UNKNOWN'
    active_kind = current_kind
    heading = 'preamble'
    block_heading = heading

    def flush():
        nonlocal current,current_label
        if current:
            blocks.append((current_label,block_heading,current_kind,list(current)))
            current = []
            current_label = None

    for page,lines in enumerate(pages,1):
        previous_blank = False
        for index,raw in enumerate(lines):
            line = re.sub(r'\s+',' ',raw).strip()
            if not line:
                previous_blank = True
                continue
            if RUNNING.fullmatch(line):
                continue
            if AMENDMENT.search(line):
                flush()
                active_kind = 'AMENDMENT'
            match = LABEL.match(line)
            section_heading = SECTION_HEADING.match(line)
            if match or section_heading:
                flush()
                if section_heading:
                    heading = line
                current_label = match[1] if match else None
                block_heading = heading
                current_kind = active_kind
            elif not current:
                block_heading = heading
                current_kind = active_kind
            elif current_label is None and previous_blank:
                # Unnumbered front matter: actual blank-line boundaries, no arbitrary
                # sentence splitting that could detach a condition or list item.
                flush()
                block_heading = heading
                current_kind = active_kind
            current.append((page,line))
            previous_blank = False
        # Unnumbered matter should not absorb the next page's cover/front matter.
        if current_label is None:
            flush()
    flush()
    counts = Counter()
    result = []
    for label,heading,kind,lines in blocks:
        first,last = lines[0][0],lines[-1][0]
        section = heading if label else f'page-{first}'
        key = label or 'block'
        counts[(section,key)] += 1
        number = key if counts[(section,key)] == 1 else f'{key}~{counts[(section,key)]}'
        spans = []
        for page,line in lines:
            if spans and spans[-1]['page'] == page:
                spans[-1]['text'] += '\n' + line
            else:
                spans.append({'page':page,'text':line})
        flags = ['AMENDMENT_NOT_CONSOLIDATED'] if kind == 'AMENDMENT' else []
        if first != last:
            flags.append('CROSS_PAGE_PARAGRAPH')
        if label and len(lines)>1 and re.match(r'^[A-Z]\s+\S',lines[1][1]):
            flags.append('POSSIBLE_WRAPPED_LABEL')
        if label and re.search(r'\.\.\.|…', ' '.join(line for _,line in lines)):
            flags.append('OMITTED_TEXT_MARKER')
        result.append(Paragraph(section,number,' '.join(line for _,line in lines),(heading,),first,
            'pdf_printed' if label else 'pdf_extracted',last,label,kind,tuple(flags),tuple(spans)))
    return tuple(result)
