"""Store a board decision's text for the decision layer (regchain.tr.decisions).

    python scripts/tr_decision_import.py <pdf or txt> --id TR:KURUL_KARARI:<ID> --no <decision no> --title "<title>" \
        --url <document url on the authority's host> [--date YYYY-MM-DD] [--review-date YYYY-MM-DD] [--basis "<legal basis>"] \
        [--institution REKABET] [--sector-tags BEVERAGE ...] [--activity-tags ...] [--root <decision store root>] [--synthetic]

Reads the text (pypdf for a PDF), stores text.txt + decision.json under the root (default: evaluation/external/decisions,
not committed) and prints the numbered items the reader will see, so the item parse can be checked by eye before the
decision is assessed with `python -m regchain.tr decisions`. The record is what is typed here, checked by hash from
then on; it is not verified against a catalogue of the authority.
"""
import argparse
import logging
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from regchain.tr.decisions import RECORD_FORMAT, DecisionRecord, DecisionStore, commitment_sections, text_hash   # noqa: E402

ROOT = Path(__file__).resolve().parents[2] / 'evaluation' / 'external' / 'decisions'


def read_text(path: Path) -> str:
    if path.suffix.lower() != '.pdf':
        return path.read_text(encoding='utf-8')
    from pypdf import PdfReader
    logging.getLogger('pypdf').setLevel(logging.ERROR)
    pages = [page.extract_text() or '' for page in PdfReader(str(path)).pages]
    text = '\n'.join(pages)
    text = re.sub(r'-\n(?=[a-zçğıöşü])', '', text)              # hyphenated line breaks
    text = re.sub(r'[ \t]+', ' ', text)
    return text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('source', help='the decision text: a PDF or a UTF-8 text file')
    parser.add_argument('--id', required=True, help='TR:KURUL_KARARI:<ID>')
    parser.add_argument('--no', required=True, help='the decision number as printed (e.g. 26-20/614-243)')
    parser.add_argument('--title', required=True)
    parser.add_argument('--url', required=True, help='the document URL on the authority host (allowlisted in adapters.py)')
    parser.add_argument('--date', help='decision date, ISO')
    parser.add_argument('--review-date', help='the date the commitments are reviewed, ISO')
    parser.add_argument('--basis', default='', help='the legal basis as the decision states it')
    parser.add_argument('--institution', default='REKABET')
    parser.add_argument('--sector-tags', nargs='+', default=['BEVERAGE'])
    parser.add_argument('--activity-tags', nargs='*', default=[])
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--synthetic', action='store_true', help='a test decision, never real law')
    parser.add_argument('--disclaimer', default='')
    args = parser.parse_args(argv)
    text = read_text(Path(args.source))
    record = DecisionRecord(format=RECORD_FORMAT, decision_id=args.id, institution_id=args.institution, title=args.title,
                            decision_no=args.no, decision_date=date.fromisoformat(args.date) if args.date else None,
                            document_url=args.url, legal_basis=args.basis,
                            review_date=date.fromisoformat(args.review_date) if args.review_date else None,
                            sector_tags=args.sector_tags, activity_tags=args.activity_tags, content_hash=text_hash(text),
                            retrieved_at=datetime.now(timezone.utc), synthetic=args.synthetic, disclaimer=args.disclaimer)
    directory = DecisionStore(Path(args.root)).write(record, text)
    sections = commitment_sections(text, record.label, record.title)
    print(f'stored {record.decision_id} ({len(text)} characters, hash {record.content_hash[:12]}) in {directory}')
    print(f'{len(sections)} items:')
    for section in sections:
        print(f"  {section['printed_label']}: {section['text'][:110].replace(chr(10), ' ')}{'...' if len(section['text']) > 110 else ''}")
    if len(sections) <= 1:
        print('NOTE: one item only - the numbering of the text was not recognised; the whole text is read as paragraphs')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
