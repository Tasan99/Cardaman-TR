"""The official-source corpus: allowlisted Turkish regulatory texts, fetched, byte-retained and versioned.

adapters.py says which institution may publish a source; this module makes the adapter fetch. For a catalogued
regulation that names its Mevzuat Bilgi Sistemi record (core.SourceRef) it

  1. downloads the consolidated text from mevzuat.gov.tr through the pinned fetcher (ingestion.fetch: allowlisted host,
     validated public IP, TLS, size and time limits) and refuses a text that answers for another record,
  2. parses it article by article (ingestion.mevzuat, the wrapped-text rules), and
  3. stores it as a version: the bytes as served, their SHA-256, the hash of the parsed text, the fetch time, the
     parser version and the catalogue record of the Mevzuat Bilgi Sistemi (printed title, Resmî Gazete date and issue).

A fetch whose parsed text equals the head version is UNCHANGED and stores nothing; a different text becomes the next
version and names the one it replaced, so change handling (change.py) always compares two stored texts. Bytes alone are
not a version: the service re-renders its pages, and only a change in the parsed text is a change of the regulation.

Every version directory is also a pilot source snapshot (pilot.sources.load_sources), so the extraction and policy
comparison engine reads the corpus without a second loader. A version made for a test is marked synthetic and can never
be stored in the packaged corpus.
"""
import json
import re
import time
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

from pydantic import Field

from ..ingestion.fetch import fetch, validate_url
from ..ingestion.mevzuat import KINDS, WRAPPED_TEXT_VERSION, document_label, identity, parse_mevzuat, text_url
from ..ingestion.models import PARSER_VERSION, Document, Download, IngestionError
from ..pilot.schema import Strict
from ..pilot.sources import load_sources
from .adapters import INSTITUTIONS, OfficialSource
from .core import RegulationMeta, RegulationVersion, SourceRef

CORPUS = Path(__file__).resolve().parent / 'data' / 'corpus'
INDEX_FORMAT = 'cardaman-tr-corpus/1'
RAW_FILE = 'source.html'
SNAPSHOT_FORMAT = 'regchain-fca-snapshot-v1'          # the pilot's snapshot format; the name predates the Turkish sources
DOCUMENT_TYPE = {'KANUN': 'KANUN', 'YONETMELIK': 'YONETMELIK', 'TEBLIG': 'TEBLIG'}
SEARCH_FAMILY = {'KANUN': 'kanun', 'YONETMELIK': 'yonetmelik', 'TEBLIG': 'teblig', 'CBK': 'cbk', 'KHK': 'khk', 'TUZUK': 'tuzuk'}
ENTRY_INTO_FORCE = re.compile(r'^(?:\(1\)\s*)?Bu (?:Kanun|Yönetmelik|Tebliğ)\s+yayımı\s+tarihinde\s+yürürlüğe\s+girer\.?$')


class CorpusError(ValueError):
    pass


class CatalogueRecord(Strict):
    """What the Mevzuat Bilgi Sistemi catalogue itself says about the text."""
    title: str
    kind_name: str = ''
    number: str
    tertip: str
    gazette_date: date | None = None
    gazette_number: str = ''


class CorpusVersion(Strict):
    regulation_id: str
    version_id: str = Field(min_length=1)
    previous_version_id: str | None = None
    url: str
    directory: str
    raw_hash: str
    parsed_hash: str
    parser_version: str
    fetched_at: datetime
    title: str
    articles: int
    repealed_articles: int
    publication_date: date | None = None
    # Set only when the text's own entry-into-force article says "yayımı tarihinde yürürlüğe girer" and nothing else.
    effective_date: date | None = None
    catalogue: CatalogueRecord | None = None
    warnings: list[str] = []
    synthetic: bool = False


def fold(text: str) -> str:
    """Turkish-aware case folding for comparing a printed title with a catalogued one."""
    text = text.replace('İ', 'i').replace('I', 'ı').replace('Â', 'a').replace('â', 'a').replace('Î', 'i').replace('î', 'i')
    text = text.replace('Û', 'u').replace('û', 'u').lower()
    return re.sub(r'[^0-9a-zçğıöşü]+', ' ', text).strip()


def same_title(left: str, right: str) -> bool:
    return fold(left) == fold(right)


def storage_key(ref: SourceRef) -> str:
    return f'MEVZUAT-{ref.kind}.{ref.tertip}.{ref.number}'       # the catalogue's own file naming, as pilot.sources writes it


def source_url(ref: SourceRef) -> str:
    return text_url(ref.kind, ref.number, ref.tertip)


def untitled(document: Document, ref: SourceRef) -> bool:
    return document.title == document_label(ref.kind, ref.number)


def download_text(ref: SourceRef, fetcher=fetch) -> tuple[Download, Document]:
    """The official text of one record, fetched and parsed; raises when the service answers for another record."""
    if ref.kind not in KINDS:
        raise CorpusError(f'unknown mevzuat kind code {ref.kind}')
    source = fetcher(source_url(ref))
    if identity(source.final_url) != (ref.kind, ref.number, ref.tertip):
        raise CorpusError(f'mevzuat.gov.tr answered {source.final_url} for {storage_key(ref)}')
    return source, parse_mevzuat(source, WRAPPED_TEXT_VERSION)


def catalogue_record(ref: SourceRef, titles, search=None) -> CatalogueRecord | None:
    """The catalogue row of the record, found by searching the catalogue for its title and matching kind and number.
    None when the catalogue does not list it under any of the titles tried; the version is then stored without one."""
    if search is None:
        from ..pilot.turkiye import search_mevzuat as search
    family = SEARCH_FAMILY.get(KINDS[ref.kind][0])
    if family is None:
        return None
    for title in titles:
        words = re.sub(r'\([^)]*\)', ' ', title or '')
        query = ' '.join(re.sub(r'\s+', ' ', words).strip().split()[:9])[:110]
        if len(query) < 3:
            continue
        try:
            rows = search(query, family, 25)
        except (IngestionError, ValueError):
            continue
        for row in rows:
            if row['kind'] == ref.kind and row['number'] == ref.number and row['tertip'] == ref.tertip:
                return CatalogueRecord(title=row['title'], kind_name=row.get('kind_name', ''), number=row['number'],
                                       tertip=row['tertip'], gazette_date=gazette_date(row.get('gazette_date', '')),
                                       gazette_number=row.get('gazette_number', ''))
    return None


def gazette_date(value: str) -> date | None:
    match = re.match(r'(\d{1,2})[./](\d{1,2})[./](\d{4})', value or '') or None
    if match:
        return date(int(match[3]), int(match[2]), int(match[1]))
    match = re.match(r'(\d{4})-(\d{2})-(\d{2})', value or '')
    return date(int(match[1]), int(match[2]), int(match[3])) if match else None


def effective_from(document: Document, published: date | None) -> date | None:
    """The publication date when the text's own 'Yürürlük' article says it enters into force on publication and says
    nothing else; any staged or conditional entry into force is left to a person (None)."""
    if published is None:
        return None
    for paragraph in document.paragraphs:
        heading = paragraph.heading_path[2] if len(paragraph.heading_path) >= 3 else ''
        if fold(heading) == 'yürürlük' and ENTRY_INTO_FORCE.match(paragraph.text.strip()):
            return published
    return None


class CorpusStore:
    """Versioned texts under one root: index.json plus one snapshot directory per version."""

    def __init__(self, root: Path = CORPUS):
        self.root = Path(root)
        self._index: dict[str, list[CorpusVersion]] | None = None

    # -- index --------------------------------------------------------------------------------------
    def _load(self) -> dict[str, list[CorpusVersion]]:
        if self._index is None:
            path = self.root / 'index.json'
            if path.exists():
                data = json.loads(path.read_text(encoding='utf-8'))
                if data.get('format') != INDEX_FORMAT:
                    raise CorpusError(f'corpus index format {data.get("format")!r} is not {INDEX_FORMAT}')
                self._index = {rid: [CorpusVersion.model_validate(v) for v in versions]
                               for rid, versions in data['regulations'].items()}
            else:
                self._index = {}
        return self._index

    def _save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        data = {'format': INDEX_FORMAT,
                'note': 'Official texts fetched from mevzuat.gov.tr by regchain.tr.corpus; every version keeps the bytes as '
                        'served. Never edited by hand.',
                'regulations': {rid: [v.model_dump(mode='json') for v in versions]
                                for rid, versions in sorted(self._load().items())}}
        (self.root / 'index.json').write_text(json.dumps(data, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')

    def regulation_ids(self) -> list[str]:
        return sorted(self._load())

    def versions(self, regulation_id: str) -> list[CorpusVersion]:
        """Oldest first."""
        return list(self._load().get(regulation_id, []))

    def head(self, regulation_id: str) -> CorpusVersion | None:
        versions = self.versions(regulation_id)
        return versions[-1] if versions else None

    def version(self, regulation_id: str, version_id: str | None = None) -> CorpusVersion:
        versions = self.versions(regulation_id)
        if not versions:
            raise CorpusError(f'{regulation_id} has no stored text; run `python -m regchain.tr corpus refresh`')
        if version_id is None:
            return versions[-1]
        found = [v for v in versions if v.version_id == version_id]
        if not found:
            raise CorpusError(f'{regulation_id} has no version {version_id}')
        return found[0]

    # -- writing ------------------------------------------------------------------------------------
    def record(self, regulation_id: str, ref: SourceRef, source: Download, document: Document,
               catalogue: CatalogueRecord | None = None, synthetic: bool = False) -> tuple[str, CorpusVersion]:
        """('NEW_VERSION' | 'UNCHANGED', version). The head is returned untouched when the parsed text is the same."""
        if synthetic and self.root.resolve() == CORPUS.resolve():
            raise CorpusError('a synthetic version is never stored in the packaged corpus')
        head = self.head(regulation_id)
        if head is not None and head.parsed_hash == document.content_hash:
            return 'UNCHANGED', head
        stamp = source.fetched_at.strftime('%Y%m%d')
        version_id = f'{stamp}-{document.content_hash[:12]}'
        if any(v.version_id == version_id for v in self.versions(regulation_id)):
            raise CorpusError(f'{regulation_id}: version {version_id} is already stored with another head')
        directory = f'{storage_key(ref)}/{version_id}'
        target = self.root / directory
        target.mkdir(parents=True, exist_ok=False)
        # One fixed, short name: the hash is recorded in snapshot.json and checked on every load, and a 64-character file
        # name under a deep install path exceeds the Windows path limit for tools that copy the package.
        name = RAW_FILE
        (target / name).write_bytes(source.body)
        snapshot = {'format': SNAPSHOT_FORMAT, 'parser_version': PARSER_VERSION,
                    'sources': [{'url': source.final_url, 'requested_url': source.requested_url,
                                 'fetched_at': source.fetched_at.isoformat(), 'media_type': source.media_type,
                                 'raw_hash': source.raw_hash, 'raw_file': name, 'parsed_hash': document.content_hash,
                                 'parser_version': WRAPPED_TEXT_VERSION, 'title': document.title}]}
        (target / 'snapshot.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        published = (catalogue.gazette_date if catalogue else None) or document.publication_date
        # A text without a printed title line is titled by its label ("Kanun 4760"); the catalogue's title is used then.
        title = catalogue.title if catalogue is not None and untitled(document, ref) else document.title
        version = CorpusVersion(
            regulation_id=regulation_id, version_id=version_id, previous_version_id=head.version_id if head else None,
            url=source.final_url, directory=directory, raw_hash=source.raw_hash, parsed_hash=document.content_hash,
            parser_version=WRAPPED_TEXT_VERSION, fetched_at=source.fetched_at, title=title, articles=len(document.paragraphs),
            repealed_articles=sum('DELETED_PROVISION' in p.quality_flags for p in document.paragraphs),
            publication_date=published, effective_date=effective_from(document, published), catalogue=catalogue,
            warnings=list(document.warnings), synthetic=synthetic)
        self._load().setdefault(regulation_id, []).append(version)
        self._save()
        return 'NEW_VERSION', version

    def raw_path(self, version: CorpusVersion) -> Path:
        """The file holding the bytes as served for one stored version."""
        return self.root / version.directory / RAW_FILE

    # -- reading ------------------------------------------------------------------------------------
    def sections(self, regulation_id: str, version_id: str | None = None) -> list[dict]:
        """The provisions of one stored version in the engine's section shape, each with the line spans of its source
        paragraphs (`lines`). The stored bytes are re-hashed and re-parsed on every load; a mismatch is an error."""
        version = self.version(regulation_id, version_id)
        return [dict(row) for row in _sections(str(self.root / version.directory), version.raw_hash)]

    def section(self, regulation_id: str, printed_label: str, version_id: str | None = None) -> dict:
        found = [s for s in self.sections(regulation_id, version_id) if s['printed_label'] == printed_label]
        if not found:
            raise CorpusError(f'{regulation_id} has no provision {printed_label}')
        return found[0]

    def official_source(self, regulation_id: str, meta: RegulationMeta, version_id: str | None = None) -> OfficialSource:
        """The adapters.OfficialSource record of one stored version: who published it, in which authority class."""
        version = self.version(regulation_id, version_id)
        return OfficialSource(institution_id='MEVZUAT', authority=INSTITUTIONS['MEVZUAT'].default_authority,
                              document_url=version.url, canonical_id=regulation_id, publication_date=version.publication_date,
                              effective_date=version.effective_date, version=version.version_id, content_hash=version.parsed_hash,
                              retrieved_at=version.fetched_at, document_type=DOCUMENT_TYPE[meta.regulation_type],
                              previous_version=version.previous_version_id, synthetic=version.synthetic)


@lru_cache(maxsize=256)
def _sections(directory: str, raw_hash: str) -> tuple:
    bundle, sections = load_sources(Path(directory))
    source = bundle['sources'][0]
    path = Path(directory) / source['raw_file']
    download = Download(source['requested_url'], source['url'], path.read_bytes(), source['media_type'],
                        datetime.fromisoformat(source['fetched_at']))
    spans = {p.printed_label: list(p.source_spans) for p in parse_mevzuat(download, source['parser_version']).paragraphs}
    return tuple({**row, 'lines': spans.get(row['printed_label'], [])} for row in sections)


# -- refresh -----------------------------------------------------------------------------------------
def refresh(regulations, store: CorpusStore, only=None, fetcher=fetch, search=None, pause: float = 0.6) -> list[dict]:
    """Fetch every regulation that names a source; one report row each. A failure is reported, never stored."""
    report = []
    for meta in regulations:
        if meta.source_ref is None or (only and meta.regulation_id not in only):
            continue
        try:
            source, document = download_text(meta.source_ref, fetcher)
            titles = [t for t in ('' if untitled(document, meta.source_ref) else document.title, meta.title) if t]
            catalogue = catalogue_record(meta.source_ref, titles, search)
            status, version = store.record(meta.regulation_id, meta.source_ref, source, document, catalogue)
            report.append({'regulation_id': meta.regulation_id, 'status': status, 'version_id': version.version_id,
                           'previous_version_id': version.previous_version_id, 'parsed_hash': version.parsed_hash,
                           'title': version.title, 'articles': version.articles})
        except (IngestionError, CorpusError, ValueError, OSError) as exc:
            report.append({'regulation_id': meta.regulation_id, 'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}'})
        if pause:
            time.sleep(pause)
    return report


# -- catalogue verification --------------------------------------------------------------------------
def verified_fields(meta: RegulationMeta, store: CorpusStore) -> dict:
    """The catalogue fields the stored text supports: source URL, printed title, gazette date and issue, the version chain
    with hashes. `title_matches` says whether the catalogued title is the printed one; nothing here decides VERIFIED."""
    versions = store.versions(meta.regulation_id)
    head = versions[-1]
    chain = [RegulationVersion(regulation_id=meta.regulation_id, version_id=v.version_id, publication_date=v.publication_date,
                               effective_date=v.effective_date, source_hash=v.parsed_hash,
                               previous_version_id=v.previous_version_id).model_dump(mode='json') for v in versions]
    return {'source_url': head.url, 'printed_title': head.title, 'title_matches': same_title(head.title, meta.title),
            'gazette_date': head.publication_date.isoformat() if head.publication_date else None,
            'gazette_number': head.catalogue.gazette_number if head.catalogue else None,
            'effective_date': head.effective_date.isoformat() if head.effective_date else None, 'versions': chain}


def verify_catalogue(regulations, store: CorpusStore, scopes=()) -> list[str]:
    """What a VERIFIED catalogue entry or a RESOLVED scope claims that the stored corpus does not support. Empty = sound."""
    problems = []
    by_id = {meta.regulation_id: meta for meta in regulations}
    for meta in regulations:
        if meta.metadata_status != 'VERIFIED':
            continue
        versions = store.versions(meta.regulation_id)
        if not versions:
            problems.append(f'{meta.regulation_id}: VERIFIED without a stored text')
            continue
        head = versions[-1]
        if head.synthetic:
            problems.append(f'{meta.regulation_id}: the head version is synthetic')
        if meta.source_url != head.url or source_url(meta.source_ref) != head.url:
            problems.append(f'{meta.regulation_id}: source_url is not the stored text URL')
        if not same_title(meta.title, head.title):
            problems.append(f'{meta.regulation_id}: catalogued title differs from the printed title "{head.title}"')
        stored = {v.version_id: v.parsed_hash for v in versions}
        claimed = {v.version_id: v.source_hash for v in meta.versions}
        if claimed != stored:
            problems.append(f'{meta.regulation_id}: catalogued versions {sorted(claimed)} are not the stored versions {sorted(stored)}')
        if meta.gazette_date != head.publication_date:
            problems.append(f'{meta.regulation_id}: gazette_date {meta.gazette_date} is not the recorded {head.publication_date}')
        if meta.effective_date is not None and meta.effective_date != head.effective_date:
            problems.append(f'{meta.regulation_id}: effective_date {meta.effective_date} is not derived from the text')
    for scope in scopes:
        if scope.provision_status != 'RESOLVED' or scope.regulation_id not in by_id:
            continue
        if not store.versions(scope.regulation_id):
            problems.append(f'{scope.scope_id}: provision {scope.provision_ref} is RESOLVED but {scope.regulation_id} has no stored text')
            continue
        labels = {s['printed_label']: s for s in store.sections(scope.regulation_id)}
        for label in provision_labels(scope.provision_ref):
            if label not in labels:
                problems.append(f'{scope.scope_id}: {label} is not a provision of the stored text of {scope.regulation_id}')
            elif 'DELETED_PROVISION' in labels[label]['quality_flags']:
                problems.append(f'{scope.scope_id}: {label} is repealed in the stored text')
    return problems


def provision_labels(provision_ref: str | None) -> list[str]:
    """The printed article labels a provision reference names: 'Kanun 4250 md. 6/f.1; Kanun 4250 md. 6/f.2' → the label of
    md. 6 once. A reference is one or more '<printed label>[/f.N][/b.x][/c.N]' parts joined by '; '."""
    labels = []
    for part in (provision_ref or '').split(';'):
        label = part.strip().split('/')[0].strip()
        if label and label not in labels:
            labels.append(label)
    return labels
