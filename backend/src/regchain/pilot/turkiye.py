"""Türkiye: finding a regulation by name and scanning the Official Gazette for changes.

Both talk only to allowlisted public catalogues through the pinned fetcher, and what is
sent is the operator's search words or a date, never company or policy content.
"""
import json
import re
from datetime import date, timedelta

from regchain.ingestion.fetch import fetch
from regchain.ingestion.mevzuat import KINDS
from regchain.ingestion.models import IngestionError

SEARCH_URL = 'https://www.mevzuat.gov.tr/anasayfa/MevzuatDatatable'
# The catalogue's own search families, as its form posts them.
SEARCH_KINDS = {'kanun': 'Kanun', 'cbk': 'CumhurbaskaniKararnameleri', 'khk': 'KHK',
                'yonetmelik': 'TumYonetmelik', 'teblig': 'Teblig', 'tuzuk': 'Tuzuk'}
GAZETTE_INDEX = 'https://www.resmigazete.gov.tr/eskiler/{year}/{month:02d}/{year}{month:02d}{day:02d}.htm'
GAZETTE_KINDS = [('KANUN', re.compile(r'\bKANUN\b|Kanunu?\b', re.I)), ('CBK', re.compile(r'Cumhurbaşkanlığı Kararnamesi', re.I)),
                 ('YONETMELIK', re.compile(r'Yönetmeli', re.I)), ('TEBLIG', re.compile(r'Tebliğ', re.I)),
                 ('KARAR', re.compile(r'Karar', re.I)), ('GENELGE', re.compile(r'Genelge', re.I))]
TAG = re.compile(r'<[^>]+>')
LINK = re.compile(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.S | re.I)


def search_mevzuat(query: str, kind: str = 'yonetmelik', limit: int = 10) -> list[dict]:
    """Regulations whose title contains the words, with the identifiers the pilot fetches by."""
    query = re.sub(r'\s+', ' ', query or '').strip()
    if len(query) < 3 or len(query) > 120:
        raise ValueError('Aramak için en az 3, en fazla 120 karakter yaz.')
    if kind not in SEARCH_KINDS:
        raise ValueError('Mevzuat türü: kanun, cbk, khk, yonetmelik, teblig veya tuzuk.')
    body = {'draw': 1, 'start': 0, 'length': max(1, min(int(limit), 25)), 'columns': [], 'order': [],
            'search': {'value': '', 'regex': False},
            'parameters': {'AranacakIfade': query, 'AranacakYer': 'Baslik', 'TamCumle': False,
                           'MevzuatTur': SEARCH_KINDS[kind], 'GenelArama': False}}
    download = fetch(SEARCH_URL, json.dumps(body, ensure_ascii=False).encode('utf-8'))
    try:
        rows = json.loads(download.body.decode('utf-8'))['data']
    except (ValueError, KeyError, TypeError) as exc:
        raise IngestionError('mevzuat.gov.tr arama yanıtı okunamadı') from exc
    results = []
    for row in rows:
        code = str(row.get('mevzuatTur', ''))
        if code not in KINDS:
            continue
        title = re.sub(r'\s+', ' ', TAG.sub('', str(row.get('mevAdi', '')))).strip()
        results.append({'kind': code, 'group': KINDS[code][0], 'number': str(row.get('mevzuatNo', '')),
                        'tertip': str(row.get('mevzuatTertip', '5')), 'title': title,
                        'kind_name': str(row.get('mevzuatTurEnumString', '')),
                        'gazette_date': str(row.get('resmiGazeteTarihi') or ''), 'gazette_number': str(row.get('resmiGazeteSayisi') or '')})
    return results


def gazette_day(day: date) -> list[dict]:
    """The Official Gazette's index for one day: legislation items with their kind, or []."""
    url = GAZETTE_INDEX.format(year=day.year, month=day.month, day=day.day)
    try:
        download = fetch(url)
    except IngestionError as exc:
        if 'HTTP 404' in str(exc):
            return []
        raise
    try:
        html = download.body.decode('cp1254')
    except UnicodeDecodeError:
        html = download.body.decode('utf-8', 'ignore')
    items = []
    for href, inner in LINK.findall(html):
        title = re.sub(r'\s+', ' ', TAG.sub('', inner).replace('&nbsp;', ' ')).strip(' -–—')
        if not title or 'ilan' in href.lower() or href.lower().endswith('.pdf'):
            continue
        kind = next((name for name, pattern in GAZETTE_KINDS if pattern.search(title)), 'DIGER')
        link = href if href.startswith('http') else f'https://www.resmigazete.gov.tr/eskiler/{day.year}/{day.month:02d}/{href}'
        items.append({'date': day.isoformat(), 'title': title[:300], 'kind': kind, 'url': link})
    return items


def gazette_scan(days: int = 7, query: str = '', today: date | None = None) -> dict:
    """Horizon scanning: every legislation item of the last N gazettes, optionally filtered."""
    days = max(1, min(int(days), 31))
    today = today or date.today()
    words = [w for w in re.split(r'\s+', (query or '').strip().lower()) if w]
    items, checked = [], []
    for offset in range(days):
        day = today - timedelta(days=offset)
        found = gazette_day(day)
        checked.append({'date': day.isoformat(), 'items': len(found)})
        for item in found:
            lowered = item['title'].lower()
            if all(w in lowered for w in words):
                items.append(item)
    return {'days': days, 'query': query or '', 'checked': checked, 'items': items,
            'note': 'Resmî Gazete başlık dizini; bir başlığın şirkete uygulanıp uygulanmadığı burada değerlendirilmez.'}
