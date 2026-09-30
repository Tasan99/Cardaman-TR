"""Source-backed numeric facts the routing relies on (data/source_facts.json): each names the clause it was read from."""
import json
from functools import lru_cache
from pathlib import Path

FACTS = Path(__file__).resolve().parent / 'data' / 'source_facts.json'


@lru_cache(maxsize=1)
def source_facts() -> dict:
    data = json.loads(FACTS.read_text(encoding='utf-8'))
    if data.get('format') != 'cardaman-tr-source-facts/1':
        raise ValueError('source facts format is not cardaman-tr-source-facts/1')
    return data['facts']


def alcoholic_from() -> float:
    """The alcohol strength (% vol) from which a beverage is an alcoholic beverage (Yönetmelik 6203 md. 4)."""
    return float(source_facts()['alcoholic_beverage_abv_from']['value'])
