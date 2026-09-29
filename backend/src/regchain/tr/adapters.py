"""Allowlisted official-source adapters: identity and stored-text records, not a general scraper.

A source is accepted only when its host is on the institution allowlist. Retrieval of bytes stays
in ingestion.fetch (ALLOWED_HOSTS). This module records who published the document, its authority
class, and the stored text hash. Test fixtures must set synthetic=True so they cannot be mistaken
for real law.
"""
from datetime import date, datetime
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, model_validator

from ..pilot.schema import Strict
from .core import BindingStatus

DocumentType = Literal['KANUN', 'YONETMELIK', 'TEBLIG', 'REHBER', 'KILAVUZ', 'ACIKLAMA', 'KARAR', 'OTHER']


class Institution(Strict):
    institution_id: str
    title_tr: str
    hosts: frozenset[str]
    default_authority: BindingStatus


INSTITUTIONS = {
    'TARIM_ORMAN': Institution(institution_id='TARIM_ORMAN', title_tr='Tarım ve Orman Bakanlığı',
                               hosts=frozenset({'www.tarimorman.gov.tr', 'tarimorman.gov.tr'}),
                               default_authority='OFFICIAL_GUIDANCE'),
    'TICARET': Institution(institution_id='TICARET', title_tr='Ticaret Bakanlığı',
                           hosts=frozenset({'www.ticaret.gov.tr', 'ticaret.gov.tr'}),
                           default_authority='OFFICIAL_GUIDANCE'),
    'KVKK': Institution(institution_id='KVKK', title_tr='Kişisel Verileri Koruma Kurumu',
                        hosts=frozenset({'www.kvkk.gov.tr', 'kvkk.gov.tr'}),
                        default_authority='OFFICIAL_GUIDANCE'),
    'CEVRE_SEHIRCILIK': Institution(institution_id='CEVRE_SEHIRCILIK',
                                    title_tr='Çevre, Şehircilik ve İklim Değişikliği Bakanlığı',
                                    hosts=frozenset({'www.csb.gov.tr', 'csb.gov.tr'}),
                                    default_authority='OFFICIAL_GUIDANCE'),
    'SAGLIK': Institution(institution_id='SAGLIK', title_tr='Sağlık Bakanlığı',
                          hosts=frozenset({'www.saglik.gov.tr', 'saglik.gov.tr'}),
                          default_authority='OFFICIAL_GUIDANCE'),
    'MASAK': Institution(institution_id='MASAK', title_tr='Mali Suçları Araştırma Kurulu Başkanlığı',
                         hosts=frozenset({'www.masak.gov.tr', 'masak.gov.tr'}),
                         default_authority='OFFICIAL_GUIDANCE'),
    'MEVZUAT': Institution(institution_id='MEVZUAT', title_tr='mevzuat.gov.tr (derlenmiş mevzuat)',
                           hosts=frozenset({'www.mevzuat.gov.tr', 'mevzuat.gov.tr'}),
                           default_authority='BINDING'),
    'RESMI_GAZETE': Institution(institution_id='RESMI_GAZETE', title_tr='Resmî Gazete',
                                hosts=frozenset({'www.resmigazete.gov.tr', 'resmigazete.gov.tr'}),
                                default_authority='BINDING'),
}


class OfficialSource(Strict):
    institution_id: str
    authority: BindingStatus
    document_url: str
    canonical_id: str = Field(min_length=1)
    publication_date: date | None = None
    effective_date: date | None = None
    version: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)
    retrieved_at: datetime
    document_type: DocumentType
    text: str = ''
    previous_version: str | None = None
    synthetic: bool = False

    @model_validator(mode='after')
    def _allowlisted(self):
        if self.institution_id not in INSTITUTIONS:
            raise ValueError(f'institution {self.institution_id} is not on the official-source allowlist')
        host = urlparse(self.document_url).hostname or ''
        if host not in INSTITUTIONS[self.institution_id].hosts:
            raise ValueError(f'{self.document_url}: host {host} is not allowlisted for {self.institution_id}')
        if self.authority == 'DECISION_PRECEDENT' and self.document_type != 'KARAR':
            raise ValueError('DECISION_PRECEDENT records a KARAR')
        return self


def institution_of(url: str) -> Institution | None:
    host = urlparse(url).hostname or ''
    return next((inst for inst in INSTITUTIONS.values() if host in inst.hosts), None)
