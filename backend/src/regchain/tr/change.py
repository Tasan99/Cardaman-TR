"""Regulatory version impact: stored source versions, not a news summary.

A change is identified by content hash. When two stored versions of the same regulation differ,
the record names the changed provisions and the obligations / profile targets / mapped documents
that those provisions already reach. Horizon scanning is out of scope; this is the data structure
the later scanner writes into.
"""
from typing import Literal

from pydantic import Field

from ..pilot.schema import Strict

ChangeKind = Literal['ADDED', 'REMOVED', 'MODIFIED']
ImpactDelta = Literal['NEWLY_APPLIES', 'NO_LONGER_APPLIES', 'CHANGED_REQUIREMENT', 'UNCHANGED', 'UNKNOWN',
                      'NO_TARGET_AFFECTED']


class SourceVersion(Strict):
    regulation_id: str
    version_id: str
    content_hash: str = Field(min_length=1)
    previous_version_id: str | None = None


class ProvisionChange(Strict):
    provision_ref: str
    kind: ChangeKind
    old_hash: str | None = None
    new_hash: str | None = None
    impact_types: list[str] = []


class AffectedTarget(Strict):
    obligation_id: str
    decision_before: str | None = None
    decision_after: str | None = None
    delta: ImpactDelta
    entity_ids: list[str] = []
    facility_ids: list[str] = []
    product_ids: list[str] = []
    document_ids: list[str] = []
    control_ids: list[str] = []
    departments: list[str] = []
    actions: list[dict] = []


class ImpactRecord(Strict):
    change_id: str
    regulation_id: str
    from_version: str
    to_version: str
    hash_changed: bool
    provision_changes: list[ProvisionChange] = []
    affected: list[AffectedTarget] = []


def compare_versions(old: SourceVersion, new: SourceVersion) -> ImpactRecord:
    if old.regulation_id != new.regulation_id:
        raise ValueError(f'versions belong to different regulations: {old.regulation_id}, {new.regulation_id}')
    changed = old.content_hash != new.content_hash
    return ImpactRecord(change_id=f'{old.regulation_id}:{old.version_id}->{new.version_id}',
                        regulation_id=old.regulation_id, from_version=old.version_id, to_version=new.version_id,
                        hash_changed=changed,
                        provision_changes=[] if not changed else [ProvisionChange(provision_ref='UNRESOLVED',
                                                                                  kind='MODIFIED',
                                                                                  old_hash=old.content_hash,
                                                                                  new_hash=new.content_hash)])


def attach_targets(record: ImpactRecord, targets: list[AffectedTarget]) -> ImpactRecord:
    """Join changed provisions to the profile targets already linked to those obligations."""
    if not record.hash_changed:
        return record.model_copy(update={'affected': [AffectedTarget(obligation_id=t.obligation_id,
                                                                    delta='UNCHANGED',
                                                                    entity_ids=t.entity_ids, facility_ids=t.facility_ids,
                                                                    product_ids=t.product_ids)] for t in targets})
    if not targets:
        return record.model_copy(update={'affected': [AffectedTarget(obligation_id='NONE',
                                                                    delta='NO_TARGET_AFFECTED')]})
    return record.model_copy(update={'affected': targets})
