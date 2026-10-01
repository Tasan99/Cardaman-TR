"""Operator-only review packets; no automatic legal applicability or compliance verdict.

The reviewer identity is an assertion by the local operator, not authenticated identity.
Only packet integrity is verified. Keep the head hash in a separate trusted system.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .evidence import digest, make_event, verify_chain


class Record(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)


class Review(Record):
    reviewer: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class Policy(Record):
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    text: str = Field(min_length=1)

    # Evidence text must remain byte-exact, even at leading/trailing whitespace.
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=False)


class Citation(Record):
    policy_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=False)


class Obligation(Record):
    id: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    source_quote: str = Field(min_length=1)
    previous_source_quote: str | None = None
    action: str = Field(min_length=1)
    extraction_review: Review | None = None
    applicability: Literal['UNKNOWN', 'APPLIES', 'DOES_NOT_APPLY'] = 'UNKNOWN'
    applicability_review: Review | None = None
    coverage: Literal['UNKNOWN', 'COVERED', 'PARTIAL', 'MISSING'] = 'UNKNOWN'
    coverage_review: Review | None = None
    policy_evidence: list[Citation] = Field(default_factory=list)
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=False)


class Assessment(Record):
    format: Literal['regchain-review-input-v1']
    synthetic: bool = False
    company_id: str = Field(min_length=1)
    company_profile: dict
    policies: list[Policy]
    obligations: list[Obligation] = Field(min_length=1)

    @field_validator('policies', 'obligations')
    @classmethod
    def unique_ids(cls, rows):
        if len({r.id for r in rows}) != len(rows):
            raise ValueError('Duplicate ids are not allowed')
        return rows


def assess(data: Assessment) -> dict:
    policies = {p.id: p for p in data.policies}
    rows = []
    for item in data.obligations:
        if item.action not in item.source_quote:
            raise ValueError(f'{item.id}: action must be an exact source substring')
        evidence = []
        for citation in item.policy_evidence:
            policy = policies.get(citation.policy_id)
            if policy is None or not citation.quote.strip() or citation.quote not in policy.text:
                raise ValueError(f'{item.id}: policy citation does not match a supplied policy')
            evidence.append({**citation.model_dump(), 'policy_version': policy.version,
                             'policy_hash': digest(policy.model_dump())})
        if item.coverage in ('COVERED', 'PARTIAL') and not evidence:
            raise ValueError(f'{item.id}: coverage requires policy evidence')
        if item.extraction_review is None:
            status = 'EXTRACTION_REVIEW_REQUIRED'
        elif item.applicability == 'UNKNOWN' or item.applicability_review is None:
            status = 'APPLICABILITY_REVIEW_REQUIRED'
        elif item.applicability == 'DOES_NOT_APPLY':
            status = 'REVIEWED_NOT_APPLICABLE'
        elif item.coverage == 'UNKNOWN' or item.coverage_review is None:
            status = 'POLICY_REVIEW_REQUIRED'
        elif item.coverage == 'COVERED':
            status = 'REVIEWED_POLICY_COVERAGE'
        else:
            status = 'REVIEWED_POLICY_GAP'
        change = 'NO_BASELINE' if item.previous_source_quote is None else (
            'UNCHANGED' if item.previous_source_quote == item.source_quote else 'SOURCE_CHANGED')
        rows.append({'obligation_id': item.id, 'status': status, 'change': change,
                     'source_version': item.source_version, 'source_hash': digest(item.source_quote),
                     'previous_source_hash': digest(item.previous_source_quote) if item.previous_source_quote is not None else None,
                     'affected_policy_ids': sorted({c.policy_id for c in item.policy_evidence}),
                     'policy_evidence': evidence})
    snapshot = data.model_dump(mode='json')
    payload = {'kind': 'regchain-review-assessment-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
               'input_hash': digest(snapshot), 'input': snapshot, 'results': rows,
               'limitations': 'Operator assertions, not authenticated legal approval. Policy coverage does not prove operational compliance. No blockchain transaction or trusted timestamp.'}
    event = make_event(payload)
    return {'events': [event], 'head': event['event_hash'], 'count': 1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build')
    build.add_argument('--input', required=True)
    build.add_argument('--output', required=True)
    check = commands.add_parser('verify')
    check.add_argument('--input', required=True)
    check.add_argument('--expected-head', required=True, help='Head preserved separately from the packet')
    check.add_argument('--expected-count', required=True, type=int)
    args = parser.parse_args()
    try:
        raw = Path(args.input).read_text(encoding='utf-8-sig')
        if args.command == 'build':
            packet = assess(Assessment.model_validate_json(raw))
            # Never overwrite a prior audit artifact by accident.
            with Path(args.output).open('x', encoding='utf-8') as handle:
                json.dump(packet, handle, ensure_ascii=False, indent=2)
            print(json.dumps({'head': packet['head'], 'count': packet['count']}))
        else:
            packet = json.loads(raw)
            if not verify_chain(packet['events'], args.expected_head, args.expected_count):
                parser.exit(1, 'Evidence verification failed\n')
            print('Evidence integrity verified against supplied head and count')
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, str(exc) + '\n')


if __name__ == '__main__':
    main()
