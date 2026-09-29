"""Check retained source and policy bytes against the packet snapshot."""
from hashlib import sha256

from .paths import extended
from .policies import read_policy
from .sources import load_sources


def verify_artifacts(packet, directory):
    directory = extended(directory)
    analysis = packet['events'][0]['payload']
    _, sections = load_sources(directory/'regulatory-sources')
    by_id = {s['id']: s for s in sections}
    expected = list(analysis['scope_sources']) + [c['source'] for c in analysis['cases']]
    for source in expected:
        actual = by_id.get(source['id'])
        if actual is None or actual['text'] != source['text'] or actual['content_hash'] != source['content_hash']:
            raise ValueError('Retained FCA source differs from analysis evidence')
    for case in analysis['cases']:
        for source in case['context']['items']:
            actual = by_id.get(source['section_id'])
            if actual is None or actual['text'] != source['text'] or actual['content_hash'] != source['source_hash']:
                raise ValueError('Retained regulatory context differs from analysis evidence')
    for policy in analysis['policies']:
        files = list((directory/'policy-originals').glob(policy['raw_hash']+'.*'))
        if len(files) != 1 or sha256(files[0].read_bytes()).hexdigest() != policy['raw_hash']:
            raise ValueError('Retained policy bytes are missing or modified')
        # The packet names the extraction rules it was built with; a newer reader must not
        # make a retained original look modified.
        actual = read_policy(files[0], parser=policy.get('parser', 'pilot-policy-v1'), refuse_unreadable=False)
        if [(c['source_id'], c['text']) for c in actual['chunks']] != [(c['source_id'], c['text']) for c in policy['chunks']]:
            raise ValueError('Retained policy extraction differs from analysis evidence')
