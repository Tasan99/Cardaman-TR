"""Publish a complete, verified pilot bundle without overwriting earlier evidence."""
import json
import shutil
from pathlib import Path
from uuid import uuid4

from .evaluation import evaluate, label_template
from .paths import extended
from .report import render
from .verification import verify_artifacts


def write_json(path: Path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def save_bundle(output: Path, packet, sources: Path, policy_paths):
    # Hash-named originals under a nested run directory pass MAX_PATH on Windows.
    output, sources = extended(output), extended(sources)
    if output.exists():
        raise ValueError('Output directory already exists; use a new path')
    rendered = render(packet)
    labels = label_template(packet)
    metrics = evaluate(packet, labels)
    # A failed/crashed publication leaves a clearly named partial directory, never
    # a result that a reader could mistake for a completed analysis.
    output.parent.mkdir(parents=True, exist_ok=True)
    # The suffix only has to be unique beside one result. Keeping it short also keeps
    # retained files reachable for tools that ignore the extended namespace (Explorer,
    # OneDrive). mkdir() without exist_ok refuses a collision instead of reusing it.
    pending = output.with_name(output.name + '.partial-' + uuid4().hex[:8])
    pending.mkdir()
    write_json(pending/'packet.json', packet)
    write_json(pending/'labels.json', labels)
    write_json(pending/'evaluation.json', metrics)
    (pending/'review.html').write_text(rendered, encoding='utf-8')
    shutil.copytree(sources, pending/'regulatory-sources')
    originals = pending/'policy-originals'
    originals.mkdir()
    policies = packet['events'][0]['payload']['policies']
    if len(policy_paths) != len(policies):
        raise ValueError('Policy originals do not match the analysis')
    for path, policy in zip(policy_paths, policies):
        path = extended(path)
        shutil.copyfile(path, originals/(policy['raw_hash']+path.suffix.lower()))
    verify_artifacts(packet, pending)
    write_json(pending/'receipt.json', {'head': packet['head'], 'count': packet['count'],
        'notice': 'Keep head/count separately. This adjacent receipt is not an external anchor.'})
    # Windows and POSIX refuse to replace an existing nonempty result directory.
    if output.exists():
        raise ValueError('Output appeared during publication; prior evidence is retained')
    pending.rename(output)

