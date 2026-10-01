"""Replay a saved development sample on the operator-configured local model."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from regchain.evidence import digest
from .pipeline import extract
from .providers import configured_provider, runtime_manifest
from .retrieval import ContextPacket, RETRIEVER_VERSION
from .service import PROMPT_HASH


def cases_from_report(report):
    cases = []
    for row in report['results']:
        manifest = row['context']
        if manifest and manifest['retriever_version'] != RETRIEVER_VERSION:
            raise ValueError('Retriever version differs; regenerate or explicitly migrate the sample')
        packet = ContextPacket(**{k: v for k, v in manifest.items() if k != 'retriever_version'}) if manifest else None
        cases.append((row['label'], row['text'], packet))
    sample_hash = digest([{'label': label, 'text': text, 'context': packet.manifest() if packet else None}
                          for label, text, packet in cases])
    if sample_hash != report['sample_hash']:
        raise ValueError('Saved sample hash does not match its text/context')
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', required=True, help='Saved model-comparison JSON; first run defines the sample')
    parser.add_argument('--output', required=True, help='New output file (never overwritten)')
    args = parser.parse_args()
    try:
        if Path(args.output).exists():
            raise ValueError('Output already exists; use a new path')
        saved = json.loads(Path(args.sample).read_text(encoding='utf-8-sig'))[0]
        cases = cases_from_report(saved)
        provider = configured_provider('ollama')
        rows = []
        for label, text, packet in cases:
            started = time.monotonic()
            result = extract(text, provider, packet)
            rows.append({'label': label, 'text': text, 'context': packet.manifest() if packet else None,
                         'reason': result.reason, 'output': result.output.model_dump(mode='json'),
                         'diagnostics': list(result.diagnostics), 'elapsed_ms': int((time.monotonic()-started)*1000)})
            print(label, result.reason, flush=True)
        report = {'runtime': runtime_manifest(provider), 'prompt_hash': PROMPT_HASH,
                  'measured_at': datetime.now(timezone.utc).isoformat(), 'sample_hash': saved['sample_hash'],
                  'results': rows, 'limitations': 'Development sample, not independent or expert-labelled legal accuracy.'}
        with Path(args.output).open('x', encoding='utf-8') as handle:
            json.dump([report], handle, ensure_ascii=False, indent=2)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, str(exc) + '\n')


if __name__ == '__main__':
    main()
