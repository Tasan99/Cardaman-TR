"""Run: python -m regchain.pilot --help"""
import argparse
import json
import os
from pathlib import Path
from uuid import uuid4

from regchain.evidence import verify_chain
from regchain.model_router import ModelPolicyError, ModelRouter, ModelScope, bind_model_scope, model_mode
from .engine import analyze, load_packet
from .evaluation import evaluate, label_template
from .policies import read_policy
from .report import render
from .review import apply_review
from .schema import Company, Review
from .sources import load_sources, save_sources, select_targets
from .verification import verify_artifacts
from .artifacts import save_bundle


def write(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Local FCA CONC pilot with mandatory human review')
    commands = parser.add_subparsers(dest='command', required=True)
    source = commands.add_parser('sources', help='Fetch and retain an FCA Handbook chapter (default: CONC 7.1 and 7.3)')
    source.add_argument('--output', required=True)
    source.add_argument('--module', help='Handbook module such as COBS; with --chapter fetches the whole chapter')
    source.add_argument('--chapter', help='Chapter number such as 4 or 5D')
    run = commands.add_parser('run')
    run.add_argument('--company', required=True)
    run.add_argument('--policy', action='append', required=True, help='Repeat for each TXT/MD/PDF/DOCX policy')
    run.add_argument('--sources', required=True)
    run.add_argument('--output', required=True, help='New directory; existing outputs are never overwritten')
    run.add_argument('--label', action='append', help='Default: CONC 7.3.4 and CONC 7.3.4B')
    run.add_argument('--module', default='CONC')
    run.add_argument('--chapter', default='7')
    run.add_argument('--selection', choices=['labels', 'rules', 'all'], default='labels',
                     help='rules/all analyse every provision of the chapter found in --sources')
    run.add_argument('--section', action='append', default=[], help='With rules/all: only this section, e.g. 4.2')
    run.add_argument('--provider', choices=['ollama', 'rules'], required=True)
    run.add_argument('--retrieval', choices=['lexical', 'hybrid'], default='lexical',
                     help='hybrid adds local embedding similarity (EMBED_MODEL); never a silent fallback')
    run.add_argument('--previous', help='Previous analysis packet to compare source/company/policy versions')
    review = commands.add_parser('review')
    review.add_argument('--packet', required=True)
    review.add_argument('--decisions', required=True)
    review.add_argument('--output', required=True)
    check = commands.add_parser('verify')
    check.add_argument('--packet', required=True)
    check.add_argument('--expected-head', required=True)
    check.add_argument('--expected-count', type=int, required=True)
    check.add_argument('--artifacts', help='Directory containing retained originals; defaults to packet directory')
    evaluation = commands.add_parser('evaluate')
    evaluation.add_argument('--packet', required=True)
    evaluation.add_argument('--labels', required=True)
    evaluation.add_argument('--output', required=True)
    report = commands.add_parser('report', help='Render an existing packet without another model call')
    report.add_argument('--packet', required=True)
    report.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        if args.command == 'sources':
            if bool(args.module) != bool(args.chapter):
                raise ValueError('Give --module and --chapter together')
            bundle = save_sources(Path(args.output), module=args.module, chapter=args.chapter)
            print('Retained', len(bundle['sources']), 'FCA chapter sources')
        elif args.command == 'run':
            if model_mode() != 'development':
                raise ModelPolicyError('The local pilot CLI requires development mode; authenticated company API scope is not available here')
            output = Path(args.output)
            if output.exists():
                raise ValueError('Output directory already exists; retain prior evidence and use a new path')
            company = Company.model_validate_json(Path(args.company).read_text(encoding='utf-8-sig'))
            bundle, sections = load_sources(Path(args.sources))
            policies = [read_policy(Path(p)) for p in args.policy]
            if len({p['raw_hash'] for p in policies}) != len(policies):
                raise ValueError('Duplicate policy file content')
            previous, previous_head, previous_payload = None, None, None
            if args.previous:
                prior = load_packet(Path(args.previous))
                previous_payload = prior['events'][0]['payload']
                if previous_payload['company']['id'] != company.id:
                    raise ValueError('Previous analysis belongs to another company')
                previous = [c['source'] for c in previous_payload['cases']]
                previous_head = prior['head']
            labels = select_targets(sections, args.module, args.chapter, args.selection, args.section,
                                    args.label or ['CONC 7.3.4', 'CONC 7.3.4B'])
            with bind_model_scope(ModelScope(os.environ.get('TENANT_ID', 'local'), company.id, uuid4().hex)):
                models = ModelRouter(args.provider)
                provider = models.extraction_provider()
                embedder = models.embedder() if args.retrieval == 'hybrid' else None
                packet = analyze(company, policies, sections, provider, labels, previous, previous_head,
                                 previous_payload, embedder=embedder, judge=models.judge_provider())
            save_bundle(output, packet, Path(args.sources), args.policy)
            print(json.dumps({'output': str(output.resolve()), 'head': packet['head'], 'count': packet['count'],
                'candidates': len(packet['events'][0]['payload']['obligations']), 'expert_review': 'PENDING'}))
        elif args.command == 'review':
            packet = load_packet(Path(args.packet))
            decisions = Review.model_validate_json(Path(args.decisions).read_text(encoding='utf-8-sig'))
            result = apply_review(packet, decisions)
            write(Path(args.output), result)
            print(json.dumps({'head': result['head'], 'count': result['count']}))
        elif args.command == 'verify':
            packet = load_packet(Path(args.packet))
            if not verify_chain(packet['events'], args.expected_head, args.expected_count):
                raise ValueError('Packet does not match separately supplied head/count')
            verify_artifacts(packet, Path(args.artifacts) if args.artifacts else Path(args.packet).parent)
            print('Packet and retained source/policy integrity verified; no identity or legal accuracy claim')
        elif args.command == 'evaluate':
            packet = load_packet(Path(args.packet))
            labels = json.loads(Path(args.labels).read_text(encoding='utf-8-sig'))
            result = evaluate(packet, labels)
            write(Path(args.output), result)
            print(json.dumps(result))
        elif args.command == 'report':
            packet = load_packet(Path(args.packet))
            with Path(args.output).open('x', encoding='utf-8') as handle:
                handle.write(render(packet))
            print(str(Path(args.output).resolve()))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, str(exc)+'\n')


if __name__ == '__main__':
    main()
