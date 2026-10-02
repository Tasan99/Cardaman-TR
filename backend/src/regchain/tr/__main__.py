"""Run: python -m regchain.tr --help

corpus refresh   fetch the official text of every catalogued regulation that names its source; store new versions
corpus status    the stored versions
corpus verify    what the catalogue or the scopes claim that the stored texts do not support
corpus apply     write the fields the stored texts support into the catalogue (regulations.json)
obligations      the clause-level obligations of a stored regulation; with --profile, where each applies
gaps             obligation -> policy -> control -> evidence -> gap for a pilot profile and its (synthetic) register;
                 with --engine, joined with the model judgements of recorded engine runs for the same entity
changes          the clauses an amendment touched since a date, the duties they carry and, with --profile, whom they reach
decisions        a stored board decision (tr/decisions.py: text, items, duties); with --profile, bound to its addressee entities and assessed
qdms export      the gap rows of a profile (sector engines, and --decision layers) as QDMS change requests: JSON, CSV, an approvals
                 template; every action stays DRAFT until a person approves its row
qdms approve     apply a filled approvals file to an export: approved rows' actions become READY_FOR_QDMS (ready-actions.json)
ai units         the reading units of some articles (what a model is given)
ai compare       rule reader against a recorded model reading: agreement, and the layer of every disagreement
ai evaluate      BEVERAGE_TR_DEV_V2 (developer labels, INDICATIVE): tasks, contrast pairs, wrong results by layer
ai read          LIVE: put the reading units of --select REGULATION=ARTICLE,ARTICLE to the local extraction model; writes --out
ai analyze       LIVE: one engine analysis of --entity of --profile against --regulation --article ...; writes --out

The two live actions call the local models the ModelRouter names (LLM_MODEL, JUDGE_MODEL, EMBED_MODEL and their digests in
the environment, CARDAMAN_MODE=development); one GPU runs them one after another. Everything else reads files only.
"""
import argparse
import json
import sys
from pathlib import Path

from .corpus import CORPUS, CorpusStore, refresh, verified_fields, verify_catalogue
from .packs import DATA, Registry, load_json


def _print(value):
    sys.stdout.write(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def apply_verified(store: CorpusStore, path: Path = DATA / 'regulations.json') -> list[dict]:
    """Rewrite the catalogue entries from the stored texts. An entry becomes VERIFIED only when its catalogued title is
    the printed title of the stored text; otherwise it keeps UNVERIFIED and the report names the printed title."""
    from .core import RegulationMeta
    catalogue = load_json(path)
    report = []
    for entry in catalogue['regulations']:
        meta = RegulationMeta.model_validate(entry)
        if meta.source_ref is None or not store.versions(meta.regulation_id):
            report.append({'regulation_id': meta.regulation_id, 'status': 'NO_STORED_TEXT'})
            continue
        fields = verified_fields(meta, store)
        entry['source_url'] = fields['source_url']
        entry['gazette_date'] = fields['gazette_date']
        entry['gazette_number'] = fields['gazette_number']
        entry['versions'] = fields['versions']
        entry['effective_date'] = fields['effective_date']
        entry['effective_status'] = 'IN_FORCE' if fields['effective_date'] else 'UNKNOWN'
        if fields['title_matches']:
            entry['metadata_status'] = 'VERIFIED'
            head = store.head(meta.regulation_id)
            entry['verification_note'] = (
                f'Title, Mevzuat Bilgi Sistemi identifiers, Resmî Gazete date and issue and the text hash were checked against '
                f'the text fetched from mevzuat.gov.tr on {head.fetched_at.date().isoformat()} ({head.articles} articles, '
                f'{head.repealed_articles} repealed). Annex tables and footnotes are not parsed; '
                + ('the effective date is the publication date the text itself names.' if fields['effective_date'] else
                   'the effective date is not derived (staged or non-standard entry into force).'))
            report.append({'regulation_id': meta.regulation_id, 'status': 'VERIFIED', 'title': fields['printed_title']})
        else:
            entry['metadata_status'] = 'UNVERIFIED'
            report.append({'regulation_id': meta.regulation_id, 'status': 'TITLE_MISMATCH', 'catalogued': meta.title,
                           'printed': fields['printed_title']})
        RegulationMeta.model_validate(entry)
    path.write_text(json.dumps(catalogue, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def _profile(registry: Registry, profile_id: str):
    from .profile import load_pilot_profiles
    profiles = {p.profile_id: p for p in load_pilot_profiles(registry.vocabulary).values()}
    if profile_id not in profiles:
        raise SystemExit(f'unknown profile {profile_id}; known: {sorted(profiles)}')
    return profiles[profile_id]


def _obligation_row(obligation, decisions=None) -> dict:
    row = {'ref': obligation.provision_ref, 'kind': obligation.kind, 'topic': obligation.topic, 'addressed_by': obligation.basis,
           'level': obligation.scope.level, 'activities': obligation.scope.activity_classes, 'entities': obligation.scope.entity_classes,
           'products': obligation.scope.product_classes or obligation.scope.alcohol_scope,
           'conditions': [c['quote'] for c in obligation.scope.conditions], 'exceptions': [e['quote'][:120] for e in obligation.scope.exceptions],
           'flags': obligation.flags, 'text': obligation.text}
    if decisions is not None:
        row['applies'] = {d.target_id: {'status': d.status, 'reasons': d.reason_codes} for d in decisions if d.status != 'DOES_NOT_APPLY'}
    return row


def _domain(args, store: CorpusStore) -> int:
    from .extraction import extract_regulation, route
    registry = Registry.load()
    articles = getattr(args, 'article', None) or None
    if args.command == 'obligations':
        profile = _profile(registry, args.profile) if args.profile else None
        _, obligations = extract_regulation(args.regulation, registry, store, articles=articles)
        _print([_obligation_row(o, route(o, profile, registry, store)[0] if profile else None) for o in obligations])
        return 0
    if args.command == 'gaps':
        from .compare import compare_profile, load_register
        profile = _profile(registry, args.profile)
        regulations = args.regulation or [rid for rid in registry.regulations if store.versions(rid)]
        obligations = [o for rid in regulations for o in extract_regulation(rid, registry, store, articles=articles)[1]]
        second = {}
        for path in args.engine:
            from .ai import second_readings
            run = json.loads(Path(path).read_text(encoding='utf-8'))
            if run['profile_id'] == profile.profile_id:
                second.setdefault(run['entity_id'], {}).update(
                    second_readings(run, [o for o in obligations if o.regulation_id == run['regulation_id']]))
        statistics, escalations = None, {}
        if args.selective:
            # Rules first; a candidate statement or an open coverage goes to the strong model, which never overrules a rule decision.
            from .adjudicate import Adjudicator, assess_profile, load_adjudications
            from .semantic import SimilarityTable, duty_text
            register = load_register(profile.profile_id)
            table = SimilarityTable.load(args.similarities) if args.similarities and Path(args.similarities).exists() else SimilarityTable()
            recorded = load_adjudications(args.adjudications)[1] if args.adjudications and Path(args.adjudications).exists() else {}
            adjudicator = Adjudicator(recorded=recorded)
            if args.embed or args.adjudicate:
                from ..model_router import ModelRouter
                from ..pilot.engine import quick
                from .live import scope
                with scope(f'tr-gaps-{profile.profile_id}'):
                    router = ModelRouter('ollama')
                    if args.embed:
                        applying = [o for o in obligations if any(d.status in ('APPLIES', 'PARTIAL') for d in route(o, profile, registry, store)[0])]
                        table.add([duty_text(o) for o in applying], [passage.text for passage in register.passages], router.embedder())
                        if args.similarities:
                            table.save(args.similarities)
                    if args.adjudicate:
                        judge = router.judge_provider()
                        adjudicator = Adjudicator(quick(judge) if args.quick else judge, args.adjudications, recorded)
                    report, assessments, statistics = assess_profile(profile, obligations, register, registry, store, table, adjudicator)
            else:
                report, assessments, statistics = assess_profile(profile, obligations, register, registry, store, table, adjudicator)
            escalations = {(a.obligation_id, a.target_id): {'reasons': a.escalation.reasons, 'statements': a.escalation.statements,
                                                           'disagreement': a.disagreement} for a in assessments if a.escalation is not None}
        else:
            report = compare_profile(profile, obligations, load_register(profile.profile_id), registry, store, second or None)
        rows = [{'ref': r.provision_ref, 'target': r.target_id, 'applicability': r.applicability, 'status': r.mapping.status,
                 'document_coverage': r.document_coverage, 'coverage_basis': r.coverage_basis, 'rule_coverage': r.rule_coverage,
                 'model_coverage': r.model_coverage, 'reasons': r.coverage_reasons, 'gap': r.gap, 'actions': r.actions,
                 'products': [f.model_dump(mode='json') for f in r.product_findings if f.result != 'WITHIN_LIMIT'],
                 'review_required': r.review_required, **({'escalation': escalations[(r.obligation_id, r.target_id)]}
                                                          if (r.obligation_id, r.target_id) in escalations else {})}
                for r in report.rows if args.all or r.mapping.status != 'COVERED']
        _print({'profile_id': report.profile_id, 'register_synthetic': report.register_synthetic, 'summary': report.summary,
                **({'selective': statistics} if statistics else {}), 'rows': rows})
        return 0
    if args.command == 'assess':
        # every sector engine the profile falls under, over the shared expert services; a recorded run replays without a model
        from .engines import ExpertServices, assess_by_engine
        profile = _profile(registry, args.profile)
        services = ExpertServices.recorded(args.similarities, [Path(p) for p in args.adjudications], registry, store)
        result = assess_by_engine(profile, services)
        rows = {pack: [{'ref': r.provision_ref, 'target': r.target_id, 'status': r.mapping.status, 'document_coverage': r.document_coverage,
                        'coverage_basis': r.coverage_basis, 'review_required': r.review_required, 'decision': a.decision,
                        'review_reasons': a.review_reasons, 'proposal': a.proposal, 'applicability_basis': a.applicability_basis}
                       for r, a in zip(report.rows, result['assessments'][pack]) if args.all or r.mapping.status != 'COVERED' or a.decision != 'AUTO']
                for pack, report in result['reports'].items()}
        reviews = {pack: [r.model_dump(mode='json') for r in report.applicability_reviews] for pack, report in result['reports'].items()}
        payload = {'profile_id': profile.profile_id, 'version': result['version'],
                   'engines': [e.model_dump(mode='json') for e in result['engines']], 'rows': rows, 'applicability_reviews': reviews}
        if args.out:
            Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
            _print({k: v for k, v in payload.items() if k not in ('rows', 'applicability_reviews')})
        else:
            _print(payload)
        return 0
    if args.command == 'decisions':
        from .decisions import DECISIONS, DecisionStore, assess_decision, decision_applicability, decision_obligations
        decisions = DecisionStore(Path(args.decisions_root) if args.decisions_root else DECISIONS)
        if args.decision not in decisions.decision_ids():
            raise SystemExit(f'no stored decision {args.decision}; stored: {decisions.decision_ids()}')
        record = decisions.record(args.decision)
        if not args.profile:
            frames, obligations = decision_obligations(record, decisions, registry)
            _print({'decision': record.model_dump(mode='json'), 'items': len(decisions.sections(record.decision_id)),
                    'frames': [{'ref': f.ref, 'kind': f.kind, 'modality': f.modality, 'marker': f.marker, 'text': f.text} for f in frames],
                    'duties': [_obligation_row(o) for o in obligations]})
            return 0
        from .compare import load_register
        from .engines import ExpertServices
        profile = _profile(registry, args.profile)
        services = ExpertServices.recorded(args.similarities, [Path(p) for p in args.adjudications], registry, store)
        report, assessments, run = assess_decision(record, profile, registry, store, decisions, load_register(profile.profile_id),
                                                   services.table, services.adjudicator, named=args.addressee)
        applicability = [{'ref': o.provision_ref, 'entity': d.target_id, 'status': d.status, 'reasons': d.reason_codes}
                         for o, d in decision_applicability(record, profile, registry, store, decisions, named=args.addressee)]
        rows = [{'ref': r.provision_ref, 'target': r.target_id, 'status': r.mapping.status, 'document_coverage': r.document_coverage,
                 'coverage_basis': r.coverage_basis, 'decision': a.decision, 'review_reasons': a.review_reasons, 'proposal': a.proposal,
                 'applicability_basis': a.applicability_basis, 'duty': r.quote, 'statements': [(s.passage_id, s.relation, s.quote) for s in r.readings
                                                                                              if s.relation != 'UNRELATED']}
                for r, a in zip(report.rows, assessments)]
        payload = {'profile_id': profile.profile_id, 'run': run.model_dump(mode='json'), 'applicability': applicability, 'rows': rows}
        if args.out:
            Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
            _print({k: v for k, v in payload.items() if k != 'rows'})
        else:
            _print(payload)
        return 0
    if args.command == 'qdms':
        from . import qdms as q
        out = Path(args.out)
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f'{out} is not empty; an export is never overwritten')
        if args.action == 'approve':
            if not args.export or not args.approvals:
                raise SystemExit('approve needs --export and --approvals')
            export, problems = q.apply_approvals(q.load_export(Path(args.export)), q.load_approvals(Path(args.approvals)))
            paths = q.write(export, out)
            (out / 'approval-problems.json').write_text(json.dumps(problems, ensure_ascii=False, indent=1), encoding='utf-8')
            _print({'summary': export.summary, 'problems': problems, 'files': {k: str(v) for k, v in paths.items()}})
            return 1 if problems else 0
        if not args.profile:
            raise SystemExit('export needs --profile')
        from .compare import load_register
        from .engines import ExpertServices, assess_by_engine
        profile = _profile(registry, args.profile)
        register = load_register(profile.profile_id)
        services = ExpertServices.recorded(args.similarities, [Path(p) for p in args.adjudications], registry, store)
        result = assess_by_engine(profile, services, register)
        export = q.export_rows(profile, register, registry, result['reports'], result['assessments'], args.all, store=store)
        if args.decision:
            from .decisions import DECISIONS, DecisionStore, LayeredStore, assess_decision, decision_meta, registry_with
            decisions = DecisionStore(Path(args.decisions_root) if args.decisions_root else DECISIONS)
            for decision_id in args.decision:
                record = decisions.record(decision_id)
                report, assessments, _ = assess_decision(record, profile, registry, store, decisions, register, services.table,
                                                         services.adjudicator, named=args.addressee)
                layer = q.export_rows(profile, register, registry_with(registry, [decision_meta(record)]), {decision_id: report},
                                      {decision_id: assessments}, True, 'DECISION', store=LayeredStore(store, decisions))
                export = q.merge(export, layer)
        paths = q.write(export, out)
        _print({'profile_id': export.profile_id, 'register_synthetic': export.register_synthetic, 'summary': export.summary,
                'files': {k: str(v) for k, v in paths.items()}})
        return 0
    if args.command == 'changes':
        from datetime import date
        from .change import impact, note_changes
        record, old, new = note_changes(args.regulation, date.fromisoformat(args.since), registry, store)
        if args.profile:
            from .compare import load_register
            profile = _profile(registry, args.profile)
            record = impact(record, old, new, profile, registry, store, load_register(profile.profile_id))
        _print(record.model_dump(mode='json'))
        return 0
    raise SystemExit(f'unknown command {args.command}')


def _ai(args, store: CorpusStore) -> int:
    from . import ai, devset
    from .live import load_readings, units_of
    registry = Registry.load()
    if args.action == 'units':
        wanted = set(args.article)
        _print([{k: u[k] for k in ('unit_id', 'mode', 'refs', 'start', 'end', 'input')}
                for section in store.sections(args.regulation) if not wanted or section['paragraph_number'] in wanted
                for u in units_of(section, args.regulation, store.head(args.regulation).version_id)])
        return 0
    if args.action in ('read', 'analyze'):
        from ..model_router import ModelRouter
        from .live import analyze_entity, read_units, scope, select_units
        if not args.out:
            raise SystemExit('--out is required')
        if args.action == 'read':
            selection = {rid: numbers.split(',') for rid, _, numbers in (item.partition('=') for item in args.select)}
            units = select_units(selection, registry, store)
            with scope('tr-corpus-read'):
                summary = read_units(units, ModelRouter('ollama').extraction_provider(), store, Path(args.out),
                                     progress=lambda i, n, ref: print(f'{i + 1}/{n} {ref}', file=sys.stderr, flush=True))
        else:
            profile = _profile(registry, args.profile)
            with scope(f'{profile.profile_id}/{args.entity}'):
                summary = analyze_entity(profile, args.entity, args.regulation, args.article, registry, store, Path(args.out),
                                         ModelRouter('ollama'), units=not args.whole_articles,
                                         progress=lambda i, n, label, stage: print(f'{stage} {i + 1}/{n} {label}', file=sys.stderr, flush=True))
        _print(summary)
        return 0
    if args.action == 'compare':
        from .extraction import extract_regulation
        header, records = load_readings(args.readings)
        regulations = sorted({r['regulation_id'] for r in records})
        refs = {ref for r in records for ref in r['refs']}
        frames = [f for rid in regulations for f in extract_regulation(rid, registry, store)[0] if f.ref in refs]
        readings = ai.compare_readings(frames, records)
        _print({'runtime': header.get('runtime'), 'summary': ai.reading_summary(readings),
                'disagreements': [r.model_dump(mode='json') for r in readings if r.diagnosis]})
        return 0
    runs = [json.loads(Path(path).read_text(encoding='utf-8')) for path in args.engine]
    table, adjudications = None, None
    if args.similarities or args.adjudications:
        from .adjudicate import load_adjudications
        from .semantic import SimilarityTable
        table = SimilarityTable.load(args.similarities) if args.similarities else None
        adjudications = {key: record for path in args.adjudications for key, record in load_adjudications(path)[1].items()}
    result = devset.evaluate(registry=registry, store=store, readings=load_readings(args.readings)[1] if args.readings else None,
                             engine_runs=runs, table=table, adjudications=adjudications)
    if not args.wrong:
        result = {k: v for k, v in result.items() if k != 'wrong'}
    _print(result)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m regchain.tr', description='Cardaman TR: official-source corpus and beverage packs')
    commands = parser.add_subparsers(dest='command', required=True)
    corpus = commands.add_parser('corpus', help='the official-source corpus')
    corpus.add_argument('action', choices=['refresh', 'status', 'verify', 'apply'])
    corpus.add_argument('--only', action='append', default=[], help='regulation id; repeat to name several')
    obligations = commands.add_parser('obligations', help='clause-level obligations of a stored regulation')
    obligations.add_argument('--regulation', required=True)
    obligations.add_argument('--profile', help='a pilot profile id: adds where each obligation applies')
    gaps = commands.add_parser('gaps', help='obligation -> policy -> control -> evidence -> gap for a pilot profile')
    gaps.add_argument('--profile', required=True)
    gaps.add_argument('--regulation', action='append', default=[], help='regulation id; repeat to name several (default: every stored text)')
    gaps.add_argument('--all', action='store_true', help='also list the covered rows')
    gaps.add_argument('--engine', action='append', default=[], help='a recorded engine run to join (live.analyze_entity); repeat to name several')
    gaps.add_argument('--selective', action='store_true',
                      help='rules, candidate statements, escalation and the recorded adjudications (adjudicate.py); prints the escalation statistics')
    gaps.add_argument('--similarities', help='selective: a recorded similarity table (semantic.SimilarityTable); written by --embed')
    gaps.add_argument('--adjudications', help='selective: a recorded adjudication file to replay; --adjudicate appends to it')
    gaps.add_argument('--embed', action='store_true', help='selective: compute the similarities with the local embedder (needs the model runtime)')
    gaps.add_argument('--adjudicate', action='store_true', help='selective: ask the strong local model about the escalated rows')
    gaps.add_argument('--quick', action='store_true', help='selective: the strong model without thinking')
    assess = commands.add_parser('assess', help='every sector engine the profile falls under, over the shared expert services (engines.py)')
    assess.add_argument('--profile', required=True)
    assess.add_argument('--similarities', help='a recorded similarity table to replay (semantic.SimilarityTable)')
    assess.add_argument('--adjudications', action='append', default=[], help='a recorded adjudication file to replay; repeat to name several')
    assess.add_argument('--all', action='store_true', help='also list the covered, automatic rows')
    assess.add_argument('--out', help='write the full result (rows included) to this file and print the engine summaries')
    decisions = commands.add_parser('decisions', help='a stored board decision (tr/decisions.py) read for its addressee entities, beside the packs')
    decisions.add_argument('--decision', required=True, help='decision id (TR:KURUL_KARARI:...), or omit --profile to list what is stored')
    decisions.add_argument('--profile', help='a pilot profile id')
    decisions.add_argument('--addressee', action='append', default=[], help='a legal entity the decision binds (also: LegalEntity.bound_by_decisions)')
    decisions.add_argument('--decisions-root', help='decision store root (default: the packaged store, data/decisions)')
    decisions.add_argument('--similarities', help='a recorded similarity table to replay')
    decisions.add_argument('--adjudications', action='append', default=[], help='a recorded adjudication file to replay; repeat to name several')
    decisions.add_argument('--out', help='write the full result (rows included) to this file and print the run record')
    qdms = commands.add_parser('qdms', help='regulation -> reasoning -> impacted policy/control -> human approval -> QDMS action (qdms.py)')
    qdms.add_argument('action', choices=['export', 'approve'])
    qdms.add_argument('--profile', help='export: a pilot profile id')
    qdms.add_argument('--similarities', help='export: a recorded similarity table to replay')
    qdms.add_argument('--adjudications', action='append', default=[], help='export: a recorded adjudication file to replay; repeat to name several')
    qdms.add_argument('--decision', action='append', default=[], help='export: also a stored board decision (TR:KURUL_KARARI:...); repeat to name several')
    qdms.add_argument('--addressee', action='append', default=[], help='export: a legal entity the decisions bind (also: LegalEntity.bound_by_decisions)')
    qdms.add_argument('--decisions-root', help='export: decision store root (default: the packaged store)')
    qdms.add_argument('--all', action='store_true', help='export: also the covered, automatic rows')
    qdms.add_argument('--export', help='approve: the export.json to apply the approvals to')
    qdms.add_argument('--approvals', help='approve: the filled approvals file (approvals-template.json with decisions)')
    qdms.add_argument('--out', required=True, help='the output directory (must not exist, or be empty)')
    changes = commands.add_parser('changes', help='clauses amended since a date, their duties and whom they reach')
    changes.add_argument('--regulation', required=True)
    changes.add_argument('--since', required=True, help='ISO date; amendment notes dated on or after it are read')
    changes.add_argument('--profile')
    ai_parser = commands.add_parser('ai', help='the two readers and the development evaluation')
    ai_parser.add_argument('action', choices=['units', 'compare', 'evaluate', 'read', 'analyze'])
    ai_parser.add_argument('--select', action='append', default=[], help='read: REGULATION=ARTICLE,ARTICLE; repeat to name several')
    ai_parser.add_argument('--profile', help='analyze: a pilot profile id')
    ai_parser.add_argument('--entity', help='analyze: a legal entity of the profile')
    ai_parser.add_argument('--out', help='read / analyze: the file to write (must not exist)')
    ai_parser.add_argument('--whole-articles', action='store_true', help='analyze: hand the engine the articles as stored, not their units')
    ai_parser.add_argument('--regulation')
    ai_parser.add_argument('--readings', help='a recorded readings file (live.read_units)')
    ai_parser.add_argument('--engine', action='append', default=[], help='a recorded engine run (live.analyze_entity); repeat to name several')
    ai_parser.add_argument('--similarities', help='evaluate: a recorded similarity table; scores the selective pipeline')
    ai_parser.add_argument('--adjudications', action='append', default=[], help='evaluate: a recorded adjudication file; repeat to name several')
    ai_parser.add_argument('--wrong', action='store_true', help='evaluate: list every wrong result with its layer')
    for sub_parser in (corpus, obligations, gaps, assess, changes, ai_parser, decisions, qdms):
        sub_parser.add_argument('--root', default=str(CORPUS), help='corpus directory (default: the packaged corpus)')
    for sub_parser in (obligations, gaps, ai_parser):
        sub_parser.add_argument('--article', action='append', default=[], help='article number; repeat to name several')
    args = parser.parse_args(argv)
    store = CorpusStore(Path(args.root))
    if args.command == 'ai':
        return _ai(args, store)
    if args.command != 'corpus':
        return _domain(args, store)
    if args.action == 'refresh':
        catalogue = Registry.load().regulations.values()
        report = refresh(catalogue, store, only=set(args.only) or None)
        _print(report)
        return 1 if any(row['status'] == 'FAILED' for row in report) else 0
    if args.action == 'status':
        _print({rid: [{'version_id': v.version_id, 'previous': v.previous_version_id, 'fetched_at': v.fetched_at,
                       'articles': v.articles, 'repealed': v.repealed_articles, 'title': v.title,
                       'gazette_date': v.publication_date} for v in store.versions(rid)] for rid in store.regulation_ids()})
        return 0
    if args.action == 'apply':
        _print(apply_verified(store))
        return 0
    registry = Registry.load()
    problems = verify_catalogue(registry.regulations.values(), store, registry.scopes.values())
    _print({'problems': problems})
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
