"""Pending label changes and their approval (v0.19 Phase 14): nobody changes a golden label alone.

Until v0.18 a label change was one entry typed into evaluation/reviewed_labels/<dataset>.reviewed.json
with a hand-written status: no approver, no history, and nothing kept an ACCEPTED status from being
typed by the person who wanted the number to move. Here a change goes through three files next to
that reviewed file, and the dataset file itself is never written:

- ``<dataset_id>.pending.json`` (cardaman-pending-labels-v1): proposed changes. ``propose`` checks
  that the change names exactly one expectation and that its "from" is the label as it is now, and
  records who proposed it, why, where the proposal came from (label-audit, pilot-review,
  expert-review, manual) and the evidence it rests on. A proposal stays PENDING until a person acts.
- ``<dataset_id>.audit.jsonl``: append-only, one event per line {seq, at, action (PROPOSED |
  APPROVED | REJECTED | WITHDRAWN), entry_id, actor, role, entry_sha256, prev_hash, hash}; ``hash``
  is regchain.evidence.digest over the other fields, ``prev_hash`` the hash of the line before
  (64 zeros for the first). entry_sha256 is the digest of the entry as that step left it. The
  pending file records the head and count of the log, so an edited, removed or re-ordered line and
  a cut tail are all found by ``verify``, and every step (propose, approve, reject, withdraw) refuses
  to append to a log whose chain or head does not verify. A change's "from" is the label as the
  approved entries left it, and one label field has at most one PENDING change.
- ``<dataset_id>.reviewed.json`` (cardaman-reviewed-labels-v1, v0.18): ``approve`` moves an entry
  there as ACCEPTED with approved_by, approved_at, entry_id, pending_sha256 (the digest of the
  proposal as approved) and label_source_after=HUMAN_REVIEWED. The approver must be another person
  than the proposer (four-eyes) unless ``allow_self_approval`` is given, which the entry then says.

labels.apply_reviewed(strict=True) lays over only entries this log approved, unedited. The chain
proves the files were not edited after the fact; it does not authenticate people: actor and role
are what the operator typed, as for pilot reviews (identity_assurance LOCAL_OPERATOR_ASSERTION).

``import_review`` turns the OVERRIDE decisions of a pilot review of an evaluation packet into
proposals (the review's analysis_head names the packet of one case of the run); ``import_sheet``
does the same for a filled expert-review sheet (expert_export.py). Both only propose.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from regchain.evidence import digest
from .identifiers import clause_id
from .labels import EDITABLE, REVIEWED_FORMAT, file_sha256, find_targets, resolved_source, reviewed_template, row_id

PENDING_FORMAT = 'cardaman-pending-labels-v1'
AUDIT_ACTIONS = ('PROPOSED', 'APPROVED', 'REJECTED', 'WITHDRAWN')
TERMINAL = ('APPROVED', 'REJECTED', 'WITHDRAWN')
ORIGINS = ('label-audit', 'pilot-review', 'expert-review', 'manual')
EVENT_FIELDS = ('seq', 'at', 'action', 'entry_id', 'actor', 'role', 'entry_sha256', 'prev_hash')
GENESIS = '0' * 64
HUMAN_REVIEWED = 'HUMAN_REVIEWED'
PENDING_NOTE = ('Proposed label changes waiting for a decision. Nothing here changes a score: an entry counts only after '
                '`python -m regchain.evaluation labels approve --dataset <file> --ids <entry_id> --approver <name>` by a person other '
                'than the proposer moves it to <dataset_id>.reviewed.json. Every step is appended to <dataset_id>.audit.jsonl '
                '(hash-chained); `labels verify` checks the chain and that the three files agree. The dataset file is never written.')


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def plain(value):
    """JSON the evidence digest accepts: floats (refused by regchain.evidence) become strings."""
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def entry_digest(entry: dict) -> str | None:
    """The digest of an entry as stored (None when it holds something the canonical form refuses)."""
    try:
        return digest(entry)
    except (ValueError, TypeError):
        return None


def event_hash(event: dict) -> str:
    return digest({key: event.get(key) for key in EVENT_FIELDS})


def default_directory(dataset_path) -> Path:
    """evaluation/reviewed_labels for a dataset under evaluation/<folder>/ (datasets/, independent/)."""
    return Path(dataset_path).resolve().parent.parent / 'reviewed_labels'


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path: Path, data: dict):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read_audit(path: Path) -> list:
    """The events of an audit log; a line that is not JSON becomes {'unreadable': line} so verify can name it."""
    events = []
    if Path(path).is_file():
        for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    events.append({'unreadable': line[:120]})
    return events


def chain_problems(events) -> list:
    """What is wrong with the hash chain itself (sequence, links, hashes); [] when it holds."""
    problems, previous = [], GENESIS
    for index, event in enumerate(events, 1):
        if 'unreadable' in event or not isinstance(event, dict):
            problems.append(f'line {index}: not a JSON event')
            previous = None
            continue
        missing = [key for key in (*EVENT_FIELDS, 'hash') if key not in event]
        if missing:
            problems.append(f'line {index}: missing {", ".join(missing)}')
        if event.get('seq') != index:
            problems.append(f'line {index}: seq {event.get("seq")!r} (expected {index})')
        if event.get('action') not in AUDIT_ACTIONS:
            problems.append(f'line {index}: unknown action {event.get("action")!r}')
        if previous is not None and event.get('prev_hash') != previous:
            problems.append(f'line {index}: prev_hash does not link to line {index - 1}')
        try:
            if event.get('hash') != event_hash(event):
                problems.append(f'line {index}: hash does not match the event (edited)')
        except (ValueError, TypeError):
            problems.append(f'line {index}: holds values the canonical form refuses')
        previous = event.get('hash')
    return problems


def approvals(audit_path: Path):
    """({entry_id: APPROVED event}, chain problems) of an audit log, for labels.apply_reviewed(strict=True)."""
    events = read_audit(audit_path)
    return {e['entry_id']: e for e in events if isinstance(e, dict) and e.get('action') == 'APPROVED'}, chain_problems(events)


def parse_value(field: str, text):
    """A label value typed on the command line or in a sheet, as the dataset stores it."""
    if not isinstance(text, str):
        return text
    raw = text.strip()
    if field in ('conflict', 'required'):
        # Turkish Excel re-saves a true/false text as DOĞRU/YANLIŞ; the Turkish letters are folded to ASCII to compare.
        folded = raw.lower().translate(str.maketrans('çğıöşü', 'cgiosu'))
        if folded in ('true', 'yes', 'evet', '1', 'var', 'dogru'):
            return True
        if folded in ('false', 'no', 'hayir', '0', 'yok', 'yanlis'):
            return False
        if field == 'conflict' and folded in ('null', 'none', ''):
            return None
        raise ValueError(f'{field} must be true or false' + (' or null' if field == 'conflict' else '') + f', not {text!r}')
    if field == 'evidence':
        value = json.loads(raw) if raw.startswith('[') else [part.strip() for part in raw.split('||') if part.strip()]
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ValueError('evidence must be a JSON list of strings')
        return value
    return raw.upper()


class Ledger:
    """The pending file, the audit log and the reviewed file of one dataset (the dataset is only read)."""

    def __init__(self, dataset_path, directory=None):
        from .harness import load_dataset
        self.dataset_path = Path(dataset_path)
        self.dataset = load_dataset(self.dataset_path)
        self.dataset_id = self.dataset.dataset_id
        self.directory = Path(directory) if directory else default_directory(self.dataset_path)
        self.pending_path = self.directory / f'{self.dataset_id}.pending.json'
        self.audit_path = self.directory / f'{self.dataset_id}.audit.jsonl'
        self.reviewed_path = self.directory / f'{self.dataset_id}.reviewed.json'
        self.data = json.loads(self.dataset.model_dump_json())
        self.cases = {case['case_id']: case for case in self.data['cases']}

    # ------------------------------------------------------------------ files
    def pending(self) -> dict:
        if self.pending_path.is_file():
            record = read_json(self.pending_path)
            if record.get('format') != PENDING_FORMAT or record.get('dataset_id') != self.dataset_id:
                raise ValueError(f'{self.pending_path} is not a {PENDING_FORMAT} file of {self.dataset_id}')
            return record
        return {'format': PENDING_FORMAT, 'dataset_id': self.dataset_id, 'dataset_version': self.dataset.version,
                'dataset_path': self.dataset_path.as_posix(), 'note': PENDING_NOTE,
                'audit': {'file': self.audit_path.name, 'head': GENESIS, 'count': 0}, 'entries': []}

    def reviewed(self) -> dict:
        if self.reviewed_path.is_file():
            record = read_json(self.reviewed_path)
            if record.get('format') != REVIEWED_FORMAT or record.get('dataset_id') != self.dataset_id:
                raise ValueError(f'{self.reviewed_path} is not a {REVIEWED_FORMAT} file of {self.dataset_id}')
            return record
        return reviewed_template(self.dataset_path)

    def events(self) -> list:
        return read_audit(self.audit_path)

    def append(self, events, action, entry, actor, role) -> dict:
        """Append one event to the log (and to ``events``, the log as read at the start of this command)."""
        previous = events[-1]['hash'] if events else GENESIS
        event = {'seq': len(events) + 1, 'at': now(), 'action': action, 'entry_id': entry['entry_id'], 'actor': actor, 'role': role,
                 'entry_sha256': digest(entry), 'prev_hash': previous}
        event['hash'] = event_hash(event)
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open('a', encoding='utf-8', newline='\n') as log:
            log.write(json.dumps(event, ensure_ascii=False, separators=(',', ':')) + '\n')
        events.append(event)
        return event

    def save_pending(self, record, events):
        record['audit'] = {'file': self.audit_path.name, 'head': events[-1]['hash'] if events else GENESIS, 'count': len(events)}
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(self.pending_path, record)

    def usable_chain(self, events, record):
        """The log, when a new step may be appended to it: its hash chain holds and it is the log the pending file recorded.

        The chain alone does not show a cut tail (the remaining lines still link), and save_pending then overwrote
        the recorded head and count with the cut log, so a removed REJECTED line was caught only if `labels
        verify` happened to run in between (review of 24 September 2026: the rejected proposal came back and was
        approved with no trace). Every step now makes the same head check verify makes.
        """
        problems = chain_problems(events)
        if problems:
            raise ValueError(f'{self.audit_path} fails verification ({problems[0]}); run `labels verify` and repair it before any new step')
        if events and not self.pending_path.is_file():
            raise ValueError(f'{self.audit_path.name} has events but {self.pending_path.name} is missing; run `labels verify` and repair it '
                             'before any new step')
        head = record.get('audit') or {}
        actual = (events[-1]['hash'] if events else GENESIS, len(events))
        if (head.get('head'), head.get('count')) != actual:
            raise ValueError(f'the pending file records the log at {head.get("count")} events / head {str(head.get("head"))[:12]}, the log has '
                             f'{actual[1]} / {actual[0][:12]} (lines cut, added or edited); run `labels verify` and repair it before any new step')
        return events

    # ------------------------------------------------------------------ steps
    def current_cases(self) -> dict:
        """{case_id: dumped case} with the labels as they stand: the dataset with the reviewed file's ACCEPTED entries laid
        over it, as scoring lays them (labels.apply_reviewed; strict when the audit log exists)."""
        if not self.reviewed_path.is_file():
            return self.cases
        from .labels import apply_reviewed
        overlaid, _ = apply_reviewed(self.dataset, self.reviewed_path, self.dataset_path, audit_path=self.audit_path)
        return {case['case_id']: case for case in json.loads(overlaid.model_dump_json())['cases']}

    def target(self, case_id, article, clause='', action_keywords=None, cases=None) -> dict:
        case = (self.cases if cases is None else cases).get(case_id)
        if case is None:
            raise ValueError(f'{self.dataset_id} has no case {case_id}')
        targets = find_targets(case, article, clause, action_keywords)
        if len(targets) != 1:
            where = f'{case_id} md.{article}' + (f'({clause_id(clause)})' if clause_id(clause or '') else '')
            raise ValueError(f'{len(targets)} expectations of {where} match' + (' (name one with its action keywords)' if targets else ''))
        return targets[0]

    def check_change(self, change: dict, cases=None) -> tuple:
        """(target, from, to) of a proposed change, or ValueError saying why it cannot be proposed.

        "from" is checked against the label as it stands (``cases``, default current_cases()): the dataset with
        the approved entries applied. Checked against the dataset file alone, a second change of an approved
        field (COVERS_TEXT -> NO_EVIDENCE after COVERS_TEXT -> PARTIAL) was approved too, and the overlay then
        skipped the later human decision in silence. The returned target is the dict in ``cases``.
        """
        from .schema import ExpectedObligation
        field = change.get('field')
        if field not in EDITABLE:
            raise ValueError(f'field {field!r} cannot be reviewed (one of {", ".join(EDITABLE)})')
        target = self.target(change.get('case_id'), change.get('article'), change.get('clause') or '', change.get('action_keywords'),
                             self.current_cases() if cases is None else cases)
        current = target[field]
        before = parse_value(field, change['from']) if 'from' in change else current
        if before != current:
            raise ValueError(f'the label is now {current!r}, the proposal changes {before!r}')
        after = parse_value(field, change.get('to'))
        if after == current:
            raise ValueError(f'the label already is {current!r}')
        ExpectedObligation.model_validate({**target, field: after})       # the new label must satisfy the schema
        return target, current, after

    def propose(self, changes, actor, role='proposer', origin='manual') -> list:
        """Validate every change, then append them as PENDING (all or none)."""
        if not str(actor or '').strip() or not str(role or '').strip():
            raise ValueError('a proposal needs a nonblank actor and role')
        if origin not in ORIGINS:
            raise ValueError(f'origin must be one of {", ".join(ORIGINS)}')
        record = self.pending()
        events = self.usable_chain(self.events(), record)
        # One open change per label field: a second proposal for a field that already has a PENDING one (another
        # "to") would be approvable only until the first is approved; it is refused until that one is decided.
        open_changes = {self.field_key(e): e for e in record['entries'] if e.get('status') == 'PENDING'}
        at, sha, entries, cases = now(), file_sha256(self.dataset_path), [], self.current_cases()
        for change in changes:
            if not str(change.get('reason') or '').strip():
                raise ValueError(f'{change.get("case_id")} md.{change.get("article")} {change.get("field")}: a proposal needs a reason')
            target, before, after = self.check_change(change, cases)
            body = {'case_id': change['case_id'], 'article': target['article'], 'clause': target['clause'],
                    'action_keywords': list(target['action_keywords']), 'field': change['field'], 'from': before, 'to': after}
            key = self.field_key(body)
            if key in open_changes:
                other = open_changes[key]
                raise ValueError(f'the same change is already pending as {other["entry_id"]}' if other['to'] == after else
                                 f'another change of this field is already pending as {other["entry_id"]} ({other["from"]!r} -> {other["to"]!r}); '
                                 'approve, reject or withdraw it first')
            entry = {'entry_id': digest({'dataset_id': self.dataset_id, **body, 'proposed_at': at, 'actor': actor})[:16], 'status': 'PENDING',
                     **body, 'reason': str(change['reason']).strip(), 'origin': origin, 'proposed_by': actor, 'proposer_role': role,
                     'proposed_at': at, 'dataset_sha256': sha,
                     'label_source_before': resolved_source(target, change['field'], self.data.get('default_label_source')),
                     'evidence': plain(change.get('evidence') or {})}
            open_changes[key] = entry
            entries.append(entry)
        for entry in entries:
            self.append(events, 'PROPOSED', entry, actor, role)
            record['entries'].append(entry)
        if entries:
            self.save_pending(record, events)
        return entries

    @staticmethod
    def change_key(entry) -> str:
        return json.dumps([entry['case_id'], str(entry['article']), clause_id(entry.get('clause') or ''), list(entry.get('action_keywords') or []),
                           entry['field'], entry['to']], ensure_ascii=False)

    @staticmethod
    def field_key(entry) -> str:
        """The label field an entry changes (row and field, whatever the new value)."""
        return json.dumps([entry['case_id'], str(entry['article']), clause_id(entry.get('clause') or ''), list(entry.get('action_keywords') or []),
                           entry['field']], ensure_ascii=False)

    def select(self, record, ids, status='PENDING') -> list:
        """The entries named by id (a unique prefix of at least 6 characters is enough), all with ``status``."""
        chosen = []
        for wanted in ids:
            found = [e for e in record['entries'] if e['entry_id'] == wanted] or \
                    ([e for e in record['entries'] if len(wanted) >= 6 and e['entry_id'].startswith(wanted)])
            if len(found) != 1:
                raise ValueError(f'{wanted}: {len(found)} pending entries match')
            if found[0].get('status') != status:
                raise ValueError(f'{found[0]["entry_id"]} is {found[0].get("status")}, not {status}')
            if found[0] not in chosen:
                chosen.append(found[0])
        if not chosen:
            raise ValueError('no entry named')
        return chosen

    def approve(self, ids, approver, role='approver', allow_self_approval=False, note='') -> list:
        if not str(approver or '').strip() or not str(role or '').strip():
            raise ValueError('an approval needs a nonblank approver and role')
        record = self.pending()
        events = self.usable_chain(self.events(), record)
        chosen = self.select(record, ids)
        proposals = {e['entry_id']: e for e in events if e.get('action') == 'PROPOSED'}
        moved, cases = [], self.current_cases()
        for entry in chosen:
            if entry['proposed_by'].strip().casefold() == approver.strip().casefold() and not allow_self_approval:
                raise ValueError(f'{entry["entry_id"]} was proposed by {entry["proposed_by"]}; another person approves it (four-eyes), '
                                 'or pass --allow-self-approval, which the entry then records')
            proposal = proposals.get(entry['entry_id'])
            if proposal is None or proposal['entry_sha256'] != entry_digest(entry):
                raise ValueError(f'{entry["entry_id"]} differs from the proposal the audit log recorded (edited after it was proposed)')
            # Still the label it changes (as the entries approved before it, also in this command, left it), still valid.
            target, before, after = self.check_change(entry, cases)
            target[entry['field']] = after
            at = now()
            reviewed = {'case_id': entry['case_id'], 'article': entry['article'], 'clause': entry['clause'],
                        **({'action_keywords': entry['action_keywords']} if entry['action_keywords'] else {}),
                        'field': entry['field'], 'from': before, 'to': after, 'reviewer': approver, 'reviewed_at': at, 'reason': entry['reason'],
                        'status': 'ACCEPTED', 'entry_id': entry['entry_id'], 'approved_by': approver, 'approver_role': role, 'approved_at': at,
                        'pending_sha256': proposal['entry_sha256'], 'label_source_before': entry.get('label_source_before'),
                        'label_source_after': HUMAN_REVIEWED, 'proposed_by': entry['proposed_by'], 'proposer_role': entry['proposer_role'],
                        'proposed_at': entry['proposed_at'], 'origin': entry['origin'], 'evidence': entry.get('evidence') or {},
                        'self_approved': entry['proposed_by'].strip().casefold() == approver.strip().casefold()}
            if str(note or '').strip():
                reviewed['approval_note'] = str(note).strip()
            moved.append(reviewed)
        # The reviewed file first: an entry there without its APPROVED event is ignored by a strict overlay,
        # so a command stopped half-way can leave an inconsistency verify reports, never an unapproved label.
        book = self.reviewed()
        book['entries'] = [*(book.get('entries') or []), *moved]
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(self.reviewed_path, book)
        for reviewed in moved:
            self.append(events, 'APPROVED', reviewed, approver, role)
        ids_moved = {r['entry_id'] for r in moved}
        record['entries'] = [e for e in record['entries'] if e['entry_id'] not in ids_moved]
        self.save_pending(record, events)
        return moved

    def close(self, ids, actor, reason, role, action) -> list:
        """REJECTED (by an approver) or WITHDRAWN (by anyone, usually the proposer): kept in the pending file with the decision."""
        if not str(actor or '').strip() or not str(role or '').strip() or not str(reason or '').strip():
            raise ValueError(f'{action.lower()} needs a nonblank actor, role and reason')
        record = self.pending()
        events = self.usable_chain(self.events(), record)
        chosen = self.select(record, ids)
        proposals = {e['entry_id']: e for e in events if e.get('action') == 'PROPOSED'}
        at = now()
        for entry in chosen:
            proposal = proposals.get(entry['entry_id'])
            if proposal is None or proposal['entry_sha256'] != entry_digest(entry):
                raise ValueError(f'{entry["entry_id"]} differs from the proposal the audit log recorded (edited after it was proposed)')
        for entry in chosen:
            entry.update({'status': action, 'decided_by': actor, 'decider_role': role, 'decided_at': at, 'decision_reason': str(reason).strip()})
            self.append(events, action, entry, actor, role)
        self.save_pending(record, events)
        return chosen

    def reject(self, ids, actor, reason, role='approver') -> list:
        return self.close(ids, actor, reason, role, 'REJECTED')

    def withdraw(self, ids, actor, reason, role='proposer') -> list:
        return self.close(ids, actor, reason, role, 'WITHDRAWN')

    # ------------------------------------------------------------------ checking
    def overlay_findings(self, lifecycle, reviewed_entries) -> tuple:
        """([problems], [warnings]) of laying the reviewed file over the dataset as scoring does (labels.apply_reviewed).

        An approved entry the overlay skips because the label is no longer its "from" is a person's decision that no
        longer counts: a problem, unless the dataset already holds its "to" (a later dataset version took the change
        in), which is a warning.
        """
        from .labels import apply_reviewed
        problems, warnings = [], []
        try:
            _, overlay = apply_reviewed(self.dataset, self.reviewed_path, self.dataset_path, audit_path=self.audit_path)
        except ValueError as exc:
            return [f'{self.reviewed_path.name} cannot be laid over {self.dataset_path.name}: {exc}'], []
        for skip in overlay['skipped']:
            entry_id = skip.get('entry_id')
            steps = lifecycle.get(entry_id) or []
            if 'label_now' not in skip or not steps or steps[-1].get('action') != 'APPROVED':
                continue
            wanted = (reviewed_entries.get(entry_id) or {}).get('to')
            if skip['label_now'] == wanted:
                warnings.append(f'{entry_id}: the dataset already has {wanted!r} (a later dataset version holds this approved change)')
            else:
                problems.append(f'{entry_id}: approved but not applied by the overlay ({skip["reason"]}); propose the change again from the '
                                'label as it is now')
        return problems, warnings

    def verify(self) -> dict:
        """Chain, head, lifecycle and file agreement; ``ok`` only when nothing is wrong."""
        events = self.events()
        problems = chain_problems(events)
        warnings = []
        record = None
        if self.pending_path.is_file():
            try:
                record = self.pending()
            except (ValueError, json.JSONDecodeError) as exc:
                problems.append(f'pending file unreadable: {exc}')
        elif events:
            problems.append(f'{self.audit_path.name} has events but {self.pending_path.name} is missing')
        if record is not None:
            head = record.get('audit') or {}
            actual = (events[-1].get('hash') if events else GENESIS, len(events))
            if (head.get('head'), head.get('count')) != actual:
                problems.append(f'the pending file records the log at {head.get("count")} events / head {str(head.get("head"))[:12]}, '
                                f'the log has {actual[1]} / {str(actual[0])[:12]} (lines cut, added or edited)')
        book = None
        if self.reviewed_path.is_file():
            try:
                book = self.reviewed()
            except (ValueError, json.JSONDecodeError) as exc:
                problems.append(f'reviewed file unreadable: {exc}')
        lifecycle = {}
        for event in events:
            if isinstance(event, dict) and 'entry_id' in event:
                lifecycle.setdefault(event['entry_id'], []).append(event)
        for entry_id, steps in lifecycle.items():
            actions = [s.get('action') for s in steps]
            if actions[0] != 'PROPOSED' or actions.count('PROPOSED') != 1:
                problems.append(f'{entry_id}: events {actions} (one PROPOSED must come first)')
            if sum(a in TERMINAL for a in actions) > 1 or (any(a in TERMINAL for a in actions[:-1])):
                problems.append(f'{entry_id}: events {actions} (nothing may follow a decision)')
        pending_entries = {e.get('entry_id'): e for e in (record or {}).get('entries') or []}
        reviewed_entries = {}
        unaudited = 0
        for entry in (book or {}).get('entries') or []:
            if entry.get('entry_id'):
                reviewed_entries[entry['entry_id']] = entry
            else:
                unaudited += 1
        if unaudited:
            warnings.append(f'{unaudited} reviewed entries carry no entry_id (typed by hand); a strict overlay ignores them')
        for entry_id, steps in lifecycle.items():
            last, proposed = steps[-1], steps[0]
            if last.get('action') == 'APPROVED':
                entry = reviewed_entries.get(entry_id)
                if entry is None:
                    problems.append(f'{entry_id}: approved in the log but missing from {self.reviewed_path.name}')
                    continue
                if entry_digest(entry) != last.get('entry_sha256'):
                    problems.append(f'{entry_id}: the reviewed entry differs from the one approved (edited after approval)')
                if entry.get('pending_sha256') != proposed.get('entry_sha256'):
                    problems.append(f'{entry_id}: pending_sha256 is not the digest of the proposal')
                if entry.get('status') != 'ACCEPTED':
                    problems.append(f'{entry_id}: approved in the log but {entry.get("status")} in the reviewed file')
                if entry_id in pending_entries:
                    problems.append(f'{entry_id}: approved but still in the pending file')
                if str(entry.get('approved_by')).strip().casefold() == str(entry.get('proposed_by')).strip().casefold() and not entry.get('self_approved'):
                    problems.append(f'{entry_id}: approved by its proposer without the self_approved mark')
                elif entry.get('self_approved'):
                    warnings.append(f'{entry_id}: approved by its own proposer (--allow-self-approval)')
                continue
            entry = pending_entries.get(entry_id)
            status = last.get('action') if last.get('action') in TERMINAL else 'PENDING'
            if entry is None:
                problems.append(f'{entry_id}: {status} in the log but missing from {self.pending_path.name}')
                continue
            if entry.get('status') != status:
                problems.append(f'{entry_id}: {status} in the log but {entry.get("status")} in the pending file')
            if entry_digest(entry) != last.get('entry_sha256'):
                problems.append(f'{entry_id}: the pending entry differs from the one the log recorded (edited)')
        for entry_id in pending_entries:
            if entry_id not in lifecycle:
                problems.append(f'{entry_id}: in the pending file without a PROPOSED event')
        for entry_id in reviewed_entries:
            if entry_id not in lifecycle or lifecycle[entry_id][-1].get('action') != 'APPROVED':
                problems.append(f'{entry_id}: ACCEPTED in the reviewed file without an APPROVED event')
        if book is not None:
            found, cautions = self.overlay_findings(lifecycle, reviewed_entries)
            problems, warnings = problems + found, warnings + cautions
        statuses = [e.get('status') for e in pending_entries.values()]
        return {'dataset_id': self.dataset_id, 'directory': str(self.directory), 'ok': not problems, 'problems': problems, 'warnings': warnings,
                'events': len(events), 'head': events[-1].get('hash') if events else GENESIS,
                'pending': statuses.count('PENDING'), 'rejected': statuses.count('REJECTED'), 'withdrawn': statuses.count('WITHDRAWN'),
                'approved': len(reviewed_entries), 'unaudited_reviewed': unaudited}


# ---------------------------------------------------------------------- imports (they only propose)

def load_review(path: Path):
    """The pilot Review of a review file: a regchain-pilot-review-v1 JSON or a reviewed packet (its last event)."""
    from regchain.pilot.schema import Review
    data = read_json(path)
    if isinstance(data, dict) and data.get('format') == 'regchain-pilot-review-v1':
        return Review.model_validate(data)
    events = data.get('events') if isinstance(data, dict) else None
    if events and isinstance(events[-1].get('payload'), dict) and events[-1]['payload'].get('kind') == 'conc-pilot-review-v1':
        return Review.model_validate(events[-1]['payload']['review'])
    raise ValueError(f'{path} is neither a pilot review (regchain-pilot-review-v1) nor a reviewed packet')


def run_context(run_dir: Path, dataset_path: Path):
    """(results re-paired with the dataset in memory, packets, manifest) of a run directory; nothing is written."""
    from .taxonomy import run_results
    results, _, packets, _ = run_results(Path(run_dir), dataset_path)
    manifest_path = Path(run_dir) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}
    return results, packets, manifest


def review_changes(review, result, payload, run_dir, manifest) -> tuple:
    """(changes, skipped) that a pilot review's OVERRIDE decisions imply for one case's labels."""
    from regchain.pilot.engine import AUTONOMOUS_REVIEWER
    changes, skipped = [], []
    if review.reviewer == AUTONOMOUS_REVIEWER:
        return [], [{'obligation_id': None, 'reason': 'an autonomous (AI) review is not a person\'s decision'}]
    predictions = {p['obligation_id']: p for p in result.get('predictions') or []}
    by_key = {item.get('prediction'): item for item in result.get('scored') or [] if item.get('matched')}
    obligations = {o.get('id'): o for o in (payload or {}).get('obligations') or []}
    for decision in review.decisions:
        if decision.action != 'OVERRIDE':
            continue
        prediction = predictions.get(decision.obligation_id)
        item = by_key.get((prediction or {}).get('key'))
        if item is None:
            skipped.append({'obligation_id': decision.obligation_id,
                            'reason': 'no expectation of the dataset matches this duty (an unexpected candidate or another run)'})
            continue
        expected = item['expected']
        wanted = {'applicability': decision.applicability}
        if decision.applicability == 'APPLIES':
            wanted['coverage'] = decision.coverage
            if expected.get('conflict') is not None:
                wanted['conflict'] = decision.coverage == 'CONFLICT'
        else:
            # A duty that does not apply (or whose applicability is open) has no coverage or conflict to assert: every
            # DOES_NOT_APPLY label of tr-aml-v1 has coverage ANY or NOT_ASSESSED and conflict null, every UNKNOWN one ANY /
            # null, and the sheet import (expert_export.sheet_changes, AI_CORRECT) proposes the same. Left as they were,
            # the corrected row would score its old COVERS_TEXT against the NOT_ASSESSED a DOES_NOT_APPLY prediction gets.
            if expected.get('coverage') not in (None, 'ANY'):
                wanted['coverage'] = 'NOT_ASSESSED' if decision.applicability == 'DOES_NOT_APPLY' else 'ANY'
            if expected.get('conflict') is not None:
                wanted['conflict'] = None
        obligation = obligations.get(decision.obligation_id) or {}
        evidence = {'run': Path(run_dir).name, 'manifest_sha256': manifest.get('manifest_sha256'), 'analysis_head': review.analysis_head,
                    'obligation_id': decision.obligation_id, 'obligation_key': prediction.get('key'), 'review_sha256': digest(review.model_dump()),
                    'ai_proposal': {'applicability': prediction.get('applicability_raw'), 'coverage': prediction.get('coverage')},
                    'regulation_quote': (obligation.get('candidate') or {}).get('source_quote')}
        reason = f'Pilot review OVERRIDE ({review.reviewer}, {review.reviewer_role}): {decision.override_reason.strip()} — {decision.rationale.strip()}'
        for field, value in wanted.items():
            if expected.get(field) != value:
                changes.append({'case_id': result['case_id'], 'article': expected['article'], 'clause': expected['clause'],
                                'action_keywords': expected.get('action_keywords') or [], 'field': field, 'to': value, 'reason': reason,
                                'evidence': evidence})
    return changes, skipped


def import_review(dataset_path, run_dir, review_path, directory=None) -> dict:
    """Propose, as PENDING, the label changes a pilot review's OVERRIDE decisions imply (proposer = the reviewer)."""
    ledger = Ledger(dataset_path, directory)
    review = load_review(Path(review_path))
    results, packets, manifest = run_context(Path(run_dir), ledger.dataset_path)
    result = next((r for r in results if r.get('head') == review.analysis_head), None)
    if result is None:
        raise ValueError(f'no case of {run_dir} has the analysis head {review.analysis_head[:12]} the review names')
    changes, skipped = review_changes(review, result, packets.get(result['case_id']), run_dir, manifest)
    entries = propose_each(ledger, changes, review.reviewer, review.reviewer_role, 'pilot-review', skipped)
    return {'case_id': result['case_id'], 'proposed': entries, 'skipped': skipped}


def propose_each(ledger, changes, actor, role, origin, skipped) -> list:
    """Propose the changes one by one: a change that cannot be proposed is skipped with its reason, the others go in."""
    entries = []
    for change in changes:
        try:
            entries.extend(ledger.propose([change], actor, role, origin))
        except ValueError as exc:
            skipped.append({'change': f'{change["case_id"]} md.{change["article"]}{change.get("clause") or ""} {change["field"]}', 'reason': str(exc)})
    return entries


def import_sheet(dataset_path, sheet_path, actor, role='expert reviewer', directory=None) -> dict:
    """Propose, as PENDING, the label changes of a filled expert-review sheet (expert_export.py): CSV, JSON or XLSX."""
    from .expert_export import read_sheet, sheet_changes
    ledger = Ledger(dataset_path, directory)
    rows = read_sheet(Path(sheet_path))
    changes, skipped, confirmed = [], [], 0
    for row in rows:
        found, note = sheet_changes(row)
        if note == 'LABEL_CORRECT':
            confirmed += 1
        elif note:
            skipped.append({'row': row.get('row_id'), 'reason': note})
        for change in found:
            change.setdefault('evidence', {}).update({'sheet': Path(sheet_path).name, 'sheet_sha256': file_sha256(sheet_path)})
        changes.extend(found)
    entries = propose_each(ledger, changes, actor, role, 'expert-review', skipped)
    return {'rows': len(rows), 'label_confirmed': confirmed, 'proposed': entries, 'skipped': skipped}


def run_evidence(run_dir, dataset_path, change) -> dict:
    """What a run said about the row a proposal changes (for the proposal's evidence)."""
    from .labels import run_outputs
    rows, _, _ = run_outputs(Path(run_dir), dataset_path)
    manifest_path = Path(run_dir) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}
    key = row_id(change['case_id'], change)
    row = rows.get(key) or next((r for k, r in rows.items() if k[:3] == key[:3]), None)
    evidence = {'run': Path(run_dir).name, 'manifest_sha256': manifest.get('manifest_sha256')}
    if row:
        evidence.update({'obligation_key': row.get('obligation_key'), 'provision': row.get('provision'),
                         'regulation_quote': ' '.join(str(row.get('regulation_text') or '').split())[:1500] or None,
                         'system_output': {k: row.get(k) for k in ('applicability', 'coverage', 'conflict')}})
    return evidence


# ---------------------------------------------------------------------- command line

def add_commands(commands):
    """The ``labels`` subcommand group of python -m regchain.evaluation."""
    labels = commands.add_parser('labels', help='Pending label changes: propose, list, approve (four-eyes), reject, withdraw, verify, '
                                                'import pilot or expert reviews. Never edits the dataset.')
    group = labels.add_subparsers(dest='labels_command', required=True)

    def command(name, text):
        parser = group.add_parser(name, help=text)
        parser.add_argument('--dataset', required=True, help='Dataset file (it is only read)')
        parser.add_argument('--dir', default=None, help='Directory of the pending, audit and reviewed files '
                                                        '(default: evaluation/reviewed_labels beside the dataset folder)')
        return parser

    propose = command('propose', 'Propose one label change (or a JSON list with --file) as PENDING')
    propose.add_argument('--case', default=None)
    propose.add_argument('--article', default=None)
    propose.add_argument('--clause', default='')
    propose.add_argument('--keywords', nargs='*', default=None, help='Action keywords naming one duty of the clause')
    propose.add_argument('--field', default=None, choices=list(EDITABLE))
    propose.add_argument('--to', default=None, help='New label (conflict/required: true|false; evidence: JSON list)')
    propose.add_argument('--from', dest='from_value', default=None, help='The label as it is now (checked; default: read from the dataset)')
    propose.add_argument('--reason', default=None)
    propose.add_argument('--actor', required=True, help='Who proposes (recorded in the audit log)')
    propose.add_argument('--role', default='proposer')
    propose.add_argument('--origin', default='manual', choices=list(ORIGINS))
    propose.add_argument('--evidence-json', default=None, help='JSON object recorded as the proposal evidence')
    propose.add_argument('--run', default=None, help='Run directory whose output for the row is added to the evidence')
    propose.add_argument('--file', default=None, help='JSON list of {case_id, article, clause, action_keywords, field, to, from, reason, evidence}')
    listing = command('pending', 'List the pending (or, with --all, every) proposal')
    listing.add_argument('--all', action='store_true')
    listing.add_argument('--json', action='store_true')
    approve = command('approve', 'Approve pending entries: they move to the reviewed file as ACCEPTED (HUMAN_REVIEWED)')
    approve.add_argument('--ids', nargs='+', required=True)
    approve.add_argument('--approver', required=True)
    approve.add_argument('--role', default='label approver')
    approve.add_argument('--note', default='')
    approve.add_argument('--allow-self-approval', action='store_true', help='Let the proposer approve (recorded as self_approved)')
    reject = command('reject', 'Reject pending entries (kept, with the reason, as REJECTED)')
    reject.add_argument('--ids', nargs='+', required=True)
    reject.add_argument('--approver', required=True)
    reject.add_argument('--role', default='label approver')
    reject.add_argument('--reason', required=True)
    withdraw = command('withdraw', 'Withdraw pending entries (kept, with the reason, as WITHDRAWN)')
    withdraw.add_argument('--ids', nargs='+', required=True)
    withdraw.add_argument('--actor', required=True)
    withdraw.add_argument('--role', default='proposer')
    withdraw.add_argument('--reason', required=True)
    command('verify', 'Check the audit chain and that the pending, audit and reviewed files agree')
    imports = command('import-review', 'Propose the changes of a pilot review (--run + --review) or of a filled expert-review sheet (--sheet)')
    imports.add_argument('--run', default=None, help='Run directory whose packet the pilot review reviewed')
    imports.add_argument('--review', default=None, help='Pilot review JSON (regchain-pilot-review-v1) or a reviewed packet')
    imports.add_argument('--sheet', default=None, help='Filled expert_review.csv / .json / .xlsx')
    imports.add_argument('--actor', default=None, help='With --sheet: the reviewer who filled it')
    imports.add_argument('--role', default='expert reviewer')


def entry_line(entry) -> str:
    where = f'{entry["case_id"]} md.{entry["article"]}' + (f'({clause_id(entry["clause"])})' if clause_id(entry.get('clause') or '') else '')
    return (f'{entry["entry_id"]}  {entry.get("status", "ACCEPTED"):9}  {where}  {entry["field"]}: {entry["from"]!r} -> {entry["to"]!r}  '
            f'by {entry.get("proposed_by")} ({entry.get("origin")}, {str(entry.get("proposed_at"))[:10]})\n    {entry.get("reason", "")[:300]}')


def run_command(args, parser) -> int:
    """Run ``labels <command>``; a refused step prints why and returns 1 (no file is written by a refused step)."""
    import sys
    try:
        ledger = Ledger(Path(args.dataset), args.dir)
        command = args.labels_command
        if command == 'propose':
            if args.file:
                changes = json.loads(Path(args.file).read_text(encoding='utf-8-sig'))
                if not isinstance(changes, list):
                    raise ValueError('--file must hold a JSON list of changes')
            else:
                missing = [flag for flag, value in (('--case', args.case), ('--article', args.article), ('--field', args.field),
                                                    ('--to', args.to), ('--reason', args.reason)) if value is None]
                if missing:
                    parser.error('labels propose needs ' + ', '.join(missing) + ' (or --file)')
                change = {'case_id': args.case, 'article': args.article, 'clause': args.clause, 'action_keywords': args.keywords or None,
                          'field': args.field, 'to': args.to, 'reason': args.reason,
                          'evidence': json.loads(args.evidence_json) if args.evidence_json else {}}
                if args.from_value is not None:
                    change['from'] = args.from_value
                changes = [change]
            if args.run:
                for change in changes:
                    change['evidence'] = {**run_evidence(Path(args.run), ledger.dataset_path, {**change, 'clause': change.get('clause') or ''}),
                                          **(change.get('evidence') or {})}
            entries = ledger.propose(changes, args.actor, args.role, args.origin)
            print(json.dumps({'proposed': [e['entry_id'] for e in entries], 'status': 'PENDING', 'pending_file': str(ledger.pending_path)},
                             ensure_ascii=False))
            return 0
        if command == 'pending':
            entries = [e for e in ledger.pending()['entries'] if args.all or e.get('status') == 'PENDING']
            if args.json:
                print(json.dumps(entries, ensure_ascii=False, indent=1))
            else:
                print(f'{len(entries)} {"entries" if args.all else "pending entries"} for {ledger.dataset_id} ({ledger.pending_path})')
                for entry in entries:
                    print(entry_line(entry))
            return 0
        if command == 'approve':
            moved = ledger.approve(args.ids, args.approver, args.role, args.allow_self_approval, args.note)
            print(json.dumps({'approved': [e['entry_id'] for e in moved], 'reviewed_file': str(ledger.reviewed_path)}, ensure_ascii=False))
            return 0
        if command in ('reject', 'withdraw'):
            actor = args.approver if command == 'reject' else args.actor
            closed = ledger.close(args.ids, actor, args.reason, args.role, 'REJECTED' if command == 'reject' else 'WITHDRAWN')
            print(json.dumps({command + 'ed': [e['entry_id'] for e in closed]}, ensure_ascii=False))
            return 0
        if command == 'verify':
            report = ledger.verify()
            print(json.dumps(report, ensure_ascii=False, indent=1))
            return 0 if report['ok'] else 1
        if command == 'import-review':
            if args.sheet:
                if not args.actor:
                    parser.error('labels import-review --sheet needs --actor (the reviewer who filled the sheet)')
                outcome = import_sheet(ledger.dataset_path, Path(args.sheet), args.actor, args.role, args.dir)
            elif args.run and args.review:
                outcome = import_review(ledger.dataset_path, Path(args.run), Path(args.review), args.dir)
            else:
                parser.error('labels import-review needs --run and --review, or --sheet and --actor')
            print(json.dumps({**{k: v for k, v in outcome.items() if k != 'proposed'}, 'proposed': [e['entry_id'] for e in outcome['proposed']],
                              'status': 'PENDING'}, ensure_ascii=False))
            return 0
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f'labels {args.labels_command} refused: {exc}', file=sys.stderr)
        return 1
    return 2
