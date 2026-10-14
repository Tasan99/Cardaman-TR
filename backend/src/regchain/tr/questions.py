"""Profile questions: a missing company fact is asked once, the answer is applied with its basis, and only the
evaluations it affects run again.

An applicability review of type COMPLETE_PROFILE_FACT names a gate the profile cannot decide (an activity, an entity
class, a licence, a sales channel, a product's packaging tag). Many duties wait on the same fact of the same entity: one
question per (target, gate) gathers them - which of the values the waiting duties ask about does the target have, and is
the list now complete? The answer is applied to a copy of the profile (the pilot profile file is never changed): the
values named are added, the dimension is stated complete or left open, and the answer's author, time and basis are kept
in `facts_basis`. Then every duty is routed again on the targets of the answered entities only; a decision that changed
is assessed against the register, and the delta names each change with its gates.

Nothing is inferred: a value the answer does not name is absent only when the answer states the list complete.
"""
from datetime import datetime

from pydantic import Field

from ..pilot.schema import Strict
from .adjudicate import assess_obligation
from .compare import Register, applicability_review

# gate -> (target kind, list field, completeness field, vocabulary kind)
GATES = {
    'ACTIVITY_CLASS': ('LEGAL_ENTITY', 'activity_classes', 'activities_complete', 'activity_classes'),
    'ENTITY_CLASS': ('LEGAL_ENTITY', 'entity_classes', 'entity_classes_complete', 'entity_classes'),
    'SALES_CHANNEL': ('LEGAL_ENTITY', 'sales_channels', 'sales_channels_complete', 'sales_channels'),
    'PRODUCT_ATTRIBUTE': ('PRODUCT', 'tags', 'tags_complete', 'product_tags'),
}
FIELD_TR = {'activity_classes': 'faaliyet', 'entity_classes': 'şirket türü', 'sales_channels': 'satış kanalı', 'tags': 'ürün niteliği'}


class AnswerError(ValueError):
    pass


class Question(Strict):
    question_id: str
    target_level: str
    target_id: str
    target_name: str
    gate: str
    field: str
    complete_field: str
    vocabulary: str
    stated: list[str]
    stated_complete: bool
    asked: list[dict]                       # [{'value', 'label'}]: what the waiting duties ask about and the profile does not state
    stated_labels: list[str] = []           # the stated values as the company reads them
    blocked_reviews: list[str]
    prompt_tr: str


class Answer(Strict):
    question_id: str
    present: list[str] = []                 # the values the target has (among those asked, or any of the vocabulary)
    list_complete: bool                     # the list, with these values, is complete
    answered_by: str = Field(min_length=1)
    answered_at: str = Field(min_length=1)
    basis: str = Field(min_length=1)        # where the fact comes from (a document, a person, a record)


def _target(profile, level: str, target_id: str):
    return profile.entity(target_id) if level == 'LEGAL_ENTITY' else profile.product(target_id)


def profile_questions(profile, reviews, vocabulary) -> list[Question]:
    """One question per (target, gate) that blocks at least one review of a missing company fact."""
    waiting: dict[tuple, list] = {}
    for review in reviews:
        if review.review_type not in ('COMPLETE_PROFILE_FACT', 'RESOLVE_PROFILE_CONFLICT'):
            continue
        for fact in review.missing_facts:
            gate = fact.get('gate')
            if gate == 'PRODUCT_SCOPE' and review.level == 'PRODUCT' and review.source_evidence.get('scope', {}).get('product_attributes'):
                gate, fact = 'PRODUCT_ATTRIBUTE', {**fact, 'required': review.source_evidence['scope']['product_attributes']}
            if gate not in GATES or GATES[gate][0] != ('PRODUCT' if review.level == 'PRODUCT' else 'LEGAL_ENTITY'):
                continue
            target = review.target_id if review.level in ('LEGAL_ENTITY', 'PRODUCT') else review.entity_id
            if target is None:
                continue
            level = 'PRODUCT' if review.level == 'PRODUCT' else 'LEGAL_ENTITY'
            waiting.setdefault((level, target, gate), []).append((review, fact))
    out = []
    for (level, target_id, gate), items in sorted(waiting.items()):
        _, field, complete_field, kind = GATES[gate]
        target = _target(profile, level, target_id)
        stated = list(getattr(target, field))
        complete = target.complete(field) if hasattr(target, 'complete') and field in ('activity_classes', 'entity_classes') else bool(getattr(target, complete_field))
        wanted = sorted({v for _, fact in items for v in fact.get('required') or []} - set(stated))
        asked = [{'value': v, 'label': vocabulary.label(kind, v) if v in getattr(vocabulary, kind) else v} for v in wanted]
        reviews = sorted({review.review_id for review, _ in items})
        name = getattr(target, 'name', target_id)
        what = FIELD_TR[field]
        shown = [vocabulary.label(kind, v) if v in getattr(vocabulary, kind) else v for v in stated]
        prompt = (f"{name} ({target_id}): {what} bilgisi eksik. Profilde: {', '.join(shown) or '—'} "
                  f"({'liste tam' if complete else 'liste tam değil'}). Bu bilgi {len(reviews)} değerlendirmeyi bekletiyor. "
                  f"Aşağıdakilerden hangileri var ve liste bu cevapla tam mı?")
        labels = [vocabulary.label(kind, v) if v in getattr(vocabulary, kind) else v for v in stated]
        out.append(Question(question_id=f'{level}:{target_id}:{gate}', target_level=level, target_id=target_id, target_name=name, gate=gate,
                            field=field, complete_field=complete_field, vocabulary=kind, stated=stated, stated_complete=complete,
                            asked=asked, stated_labels=labels, blocked_reviews=reviews, prompt_tr=prompt))
    return out


def apply_answers(profile, answers: list[Answer], questions: list[Question], vocabulary):
    """(a new profile with the answers applied, what was applied). The given profile is not changed."""
    by_id = {x.question_id: x for x in questions}
    entities = {e.entity_id: e for e in profile.legal_entities}
    products = {p.product_id: p for p in profile.products}
    applied = []
    for answer in answers:
        question = by_id.get(answer.question_id)
        if question is None:
            raise AnswerError(f'no question {answer.question_id}')
        known = getattr(vocabulary, question.vocabulary)
        unknown = [v for v in answer.present if v not in known]
        if unknown:
            raise AnswerError(f'{answer.question_id}: {unknown} are not {question.vocabulary} of the vocabulary')
        pool = entities if question.target_level == 'LEGAL_ENTITY' else products
        if question.target_id not in pool:
            raise AnswerError(f'{answer.question_id}: {question.target_id} is not in profile {profile.profile_id}; '
                              f'the questions were written for another profile')
        target = pool[question.target_id]
        values = list(dict.fromkeys([*getattr(target, question.field), *answer.present]))
        basis = f'{answer.answered_by} {answer.answered_at}: {answer.basis}'
        update = {question.field: values}
        if answer.list_complete:
            update[question.complete_field] = True
        if question.target_level == 'LEGAL_ENTITY':
            update['facts_basis'] = {**target.facts_basis, question.field: basis}
        pool[question.target_id] = target.model_copy(update=update)
        applied.append({'question_id': answer.question_id, 'target_id': question.target_id, 'field': question.field,
                        'added': [v for v in answer.present if v not in getattr(target, question.field)],
                        'list_complete': answer.list_complete, 'basis': basis, 'blocked_reviews': len(question.blocked_reviews)})
    new = profile.model_copy(update={'legal_entities': [entities[e.entity_id] for e in profile.legal_entities],
                                     'products': [products[p.product_id] for p in profile.products]})
    type(profile).model_validate(new.model_dump())          # the answered profile is a valid profile
    return new, applied


def _entity_of(decision) -> str | None:
    return decision.entity_id or (decision.target_id if decision.level == 'LEGAL_ENTITY' else None)


def reassess(old, new, obligations, register: Register, registry, store, table=None, adjudicator=None) -> dict:
    """Route every duty again on the targets of the entities (and products) the answers changed; assess what now applies.
    {'affected_entities', 'affected_products', 'evaluated', 'changes', 'rows', 'reviews'}."""
    from .extraction import route
    entities = sorted(e.entity_id for e, f in zip(new.legal_entities, old.legal_entities) if e != f)
    products = sorted(p.product_id for p, f in zip(new.products, old.products) if p != f)
    holders = {e.entity_id for e in new.legal_entities if set(e.product_ids) & set(products)}
    touched = lambda d: _entity_of(d) in entities or d.target_id in products or (d.level == 'PRODUCT' and d.target_id in products)
    evaluated, changes, rows, reviews = [], [], [], []
    for obligation in obligations:
        before = {d.target_id: d for d in route(obligation, old, registry, store)[0] if touched(d)}
        if not before:
            continue
        after = {d.target_id: d for d in route(obligation, new, registry, store)[0] if d.target_id in before}
        for target_id, decision in after.items():
            evaluated.append(f'{obligation.obligation_id}:{target_id}')
            previous = before[target_id]
            if decision.status == 'UNKNOWN':
                reviews.append(applicability_review(obligation, decision).model_dump(mode='json'))
            if decision.status == previous.status and decision.reason_codes == previous.reason_codes:
                continue
            change = {'review_id': f'{obligation.obligation_id}:{target_id}', 'obligation_id': obligation.obligation_id,
                      'provision_ref': obligation.provision_ref, 'target_id': target_id, 'entity_id': _entity_of(decision),
                      'before': previous.status, 'after': decision.status, 'reasons_before': previous.reason_codes,
                      'reasons_after': decision.reason_codes, 'gates_after': decision.gates}
            if decision.status in ('APPLIES', 'PARTIAL'):
                row, assessment = assess_obligation(obligation, decision, new, register, registry, table, adjudicator, store=store)
                rows.append({'review_id': change['review_id'], 'target_id': target_id, 'provision_ref': obligation.provision_ref,
                             'mapping': row.mapping.status, 'coverage': row.document_coverage, 'decision': assessment.decision,
                             'review_reasons': assessment.review_reasons})
                change['assessment'] = rows[-1]
            changes.append(change)
    return {'affected_entities': entities, 'affected_products': products, 'product_holders': sorted(holders), 'evaluated': evaluated,
            'changes': changes, 'rows': rows, 'reviews': reviews,
            'summary': {'evaluated': len(evaluated), 'status_changed': sum(c['before'] != c['after'] for c in changes),
                        'reasons_changed_only': sum(c['before'] == c['after'] for c in changes),
                        'moves': _count(f"{c['before']} -> {c['after']}" for c in changes if c['before'] != c['after']),
                        'new_rows': len(rows), 'still_unknown': len(reviews)}}


def _count(values) -> dict:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


# The questionnaire a company fills (CSV, UTF-8 with BOM, ';'): one line per value asked (E/H), and per question one
# line for whether the list is complete, with who answered, the basis and the date. A question left blank is not answered.
# It is read back as Excel saves it on a Turkish Windows: UTF-8 (with or without a BOM), UTF-16 with a BOM ('Unicode metin'),
# or Windows-1254 ('CSV (virgülle ayrılmış)'), with ';', ',' or TAB between the cells.
QUESTIONNAIRE_COLUMNS = ['question_id', 'satir', 'sirket_veya_urun', 'soru', 'deger_kodu', 'deger', 'cevap', 'cevaplayan', 'dayanak', 'tarih']
REQUIRED_COLUMNS = ['question_id', 'satir', 'deger_kodu', 'cevap', 'cevaplayan', 'dayanak']
YES, NO = {'E', 'EVET', 'Y', 'YES', 'X', '1'}, {'H', 'HAYIR', 'N', 'NO', '0'}
SAVE_AS = 'save it in Excel as "CSV UTF-8 (virgülle ayrılmış)"'


def _mark(cell) -> str:
    """An E/H cell as typed, spaces, case and the Turkish dotted or dotless i aside: 'e', 'Evet', 'hayır', 'HAYİR' -> E, EVET, HAYIR."""
    return (cell or '').strip().replace('İ', 'I').replace('ı', 'I').upper().replace('\u0307', '')


def _questionnaire_text(path) -> str:
    """The text of a saved questionnaire. Without a BOM, bytes that are not UTF-8 are read as Windows-1254 (strictly: a byte
    that code page does not define is refused). A workbook, or text in any other encoding, is refused saying how to save it."""
    import codecs
    from pathlib import Path
    raw = Path(path).read_bytes()
    if raw.startswith((b'PK\x03\x04', b'\xd0\xcf\x11\xe0')):
        raise AnswerError(f'{path} is an Excel workbook, not a questionnaire file; {SAVE_AS}')
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        encodings = ['utf-16']
    elif raw.startswith(codecs.BOM_UTF8):
        encodings = ['utf-8-sig']
    else:
        encodings = ['utf-8', 'cp1254']
    for encoding in encodings:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise AnswerError(f'{path} is not UTF-8, UTF-16 or Windows-1254 text; {SAVE_AS}')


def load_list(path, model, what: str) -> list:
    """A JSON list of `model` records (questions.json, an answers file), UTF-8 with or without a BOM. Anything else is refused
    naming the file."""
    import json
    from pathlib import Path
    from pydantic import ValidationError
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except ValueError as exc:                                                   # not UTF-8, not JSON
        raise AnswerError(f'{path} is not a JSON list of {what} ({exc})') from None
    if not isinstance(data, list):
        raise AnswerError(f'{path} is not a JSON list of {what}')
    out = []
    for number, item in enumerate(data):
        try:
            out.append(model.model_validate(item))
        except ValidationError as exc:
            error = exc.errors()[0]
            raise AnswerError(f"{path} is not a JSON list of {what} (item {number}: {'.'.join(map(str, error['loc']))}: {error['msg']})") from None
    return out


def write_questionnaire(questions: list[Question], path) -> int:
    """Write the questionnaire; returns the number of lines to fill."""
    import csv
    from pathlib import Path
    lines = []
    for question in questions:
        what = FIELD_TR[question.field]
        for item in question.asked:
            lines.append({'question_id': question.question_id, 'satir': 'DEGER', 'sirket_veya_urun': f'{question.target_name} ({question.target_id})',
                          'soru': f'Bu {what} var mı? (E/H)', 'deger_kodu': item['value'], 'deger': item['label'], 'cevap': '',
                          'cevaplayan': '', 'dayanak': '', 'tarih': ''})
        lines.append({'question_id': question.question_id, 'satir': 'LISTE_TAM', 'sirket_veya_urun': f'{question.target_name} ({question.target_id})',
                      'soru': (f"Profildeki {what} listesi ({', '.join(question.stated_labels or question.stated) or '—'}) ile yukarıda E dedikleriniz tam liste mi? (E/H) "
                               f"— {len(question.blocked_reviews)} değerlendirme bu cevabı bekliyor"),
                      'deger_kodu': '', 'deger': '', 'cevap': '', 'cevaplayan': '', 'dayanak': '', 'tarih': ''})
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=QUESTIONNAIRE_COLUMNS, delimiter=';')
        writer.writeheader()
        writer.writerows(lines)
    return len(lines)


def read_questionnaire(path, questions: list[Question]) -> list[Answer]:
    """The answers of a filled questionnaire, as Excel or another program saved it. A question with no value answered and
    no completeness answer is skipped; an answered question needs who answered and the basis. Refused, with their lines: a
    header without the columns read, a cell other than E/H (Evet/Hayır in any case), a line other than DEGER/LISTE_TAM,
    a value answered both E and H, two different completeness answers of one question. A line copied as it is is read once;
    a line with every cell blank is skipped."""
    import csv
    import io
    by_id = {q.question_id: q for q in questions}
    text = _questionnaire_text(path)
    header = text.split('\n', 1)[0]
    reader = csv.DictReader(io.StringIO(text, newline=''), delimiter=max((';', ',', '\t'), key=header.count))
    found = [(name or '').strip().lower() for name in reader.fieldnames or []]
    missing = [c for c in REQUIRED_COLUMNS if c not in found]
    if missing:
        raise AnswerError(f"{path}: missing columns: {', '.join(missing)} (found: {', '.join(found)}); "
                          f"the questionnaire has {';'.join(QUESTIONNAIRE_COLUMNS)}")
    reader.fieldnames = found
    grouped: dict[str, list[tuple[int, str, str, dict]]] = {}
    for number, row in enumerate(reader, 2):
        if not any((v or '').strip() for v in [*(v for k, v in row.items() if k is not None), *(row.get(None) or [])]):
            continue                                                            # only separators: a line Excel left behind
        question_id = (row.get('question_id') or '').strip()
        if question_id not in by_id:
            raise AnswerError(f'line {number}: no question {question_id!r}')
        kind = (row.get('satir') or '').strip().upper()
        if kind not in ('DEGER', 'LISTE_TAM'):
            raise AnswerError(f'line {number}: satir {row.get("satir")!r} is neither DEGER nor LISTE_TAM')
        cell = _mark(row.get('cevap'))
        if cell and cell not in YES | NO:
            raise AnswerError(f'line {number}: answer {row.get("cevap")!r} is not E or H')
        grouped.setdefault(question_id, []).append((number, kind, cell, row))
    read, conflicts = {}, []
    for question_id, lines in grouped.items():
        values: dict[str, tuple[int, bool]] = {}        # value code -> (the first line answering it, E)
        completeness: dict[tuple, int] = {}             # (E/H/None, who, basis, date) of a LISTE_TAM line -> its first line
        for number, kind, cell, row in lines:
            if kind == 'DEGER' and cell:
                code = (row.get('deger_kodu') or '').strip()
                first, yes = values.setdefault(code, (number, cell in YES))
                if yes != (cell in YES):
                    conflicts.append(f"{question_id} {code}: line {first} {'E' if yes else 'H'}, line {number} {'H' if yes else 'E'}")
            elif kind == 'LISTE_TAM':
                given = (True if cell in YES else False if cell in NO else None,
                         *((row.get(column) or '').strip() for column in ('cevaplayan', 'dayanak', 'tarih')))
                if given != (None, '', '', ''):
                    completeness.setdefault(given, number)
        if len(completeness) > 1:
            conflicts.append(f'{question_id} LISTE_TAM: ' + ', '.join(
                f"line {number} ({'E' if a[0] else 'H' if a[0] is False else '-'}; {'; '.join(a[1:])})" for a, number in completeness.items()))
        read[question_id] = (values, next(iter(completeness), (None, '', '', '')))
    if conflicts:                                       # never resolved by line order: the company says which line holds
        raise AnswerError('contradictory lines (keep one line per value and one LISTE_TAM line per question): ' + ' | '.join(conflicts))
    answers = []
    for question_id, (values, (complete, who, basis, when)) in read.items():
        if not values and complete is None:
            continue
        if not who or not basis:
            raise AnswerError(f'{question_id}: answered without who answered (cevaplayan) or the basis (dayanak)')
        answers.append(Answer(question_id=question_id, present=[code for code, (_, yes) in values.items() if yes], list_complete=bool(complete),
                              answered_by=who, answered_at=when or datetime.now().astimezone().isoformat(timespec='seconds'), basis=basis))
    return answers


def ask_interactively(questions: list[Question], input_fn=input, output=None) -> list[Answer]:
    """Ask the questions in the terminal: who answers (once), then for each question the numbers of the values present
    (empty for none), whether the list is complete, and the basis. 's' skips a question."""
    import sys
    output = output or sys.stdout
    answered_by = input_fn('Cevaplayan (ad / rol): ').strip()
    answers = []
    for index, question in enumerate(questions, 1):
        output.write(f'\n[{index}/{len(questions)}] {question.prompt_tr}\n')
        for number, item in enumerate(question.asked, 1):
            output.write(f'   {number}) {item["label"]} ({item["value"]})\n')
        picked = input_fn("Var olanların numaraları (virgülle; yoksa boş; atlamak için 's'): ").strip()
        if picked.lower() == 's':
            continue
        try:
            present = [question.asked[int(n) - 1]['value'] for n in picked.replace(' ', '').split(',') if n]
        except (ValueError, IndexError):
            output.write('   geçersiz numara; soru atlandı\n')
            continue
        complete = input_fn('Liste bu cevapla tam mı? (e/h): ').strip().lower() in ('e', 'evet', 'y', 'yes')
        basis = input_fn('Dayanak (belge / kişi / kayıt): ').strip() or 'belirtilmedi'
        answers.append(Answer(question_id=question.question_id, present=present, list_complete=complete, answered_by=answered_by,
                              answered_at=datetime.now().astimezone().isoformat(timespec='seconds'), basis=basis))
    return answers
