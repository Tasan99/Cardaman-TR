"""Change impact (v0.17): what a new regulation snapshot changed, and what that means for the
company's policy package, without re-judging what did not change.

Two snapshots of the same regulation are compared provision by provision (change_for in
sources.py). For a changed provision the kind of change is read from the text by rule:
a threshold (an amount), a deadline (a duration), the entity scope (who is named), an
exemption (exclusion wording), or plain wording. Provisions whose text did not change, when
the company profile, the policy package and the prompts are the same as in the compared
analysis, keep their previous rows (marked as carried forward) and cost no model call.
"""
import re

from .entities import COUNTERPARTIES, OBLIGED, found

# An amount with its currency on either side. EU texts put the code first ("EUR 15,000"), which
# v0.17 did not read; a prefixed amount ends on a digit, so "£10,000." and "£10,000 or" compare equal.
CURRENCY = re.compile(r'\d[\d.,]*\s*(?:TL|₺|Türk Lirası|EUR|USD|GBP|£|€|\$)'
                      r'|(?:£|€|\$|\b(?:TL|EUR|USD|GBP)\b)\s*\d(?:[\d.,]*\d)?', re.I)
# The Handbook writes most periods in words ("five years", "three months"); v0.17 knew only the Turkish ones.
NUMBER_WORDS = (r'bir|iki|üç|dört|beş|altı|yedi|sekiz|dokuz|on|onbeş|yirmi|otuz|kırk|elli|altmış|yetmiş|seksen|doksan|yüz'
                r'|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen'
                r'|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety')
# A number in words is a run of words ("on beş gün", "yüz seksen gün") and stands apart from its
# unit: glued on, "on"+"ay" is "onay" (approval) and "yüz"+"yıl" a century. Turkish attaches a
# suffix to the unit ("altı aylık", "beş yıldan", "üç ayda bir", "on iş gününde"). The suffix is
# only looked ahead at, so "30 gün" and "30 günlük" read as the same period, and "5 ayrı" is no period.
DURATION = re.compile(r'\b(?:\d+\s*|(?:' + NUMBER_WORDS + r')(?:\s+(?:' + NUMBER_WORDS + r'))*\s+)'
                      r'(?:iş\s*gün|gün|ay|yıl|hafta|saat|business days?|working days?|days?|months?|years?|weeks?|hours?)'
                      r'(?=(?:l[ıiuü]k|[ıiuü]n(?:[dt][ae]n?|[ae])?|[dt][ae]n?|y?[ıiuüae])?(?![a-zçğıöşü]))', re.I)
# Inflected forms count ("exemption", "exempted", "muafiyet", "istisnası"); "exceptional" does not.
EXCLUSION_WORDS = re.compile(r'(?<![a-zçğıöşü])(?:uygulanmaz|hariç|istisna[a-zçğıöşü]*|saklıdır|kapsam dışı|kapsamaz|muaf[a-zçğıöşü]*|does not apply'
                             r'|unless|except(?:ions?)?|other than|exempt(?:ions?|ed|s)?)(?![a-zçğıöşü])', re.I)
KINDS = ('THRESHOLD_CHANGED', 'DEADLINE_CHANGED', 'ENTITY_SCOPE_CHANGED', 'EXEMPTION_CHANGED', 'TEXT_CHANGED')
ACTIONS = {
    'THRESHOLD_CHANGED': 'Eşik veya tutar değişti: policy ve prosedürlerdeki tutar sınırlarını yeni metinle karşılaştır, kontrol kayıtlarındaki limitleri güncelle.',
    'DEADLINE_CHANGED': 'Süre değişti: policy\'deki bildirim, saklama ve tamamlama sürelerini yeni metinle karşılaştır; kontrol takvimini güncelle.',
    'ENTITY_SCOPE_CHANGED': 'Kapsam veya muhatap türü değişti: bu bendin şirkete uygulanabilirliğini yeniden değerlendir; entity gate sonucu değişmiş olabilir.',
    'EXEMPTION_CHANGED': 'İstisna değişti: daha önce muaf sayılan durumları ve buna dayanan policy hükümlerini yeniden gözden geçir.',
    'TEXT_CHANGED': 'Metin değişti: bu yükümlülüğe bağlanan policy pasajlarını yeni metne karşı yeniden incele.'}


def normalized(text: str) -> str:
    return ' '.join((text or '').split())


def change_kinds(old_text: str, new_text: str) -> list[str]:
    """Frozen v0.18 five-class comparison (retention is historically DEADLINE_CHANGED).

    New structured production fields use financial.analyze_financial_change; keep
    this contract intact for existing consumers and unchanged historical labels.
    """
    old, new = normalized(old_text), normalized(new_text)
    if old == new:
        return []
    kinds = []
    if set(CURRENCY.findall(old)) != set(CURRENCY.findall(new)):
        kinds.append('THRESHOLD_CHANGED')
    if set(m.lower() for m in DURATION.findall(old)) != set(m.lower() for m in DURATION.findall(new)):
        kinds.append('DEADLINE_CHANGED')
    entities = lambda text: {kind for kind, _ in found(COUNTERPARTIES, text)} | {kind for kind, _ in found(OBLIGED, text) if kind != 'GENERIC'}
    if entities(old) != entities(new):
        kinds.append('ENTITY_SCOPE_CHANGED')
    if sorted(m.lower() for m in EXCLUSION_WORDS.findall(old)) != sorted(m.lower() for m in EXCLUSION_WORDS.findall(new)):
        kinds.append('EXEMPTION_CHANGED')
    return kinds or ['TEXT_CHANGED']


def suggested_action(kinds) -> str:
    return ' '.join(ACTIONS[kind] for kind in KINDS if kind in kinds) or ACTIONS['TEXT_CHANGED']


def reuse_blockers(previous_payload, company_hash, policies, extraction_prompt_hash, pilot_prompt_hash):
    """Why unchanged provisions cannot simply keep their previous rows ([] when they can)."""
    reasons = []
    if previous_payload is None:
        return ['no previous analysis']
    if previous_payload.get('company_hash') != company_hash:
        reasons.append('company profile changed')
    if sorted(p['raw_hash'] for p in previous_payload.get('policies', [])) != sorted(p['raw_hash'] for p in policies):
        reasons.append('policy package changed')
    if previous_payload.get('extraction_prompt_hash') != extraction_prompt_hash or previous_payload.get('pilot_prompt_hash') != pilot_prompt_hash:
        reasons.append('prompts or judgement version changed')
    return reasons


def remap_source_ids(row: dict, mapping: dict) -> dict:
    """A carried-forward row with its regulation source ids rewritten to the new snapshot's ids."""
    swap = lambda value: mapping.get(value, value)
    copy = dict(row)
    copy['source_id'] = swap(row['source_id'])
    proposal = dict(row['proposal'])
    proposal['scope_evidence'] = [{**q, 'source_id': swap(q['source_id'])} for q in proposal.get('scope_evidence', [])]
    proposal['basis'] = [{**b, 'source_id': swap(b['source_id'])} for b in proposal.get('basis', [])]
    proposal['provenance'] = [{**p, 'source_id': swap(p['source_id']), 'passage_id': swap(p.get('passage_id'))} if p.get('kind') == 'regulation' else p
                              for p in proposal.get('provenance', [])]
    copy['proposal'] = proposal
    return copy


def impact_summary(previous_payload, previous_head, cases, rows, carried, reanalysed, blockers, current_labels):
    """The screen's 'this update may affect you' block, built from the cases' change records."""
    # Local import avoids a cycle: financial reuses the explicitly legacy classifier.
    from regchain.extraction.quantities import evidence_safe
    from .financial import analyze_financial_change, normalize_financial_impact
    previous_rows = {}
    for row in (previous_payload or {}).get('obligations', []):
        previous_rows.setdefault(row['source_label'], []).append(row)
    policies = {c['source_id']: p['filename'] for p in (previous_payload or {}).get('policies', []) for c in p['chunks']}
    changed, new = [], []
    for case in cases:
        change = case.get('change') or {}
        label = case['source']['printed_label']
        if change.get('status') == 'TEXT_CHANGED_REVIEW_REQUIRED':
            # change_for compares the raw text: a spacing-only edit is read again, and says so
            # rather than showing a change with no kind.
            kinds = change_kinds(change['old']['text'], case['source']['text']) or ['TEXT_CHANGED']
            affected = sorted({policies[q['source_id']] for row in previous_rows.get(label, []) for q in row['proposal'].get('policy_evidence', [])
                               if q['source_id'] in policies})
            changed.append({'label': label, 'kinds': kinds, 'old_text': change['old']['text'], 'new_text': case['source']['text'],
                            'affected_policies': affected,
                            'previous_coverage': [row['proposal']['coverage'] for row in previous_rows.get(label, [])],
                            'suggested_action': suggested_action(kinds),
                            'kinds_schema_version': 'v018-five-class-rule',
                            'structured_change': evidence_safe(analyze_financial_change(change['old']['text'], case['source']['text']).model_dump()),
                            # This path has no company financial facts: numeric impact is null.
                            'financial_impact': evidence_safe(normalize_financial_impact({}, old_text=change['old']['text'], new_text=case['source']['text'])),
                            'numeric_evidence_encoding': 'regchain-evidence-v1: non-integer numeric values encoded as decimal strings'})
        elif change.get('status') == 'NEW_IN_SNAPSHOT':
            new.append(label)
    deleted = sorted(set(previous_rows) - set(current_labels)) if previous_payload else []
    return {'previous_head': previous_head, 'reused_unchanged': not blockers, 'reuse_blocked_reason': '; '.join(blockers) if blockers else None,
            'carried_forward': carried, 'reanalysed': reanalysed, 'changed': changed, 'new': new, 'deleted_or_unselected': deleted,
            'affects_company': bool(changed or new)}
