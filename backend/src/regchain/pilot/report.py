"""Offline HTML review screen. All document content is escaped, no remote assets."""
import html
import json
from collections import Counter
from .semantic import judgeable


def esc(value):
    return html.escape(str(value), quote=True)


def label(value):
    # The backend state is printed beside the words: "uygulanıyor olabilir" and APPLIES used to
    # read as two different things.
    return {'APPLIES': 'Uygulanıyor — AI önerisi (APPLIES)', 'POSSIBLY_APPLIES': 'Olası uygulanabilir: özde eşleşiyor, eksik bilgi var (POSSIBLY_APPLIES)',
        'DOES_NOT_APPLY': 'Uygulanmıyor — AI önerisi (DOES_NOT_APPLY)',
        'UNKNOWN': 'Belirsiz / inceleme gerekli (UNKNOWN)', 'COVERS_TEXT': 'Yazılı kapsam desteği bulundu (COVERS_TEXT)',
        'PARTIAL': 'Kısmi kapsam (PARTIAL)', 'CONFLICT': 'Policy çelişkisi (CONFLICT)', 'NO_EVIDENCE': 'Kanıt bulunamadı (NO_EVIDENCE)',
        'NO_BASELINE': 'Önceki kaynak snapshot’ı yok', 'TEXT_UNCHANGED': 'Kaynak metni aynı',
        'TEXT_CHANGED_REVIEW_REQUIRED': 'Kaynak metni değişti; inceleme gerekli',
        'NEW_IN_SNAPSHOT': 'Bu snapshot’ta yeni provision'}.get(value, value)


GATES = {'PAGE_CRUMB': 'sayfa numarası / tarih satırı', 'HEADING_FRAGMENT': 'başlık kırıntısı', 'TORN_FRAGMENT': 'kopuk satır parçası',
         'LOW_RELEVANCE': 'ilgi eşiğinin altında'}
CONTROL = {'SUPPORTS': 'kontrol kaydı destekliyor', 'PARTIAL': 'kontrol kaydı kısmen destekliyor', 'CONFLICT': 'kontrol kaydı çelişiyor',
           'NONE': 'okunan kontrol satırları ilgisiz', 'NOT_READ': 'kontrol kaydı okunmadı'}
REMEDIATION_STATUS = {'PROPOSED': 'öneri üretildi', 'NOT_NEEDED': 'öneri gerekmedi (yazılı kapsam var)', 'NOT_APPLICABLE': 'uygulanmadığı için öneri yok',
                      'UNDETERMINED': 'kapsam belirsiz; öneri üretilmedi, insan incelemesi', 'NOT_ASSESSED': 'AI değerlendiremedi; öneri üretilemedi',
                      'DISABLED': 'öneri üretimi kapalı'}


def scope_block(data):
    """What the analysis actually covered: the targets, the helper provisions, the counts."""
    regulation = data.get('regulation') or {}
    if 'targets' not in regulation:
        return ''
    turkish = regulation.get('regulator') == 'TR'
    number = lambda text: text.split(' md. ')[-1] if turkish else text
    helpers = regulation.get('helper_provisions') or []
    roles = {'scope': 'kapsam / yükümlüler / tanımlar', 'context': 'çapraz atıf bağlamı'}
    parts = ['<section class="intro"><h2>Analiz kapsamı</h2><ul class="basis">',
             f'<li><b>Kullanıcının seçtiği hedef maddeler:</b> {esc(", ".join(regulation.get("target_filter") or []) or "filtre yok (operatif maddelerin tamamı)")}</li>',
             f'<li><b>Analiz edilen hedef provision:</b> {esc(regulation.get("target_provisions", 0))} '
             f'<span class="muted">({esc(", ".join(number(t) for t in regulation["targets"]))})</span>'
             + (f' <span class="muted">— kapsam maddesi de olan hedef: {esc(", ".join(number(t) for t in regulation.get("targets_also_scope") or []))}</span>'
                if regulation.get('targets_also_scope') else '') + '</li>',
             f'<li><b>Context olarak okunan yardımcı provision:</b> {esc(regulation.get("helper_count", len(helpers)))} '
             + ('<span class="muted">(' + esc('; '.join(f'{number(h["label"])} · {roles.get(h["role"], h["role"])}'
                                                     + (f' · {h["heading"]}' if h.get('heading') else '') for h in helpers)) + ')</span>' if helpers else '')
             + '</li>',
             '<li class="muted">Yardımcı provision\'lar uygulanabilirlik dayanağı ve çıkarım bağlamı olarak okunur; onlardan yükümlülük adayı üretilmez.</li>',
             '</ul></section>']
    return ''.join(parts)


REASONS = {'GROUNDING_REJECTED': 'Model çıktısı kaynak metinle doğrulanamadı',
    'MISSING_CONSOLIDATED_SOURCE': 'Atıf yapılan kaynak indirilen bölümde yok',
    'UNRESOLVED_CROSS_REFERENCE': 'Çapraz atıf çözülemedi', 'SECOND_PASS_UNCERTAIN': 'İkinci kontrol emin olamadı',
    'CONTEXT_BUDGET_EXCEEDED': 'Bağlam sınırı aşıldı', 'NO_EXPLICIT_OBLIGATION': 'Açık bir yükümlülük ifadesi yok',
    'PROVIDER_FAILURE': 'Model hatası', 'INSUFFICIENT_EVIDENCE': 'Yetersiz dayanak',
    'CONTEXT_REQUIRES_LLM': 'Bağlam AI gerektiriyor (kural tabanlı akış)', 'DELETED_PROVISION': 'Silinmiş provision',
    'SOURCE_SIZE_OR_EMPTY': 'Metin boş ya da çok uzun', 'AMBIGUOUS_PRINTED_LABEL': 'Etiket belirsiz',
    'SOURCE_CONTAINS_OMITTED_TEXT': 'Kaynakta atlanmış metin işareti var',
    'AMENDMENT_REQUIRES_CONSOLIDATED_SOURCE': 'Değişiklik metni; konsolide kaynak gerekir',
    'CLASSIFIED_DEFINITION': 'Tanım hükmü; şirkete yükümlülük yüklemez',
    'CLASSIFIED_SCOPE': 'Kapsam / amaç hükmü; yükümlülük değil',
    'CLASSIFIED_DELEGATION': 'Yetki / usul devri hükmü ("yönetmelikle belirlenir"); şirkete yükümlülük yüklemez',
    'CLASSIFIED_ENFORCEMENT': 'Yaptırım hükmü; yükümlülüğün kendisi değil',
    'CLASSIFIED_PERMISSION': 'Yetki veya izin ("-ebilir"); yükümlülük değil',
    'CLASSIFIED_EXEMPTION': 'İstisna hükmü; yükümlülük değil',
    'CLASSIFIED_OTHER': 'Açık bir yükümlülük ifadesi yok'}
KINDS = {'OBLIGATION': 'yükümlülük', 'PROHIBITION': 'yasak', 'RECOMMENDATION': 'tavsiye', 'GUIDANCE': 'rehber',
         'PERMISSION': 'yetki/izin', 'EXEMPTION': 'istisna', 'DEFINITION': 'tanım', 'SCOPE': 'kapsam',
         'DELEGATION': 'yetki devri', 'ENFORCEMENT': 'yaptırım', 'OTHER': 'diğer'}
PRIORITIES = {'HIGH': 'YÜKSEK', 'MEDIUM': 'ORTA', 'LOW': 'DÜŞÜK'}
REMEDIATION_TYPES = {'NEW_POLICY_CLAUSE': 'Yeni policy maddesi', 'POLICY_UPDATE': 'Policy güncellemesi',
                     'CONFLICT_RESOLUTION': 'Çelişki giderme'}
TYPES = {'RULE': 'R', 'GUIDANCE': 'G', 'DIRECTION': 'D', 'EVIDENTIAL': 'E'}


def remediation_block(row):
    """The AI proposal for a gap, marked as a proposal awaiting a person."""
    item = (row.get('proposal') or {}).get('remediation')
    if not item:
        return ''
    parts = [f'<section class="remediation"><h3>AI öneri — onay gerektirir <span class="tag">{esc(item["label"])}</span></h3>',
             f'<p><b>Tür:</b> {esc(REMEDIATION_TYPES.get(item["type"], item["type"]))} · <b>Öncelik:</b> {esc(PRIORITIES.get(item["priority"], item["priority"]))} '
             f'<span class="muted">(kural: {esc(", ".join(item.get("priority_inputs", [])))})</span> · <b>Tetikleyen:</b> {esc(label(item["trigger"]))}</p>',
             f'<p>{esc(item["recommended_action"])}</p>']
    if item.get('missing_requirements'):
        parts.append('<p><b>Eksik gereklilikler:</b> '+esc('; '.join(item['missing_requirements']))+'</p>')
    if item.get('suggested_policy_language'):
        parts.append('<p><b>Önerilen policy ifadesi (AI taslağı, kanıt değil):</b></p><blockquote>'+esc(item['suggested_policy_language'])+'</blockquote>')
    parts.append(f'<p class="muted">{esc(item.get("implementation_notes", ""))}</p></section>')
    return ''.join(parts)


ENTITY_ROLES = {'obliged_party': 'yükümlü türü (kim yapacak)', 'counterparty': 'muhatap türü (kimin için)'}
ENTITY_MATCH = {'MATCH': 'eşleşiyor', 'MISMATCH': 'EŞLEŞMİYOR', 'UNDETERMINED': 'profil söylemiyor', 'NOT_RESTRICTED': 'bent sınırlamıyor'}


def basis_block(proposal, sources=None):
    rows = proposal.get('basis') or []
    if not rows:
        return ''
    sources = sources or {}
    marks = {'YES': 'eşleşiyor', 'NO': 'eşleşmiyor', 'UNCLEAR': 'belirsiz'}
    parts = ['<p class="muted"><b>Karşılaştırılan dayanaklar (şirket bilgisi ↔ kapsam ifadesi):</b></p><ul class="basis">']
    for b in rows:
        where = sources.get(b.get('source_id'), {}).get('printed_label') or ('bu provision' if b.get('source_id') else '')
        detail = ''
        if b['match'] == 'NO':
            # Every NO says where it came from and whether it could veto (v0.16.1).
            detail = (' · <b>açık istisna</b> (veto)' if b.get('exclusionary') else
                      ' · açık istisna değil, veto etmedi' if b.get('exclusionary') is False else '')
        parts.append(f'<li><b>{esc(marks.get(b["match"], b["match"]))}</b> · “{esc(b["company_fact"])}”'
                     + (f' <span class="muted">({esc(b["company_fact_key"])})</span>' if b.get('company_fact_key') else '')
                     + f' ↔ “{esc(b["regulatory_condition"])}”' + (f' <span class="muted">[{esc(where)}]</span>' if where else '') + detail + '</li>')
    parts.append('</ul>')
    if proposal.get('applicability_rule') == 'UPGRADED_FROM_UNKNOWN_BY_BASIS':
        parts.append('<p class="muted">Model UNKNOWN demişti; eşleşen bir dayanak bulunduğu ve hiçbiri çelişmediği için kural gereği '
                     '"olası uygulanabilir" olarak kaydedildi.</p>')
    return ''.join(parts)


def scope_table(proposal):
    """The six-field applicability basis of the clause (v0.16.1)."""
    scope = proposal.get('applicability_scope')
    if not scope:
        return ''
    required = scope.get('required_entities') or []
    demanded = ('; '.join(f'{ENTITY_ROLES.get(r["role"], r["role"])}: {r["type"]} (“{r["text"]}”, {r["source"]}) → {ENTITY_MATCH.get(r["match"], r["match"])}'
                          for r in required) or 'bent belirli bir yükümlü ya da muhatap türü adlandırmıyor')
    company = (f'varlık türü: {", ".join(scope.get("company_entity_types") or []) or "profilde belirtilmemiş"} · müşteri aileleri: '
               f'{", ".join(scope.get("company_customer_families") or []) or "profilde belirtilmemiş"}' + (' · merkezi yurt dışında' if scope.get('company_foreign_hq') else ''))
    rows = [('Ana provision', scope['parent_provision']), ('Alt bent', f'{scope["child_clause"]} — {scope["child_excerpt"][:160]}'),
            ('Gerekli varlık türü', demanded), ('Şirket profili varlık türü', company),
            ('Eşleşme', ENTITY_MATCH.get(scope['match'], scope['match']) + (' · kural: entity gate' if scope['rule'] == 'ENTITY_GATE' else ' · provizyon düzeyi yargı geçerli')),
            ('Sonuç', f'{scope["final"]} (provizyon düzeyi: '
                      + (scope['provision_state'] if scope.get('provision_assessed', True) else 'değerlendirilmedi, kural karar verdi') + ')')]
    return ('<details><summary>Uygulanabilirlik dayanağı — bent düzeyi (6 alan)</summary><table>'
            + ''.join(f'<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>' for k, v in rows) + '</table>'
            + (f'<p class="muted">{esc(scope["reason"])}</p>' if scope['rule'] == 'ENTITY_GATE' else '') + '</details>')


# v0.18 applicability decision chain, in the order engine.propose runs the gates. A packet before
# v0.18 has no trace['gates'] and shows only the entity-gate table above, exactly as before.
TRACE_GATES = {'REGULATION_SUBJECT_SCOPE': 'Regülasyon yükümlü kapsamı', 'COMPANY_ENTITY': 'Şirket varlık türü', 'JURISDICTION': 'Yargı alanı',
               'CUSTOMER_ENTITY': 'Müşteri türü', 'CHILD_CLAUSE': 'Alt bent', 'EXEMPTION': 'İstisna', 'MODEL': 'Model',
               'FINAL_AGGREGATOR': 'Nihai karar',
               # v0.19 t7 (P5): a provision-level answer a clause addressed only to the obliged parties does not inherit.
               'CHILD_ADDRESSEE': 'Alt bent muhatabı (yalnız yükümlüler)'}
TRACE_STATUS = {'MATCH': 'eşleşiyor', 'MISMATCH': 'eşleşmiyor', 'UNDETERMINED': 'belirlenemedi', 'NOT_RESTRICTED': 'sınırlama yok',
                'NONE': 'yok', 'EXEMPT': 'istisna uygulanıyor', 'POSSIBLE': 'olası istisna', 'NOT_ASKED': 'sorulmadı', 'FAILED': 'başarısız',
                'APPLIES': 'uygulanıyor', 'POSSIBLY_APPLIES': 'olası uygulanabilir', 'DOES_NOT_APPLY': 'uygulanmıyor', 'UNKNOWN': 'belirsiz'}
# Badge colour only: a status that rules the duty out, and one that leaves it open for a person.
TRACE_NEGATIVE = {'MISMATCH', 'EXEMPT', 'DOES_NOT_APPLY'}
TRACE_OPEN = {'UNDETERMINED', 'POSSIBLE', 'FAILED', 'UNKNOWN', 'POSSIBLY_APPLIES'}
DECIDED_BY = {'MODEL': 'model yargısı', 'UPGRADED_FROM_UNKNOWN_BY_BASIS': 'kural: model UNKNOWN, eşleşen dayanak var',
              'DOWNGRADED_BASIS_INCONSISTENT': 'kural: model kararı kendi dayanağıyla çelişti', 'ENTITY_GATE': 'kural: bent düzeyi entity gate',
              'NOT_ASSESSED': 'değerlendirilmedi', 'SUBJECT_SCOPE_GATE': 'kural: şirket regülasyonun yükümlü listesinde yok',
              'ADDRESSEE_GATE': 'kural: yükümlülük başka bir muhataba yöneltilmiş', 'JURISDICTION_GATE': 'kural: yargı alanı eşleşmiyor',
              'EXEMPTION_GATE': 'kural: istisna uygulanıyor', 'PROFILE_INCOMPLETE': 'kural: şirket profili eksik',
              'RULE_CLEAR_MATCH': 'kural: bütün kapılar açıkça eşleşti', 'PROFILE_AMBIGUOUS': 'kural: profil belirsiz, model kararı UNKNOWN kaydedildi'}
REVIEW_FLAGS = {'RULE_MODEL_DISAGREEMENT': 'kural ile model uyuşmuyor', 'APPLIES_TRACE_INCOMPLETE': 'APPLIES izinde kritik alan eksik',
                'EXEMPTION_POSSIBLE': 'istisna ifadesi var, bu şirkete uyduğu açık değil',
                'EXTRACTION_SALVAGED': 'yükümlülük kaynağa dayandırılarak kurtarıldı',
                # v0.19 t7 review 1: the extraction's second reading kept this duty as UNCERTAIN.
                'EXTRACTION_REVIEW_UNCERTAIN': 'çıkarımın ikinci okuması bu yükümlülükten emin olamadı'}
# v0.19 t7 review 1 (PROV-P5-2): PROFILE_AMBIGUOUS also records a provision-level answer that a clause addressed only to the obliged
# parties does not inherit (the CHILD_ADDRESSEE gate, OBLIGED_ADDRESSEE_UNDETERMINED): named apart, since the model did not say UNKNOWN.
WITHHELD_BY = ('kural: bent yalnız yükümlülere yöneltilmiş ve yükümlü listesi şirketi yerleştiremedi; provizyon düzeyindeki model yanıtı '
               'bu bende devralınmadı, UNKNOWN kaydedildi (PROFILE_AMBIGUOUS · OBLIGED_ADDRESSEE_UNDETERMINED)')
# v0.19 t7: the positive evidence gate's codes on a favourable reading it rejected (conflict.support_gate), and the escalation of a
# sentence giving a prohibited act an opposed effect (review 1, F1).
POSITIVE_GATES = {'POSITIVE_GATE_ACTOR': 'başka bir taraf', 'POSITIVE_GATE_SCOPE': 'başka bir müşteri grubu',
                  'POSITIVE_GATE_RECIPIENT': 'başka bir alıcı', 'POSITIVE_GATE_POLARITY': 'karşıt etki (yasak yerine zorunluluk ya da tersi)',
                  'POSITIVE_GATE_ACTION': 'başka bir eylem', 'POSITIVE_GATE_OBJECT': 'yalnız yasağın konusu, yasağın kendisi yok',
                  'OPPOSED_EFFECT': 'yasaklanan eylem karşıt etkiyle geçiyor; güçlü doğrulayıcıya soruldu'}
APPLIES_FIELDS = {'regulation_subject_basis': 'regülasyon yükümlü dayanağı', 'company_matching_field': 'eşleşen şirket alanı',
                  'jurisdiction_basis': 'yargı alanı dayanağı', 'child_clause_match': 'alt bent eşleşmesi',
                  'exclusion_check': 'istisna kontrolü', 'final_reasoning': 'nihai gerekçe'}


def named(table, code):
    # The backend code is printed beside the words, as label() does.
    return f'{table[code]} ({code})' if code in table else str(code)


def trace_block(proposal):
    """The v0.18 decision trace: each gate's status, reason and evidence, what settled the row and
    why a person must look. Empty for a packet without trace['gates']."""
    trace = proposal.get('trace') or {}
    gates = trace.get('gates')
    if not gates:
        return ''
    text = lambda value: ', '.join(str(v) for v in value) if isinstance(value, list) else str(value)
    parts = ['<details class="trace" open><summary>Karar izi</summary><ol>']
    for gate in filter(None, gates):
        status = gate.get('status') or '—'
        tone = ' no' if status in TRACE_NEGATIVE else ' open' if status in TRACE_OPEN else ''
        evidence = gate.get('evidence') or {}
        # applicability.gate() files a gate's extras (matched_items ...) under 'evidence'; the design
        # contract names them on the gate itself, so both places are read.
        extra = lambda key: gate.get(key) or evidence.get(key)
        cited = []
        if evidence.get('company_field'):
            cited.append(f'şirket: {text(evidence["company_field"])} = “{text(evidence.get("company_value") or "—")}”')
        if evidence.get('list_item'):
            cited.append(f'liste maddesi: “{text(evidence["list_item"])}”')
        if extra('matched_items'):
            cited.append(f'eşleşen: {text(extra("matched_items"))}')
        quote = str(evidence.get('quote') or '')
        # The subject gate quotes the list item it matched; printed once.
        if quote and quote != evidence.get('list_item'):
            cited.append(f'“{quote[:240]}{"…" if len(quote) > 240 else ""}”')
        # Which list or provision the gate read, also when it quotes nothing (an UNDETERMINED subject gate).
        if evidence.get('source_label'):
            cited.append(f'kaynak: {evidence["source_label"]}')
        parts.append(f'<li><b>{esc(TRACE_GATES.get(gate.get("gate"), gate.get("gate") or "—"))}</b> '
                     f'<span class="badge{tone}">{esc(TRACE_STATUS.get(status, status))} · {esc(status)}</span> '
                     f'<span class="muted">{"kesin" if gate.get("clear") else "kesin değil"}'
                     + (' · açık uçlu liste' if extra('open_ended') else '')
                     + (f' · mod: {esc(gate["mode"])}' if gate.get('mode') else '') + '</span>'
                     + (f'<br>{esc(gate["reason"])}' if gate.get('reason') else '')
                     + (f'<br><span class="muted">{esc(" · ".join(cited))}</span>' if cited else '') + '</li>')
    parts.append('</ol>')
    decided = trace.get('decided_by') or proposal.get('applicability_rule')
    if decided == 'PROFILE_AMBIGUOUS' and any((gate or {}).get('gate') == 'CHILD_ADDRESSEE' for gate in gates):
        parts.append(f'<p><b>Karar veren:</b> {esc(WITHHELD_BY)}</p>')
    elif decided:
        parts.append(f'<p><b>Karar veren:</b> {esc(named(DECIDED_BY, decided))}</p>')
    flags = proposal.get('review_flags') or []
    applies = trace.get('applies_trace') or {}
    # The APPLIES trace is filled for every row but only means something for an APPLIES.
    if applies.get('complete') is False and (proposal.get('applicability') == 'APPLIES' or 'APPLIES_TRACE_INCOMPLETE' in flags):
        parts.append('<p class="notice"><b>APPLIES izi eksik — kritik alan yok:</b> '
                     + esc(', '.join(named(APPLIES_FIELDS, f) for f in applies.get('missing') or []) or 'belirtilmedi') + '</p>')
    if flags:
        parts.append('<p class="notice"><b>İnsan incelemesi gerekli:</b> ' + esc('; '.join(named(REVIEW_FLAGS, f) for f in flags)) + '</p>')
    parts.append('</details>')
    return ''.join(parts)


IMPACT_KINDS = {'THRESHOLD_CHANGED': 'eşik / tutar', 'DEADLINE_CHANGED': 'süre', 'ENTITY_SCOPE_CHANGED': 'kapsam / muhatap',
                'EXEMPTION_CHANGED': 'istisna', 'TEXT_CHANGED': 'metin'}
ACTIONS = [('', 'Seç…'), ('APPROVE', 'Onayla — AI önerisi aynen kalır'), ('REJECT', 'Reddet — AI önerisi yanlış'),
           ('OVERRIDE', 'Değiştir — kendi kararım (gerekçe zorunlu)'), ('NEEDS_EVIDENCE', 'Daha fazla kanıt gerekli')]


def impact_block(data):
    """What the new regulation snapshot changed for this company (v0.17 change impact)."""
    impact = data.get('impact')
    if not impact:
        return ''
    parts = []
    if impact['affects_company']:
        parts.append('<section class="impact"><h2>Bu regülasyon güncellemesi şirketinizi etkileyebilir</h2>'
                     '<p class="muted">Değişiklik türü kaynak metnin kendisinden kuralla okundu; hukuki etki değerlendirmesi değildir.</p>')
        for change in impact['changed']:
            kinds = ', '.join(IMPACT_KINDS.get(k, k) for k in change['kinds'])
            affected = ', '.join(change['affected_policies']) or 'önceki analizde bu maddeye bağlanan policy alıntısı yok'
            parts.append(f'<article><h3>{esc(change["label"])} · değişiklik: {esc(kinds)}</h3><div class="columns">'
                         f'<section><h4>Eski metin</h4><blockquote>{esc(change["old_text"])}</blockquote></section>'
                         f'<section><h4>Yeni metin</h4><blockquote>{esc(change["new_text"])}</blockquote></section></div>'
                         f'<p><b>Etkilenen policy:</b> {esc(affected)}</p><p><b>Önerilen eylem:</b> {esc(change["suggested_action"])}</p></article>')
        if impact['new']:
            parts.append('<p><b>Bu snapshot\'ta yeni provision:</b> ' + esc(', '.join(impact['new'])) + '</p>')
        parts.append('</section>')
    line = (f'Önceki analizle karşılaştırma: {impact["carried_forward"]} yükümlülük değişmeyen maddelerden taşındı (model çağrısı yapılmadı), '
            f'{impact["reanalysed"]} yükümlülük yeniden değerlendirildi.')
    if impact.get('reuse_blocked_reason'):
        line += f' Taşıma yapılmadı: {impact["reuse_blocked_reason"]}.'
    if impact.get('deleted_or_unselected'):
        line += ' Önceki analizde olup bu seçimde olmayan: ' + ', '.join(impact['deleted_or_unselected']) + '.'
    parts.append(f'<p class="muted">{esc(line)}</p>')
    return ''.join(parts)


def overview(data):
    """One line per provision, so a whole chapter can be read before any single card."""
    by_source = {}
    for row in data['obligations']:
        by_source.setdefault(row['source_id'], []).append(row)
    rows = data['obligations']
    applies = [r for r in rows if r['proposal']['applicability'] == 'APPLIES']
    possibly = [r for r in rows if r['proposal']['applicability'] == 'POSSIBLY_APPLIES']
    remediations = [r for r in rows if r['proposal'].get('remediation')]
    coverage = lambda state: sum(1 for r in rows if r['proposal']['coverage'] == state)
    failed_drafts = sum(1 for r in remediations if r['proposal']['remediation'].get('draft_status') == 'DRAFT_UNAVAILABLE')
    undetermined = sum(1 for r in rows if r['proposal'].get('remediation_status') == 'UNDETERMINED')
    counts = [(len(by_source), len(data['cases']), 'hedef provision\'dan yükümlülük adayı çıktı'),
              (len(applies), len(rows), 'aday için APPLIES (uygulanıyor) önerildi'),
              *([(len(possibly), len(rows), 'aday için POSSIBLY_APPLIES (özde eşleşme, eksik bilgi) önerildi')] if possibly else []),
              (sum(1 for r in rows if r['proposal']['applicability'] == 'DOES_NOT_APPLY'), len(rows), 'aday için DOES_NOT_APPLY önerildi'),
              *([(sum(1 for r in rows if r['proposal'].get('applicability_rule') == 'ENTITY_GATE'), len(rows),
                  'aday bent düzeyi kuralla (entity gate) profil dışı: muhatap/yükümlü türü eşleşmedi')]
                if any(r['proposal'].get('applicability_rule') == 'ENTITY_GATE' for r in rows) else []),
              (sum(1 for r in rows if r['proposal']['applicability'] == 'UNKNOWN'), len(rows), 'adayda uygulanabilirlik UNKNOWN kaldı'),
              (coverage('COVERS_TEXT'), len(rows), 'adayda yazılı policy desteği (COVERS_TEXT)'),
              (coverage('PARTIAL'), len(rows), 'adayda kısmi kapsam (PARTIAL)'),
              (sum(1 for r in rows if r['signals']), len(rows), 'adayda doğrulanmış policy çelişkisi (CONFLICT)'),
              (coverage('UNKNOWN'), len(rows), 'adayda kapsam hükmü verilemedi (UNKNOWN)'),
              (coverage('NO_EVIDENCE'), len(rows), 'adayda policy kanıtı bulunamadı (NO_EVIDENCE)'),
              *([(len(remediations), len(rows), 'aday için AI önerisi üretildi (onay gerektirir)')] if remediations else []),
              *([(failed_drafts, len(remediations), 'öneride policy ifadesi taslağı üretilemedi (kural kısmı duruyor)')] if failed_drafts else []),
              *([(undetermined, len(rows), 'adayda kapsam belirsiz olduğu için öneri üretilmedi')] if undetermined else []),
              (sum(1 for r in rows if r.get('status') == 'HUMAN_REVIEW_REQUIRED'), len(rows), 'aday insan onayı bekliyor')]
    controls = {c['source_id'] for p in data['policies'] for c in p['chunks'] if c.get('locator') == 'control_row'}
    control_covered = [r for r in applies if any(c['relation'] == 'SUPPORTS' and c['source_id'] in controls for c in r['proposal']['policy_checks'])]
    covered = sum(1 for r in applies if r['proposal']['coverage'] == 'COVERS_TEXT')
    pct = lambda a, b: f'%{round(100 * a / b)}' if b else '—'
    parts = [scope_block(data), '<section class="intro"><h2>Bölüm özeti</h2>',
             f'<p><b>Uygulanabilir (APPLIES): {esc(len(applies))} / {esc(len(rows))} ({esc(pct(len(applies), len(rows)))})</b>'
             + (f' · <b>Özde eşleşen (POSSIBLY_APPLIES): {esc(len(possibly))}</b>' if possibly else '') + ' · '
             f'<b>Yazılı policy kapsaması: {esc(covered)} / {esc(len(applies))} ({esc(pct(covered, len(applies)))})</b>'
             + (f' · <b>Kontrol kaydı kanıtı: {esc(len(control_covered))} / {esc(len(applies))} ({esc(pct(len(control_covered), len(applies)))})</b> '
                f'<span class="muted">({esc(len(controls))} kontrol satırı; kontrol kaydı yazılı policy kapsamı sayılmaz)</span>' if controls else '') + '</p>']
    categories = {}
    for r in rows:
        categories[r.get('category') or 'Sınıflanmadı'] = categories.get(r.get('category') or 'Sınıflanmadı', 0) + 1
    if any(r.get('category') for r in rows):
        parts.append('<p class="muted">Risk kategorisine göre (AI sınıflaması, yönlendirme amaçlı): ' +
                     esc(' · '.join(f'{k}: {v}' for k, v in sorted(categories.items(), key=lambda kv: -kv[1]))) + '</p>')
    parts.append('<div class="counts">')
    parts += [f'<div><strong>{esc(a)} / {esc(b)}</strong>{esc(text)}</div>' for a, b, text in counts]
    parts.append('</div><p class="muted">Bunlar AI önerilerinin sayımıdır; uyum oranı değildir. Aday çıkmayan bir provision, '
                 'yükümlülük içermediği anlamına gelmez. Her satır insan incelemesi bekler.</p>'
                 '<p class="notice"><b>“Policy desteği bulundu” çelişki olmadığı anlamına gelmez.</b> Ölçüldü: yerel modeller, '
                 'kuralın kelimelerini olumsuzlamadan özünü ihlal eden policy ifadelerini (ör. “kısa gönderilerde risk uyarısı '
                 'yazılmaz”) çoğunlukla yakalayamıyor; bu tür pasajlar arama sıralamasında da geride kaldığı için AI’a hiç '
                 'gösterilmeyebiliyor. Policy belgelerinin tamamını kendin oku.</p>'
                 '<div class="scroll"><table><thead><tr><th>Provision</th><th>Tür</th><th>Yükümlülük adayı</th><th>AI özeti · kategori</th>'
                 '<th>Şirkete uygulanabilirlik</th><th>Policy kapsamı</th></tr></thead><tbody>')
    for case in data['cases']:
        source = case['source']
        kind = esc(TYPES.get(source['legal_type'], '?'))
        found = by_source.get(source['id'], [])
        if not found:
            kinds = [KINDS.get(u['kind'], u['kind']) for u in case.get('classification') or []]
            parts.append(f'<tr class="none"><td>{esc(source["printed_label"])}</td><td>{kind}</td><td colspan="4">Aday üretilemedi: '
                         f'{esc(REASONS.get(case["reason"], case["reason"]))}'
                         + (f' <span class="muted">(sınıf: {esc(", ".join(kinds))})</span>' if kinds else '') + '</td></tr>')
        for row in found:
            action = row['candidate']['required_action'] or row['candidate']['prohibited_action'] or ''
            flag = ' <b class="flag">olası çelişki</b>' if row['signals'] else ''
            parts.append(f'<tr><td><a href="#o-{esc(row["id"])}">{esc(source["printed_label"])}</a></td><td>{kind}</td>'
                         f'<td>{esc(row["candidate"]["modality"])} · {esc(action[:160])}{"…" if len(action) > 160 else ""}</td>'
                         f'<td>{esc(row.get("summary") or "—")}{(" · <i>" + esc(row["category"]) + "</i>") if row.get("category") else ""}</td>'
                         f'<td>{esc(label(row["proposal"]["applicability"]))}</td>'
                         f'<td>{esc(label(row["proposal"]["coverage"]))}{flag}</td></tr>')
    parts.append('</tbody></table></div></section>')
    return ''.join(parts)


def options(values, selected):
    return ''.join(f'<option value="{esc(key)}"'+(' selected' if key == selected else '')+
                   f'>{esc(label)}</option>' for key, label in values)


def render(packet, hosted=False):
    data = packet['events'][0]['payload']
    sources = {s['id']: s for s in data['scope_sources']}
    policies = {c['source_id']: c for p in data['policies'] for c in p['chunks']}
    cases = {c['source']['id']: c for c in data['cases']}
    reviewed = packet['events'][-1]['payload'].get('outcomes', []) if packet['count'] > 1 else []
    statuses = {r['obligation_id']: r['status'] for r in reviewed}
    saved = packet['events'][-1]['payload'].get('review') if packet['count'] > 1 else None
    saved_decisions = {d['obligation_id']: d for d in saved['decisions']} if saved else {}
    saved_audit = {a['obligation_id']: a for a in packet['events'][-1]['payload'].get('audit', [])} if saved else {}
    body = []
    for row in data['obligations']:
        oid, proposal, candidate = row['id'], row['proposal'], row['candidate']
        case = cases[row['source_id']]
        source = case['source']
        carried_tag = (' · <span class="tag">önceki analizden taşındı — metin ve girdiler aynı, AI çıktısı yeniden üretilmedi</span>'
                       if row.get('carried_forward') else '')
        parts = [f'<article class="obligation" data-id="{oid}" id="o-{esc(oid)}"><header><span class="tag">{esc(statuses.get(oid, "İnsan incelemesi bekliyor"))}</span>',
            f'<h2>{esc(source["printed_label"])}</h2></header><p class="action">{esc(candidate["required_action"] or candidate["prohibited_action"])}</p>',
            f'<p><b>Modalite:</b> {esc(candidate["modality"])} · <b>Kaynak karşılaştırması:</b> {esc(label(case["change"]["status"]))}{carried_tag}</p>',
            '<details open><summary>Regülasyon kanıtı</summary>',
            f'<blockquote>{esc(source["text"])}</blockquote>',
            f'<p><a href="{esc(source["source_url"])}" target="_blank" rel="noopener noreferrer">{"mevzuat.gov.tr kaynak metni" if source.get("locator_kind") == "mevzuat_madde" else "FCA kaynak sayfası"}</a> · Alınma: {esc(source["fetched_at"])}</p>']
        if row.get('summary'):
            parts.append(f'<p class="muted"><b>AI özeti (yönlendirme amaçlı, kanıt değil):</b> {esc(row["summary"])}'
                         + (f' · <b>Kategori:</b> {esc(row["category"])}' if row.get('category') else '') + '</p>')
        for link in row.get('sanctions') or []:
            parts.append(f'<p class="notice"><b>Yaptırım bağlantısı · {esc(link["label"])}:</b> {esc(link["quote"])}</p>')
        if row.get('multipart'):
            parts.append(f'<p class="notice"><b>Çok parçalı provision:</b> bu aday, {esc(row["multipart"]["units"])} alt paragraflı bir '
                         'provision\'ın tek bir alt paragrafından çıkarıldı. Diğer alt paragraflar kapsamı daraltıyor ya da koşul '
                         'ekliyor olabilir; yukarıdaki tam metni oku.</p>')
        if case['change']['old']:
            parts.append(f'<p>Önceki snapshot:</p><blockquote>{esc(case["change"]["old"]["text"])}</blockquote>')
        if case.get('duty_changes', {}).get('added') or case.get('duty_changes', {}).get('removed'):
            parts.append('<p>Yükümlülük adaylarındaki değişiklikler (anlam eşdeğerliği uzman incelemesi gerektirir):</p><blockquote>'+esc(json.dumps(case['duty_changes'], ensure_ascii=False, indent=2))+'</blockquote>')
        # The trigger of an APPLIES, in one line: which provision, which profile field, which fact.
        trigger = next((b for b in proposal.get('basis') or [] if b['match'] == 'YES'), None)
        if trigger and proposal['applicability'] in ('APPLIES', 'POSSIBLY_APPLIES'):
            where = sources.get(trigger['source_id'], {}).get('printed_label') or ('bu provision' if trigger['source_id'] == source['id'] else trigger['source_id'][:12])
            trigger_line = (f'<p class="muted"><b>Dayanak provision:</b> {esc(where)} · <b>Tetikleyen şirket bilgisi:</b> '
                            f'{esc(trigger.get("company_fact_key") or "profil")} = “{esc(trigger["company_fact"])}” ↔ “{esc(trigger["regulatory_condition"][:200])}”</p>')
        else:
            trigger_line = ''
        shared = ' <span class="muted">(bu provision\'ın yükümlülükleri için bir kez yargılandı)</span>' if row.get('applicability_siblings') else ''
        control_line = (f'<p class="muted"><b>Kontrol kaydı kanıtı:</b> {esc(CONTROL.get(proposal.get("control_coverage"), proposal.get("control_coverage")))}</p>'
                        if proposal.get('control_coverage') not in (None, 'NOT_READ') else '')
        coverage_label = ('Değerlendirilmedi — bent şirkete uygulanmıyor' if proposal.get('coverage_assessed') is False else label(proposal['coverage']))
        parts.extend(['</details><div class="columns"><section><h3>Şirkete uygulanabilirlik önerisi'+shared+'</h3>',
            f'<strong>{esc(label(proposal["applicability"]))}</strong>'
            + (' <span class="tag">kural: entity gate</span>' if proposal.get('applicability_rule') == 'ENTITY_GATE' else '')
            + f'<p>{esc(proposal["applicability_reason"])}</p>'+trigger_line+basis_block(proposal, sources)+scope_table(proposal)+'</section>',
            f'<section><h3>Yazılı policy kapsamı önerisi</h3><strong>{esc(coverage_label)}</strong><p>{esc(proposal["coverage_reason"])}</p>'
            + control_line + (f'<p class="muted"><b>AI önerisi durumu:</b> {esc(REMEDIATION_STATUS.get(proposal.get("remediation_status"), proposal.get("remediation_status") or "—"))}</p>'
                              if proposal.get('remediation_status') else '') + '</section></div>'])
        if proposal['missing_information']:
            parts.append('<p class="notice">Eksik / belirsiz: '+esc('; '.join(proposal['missing_information']))+'</p>')
        # What the applicability judge answered and why it was refused, so an operator is never
        # left with "manual analysis required" and no trace of the answer (seen live on md. 4).
        scope_notes = [d for d in row.get('diagnostics', []) if 'stage' not in d and d.get('code') in
                       ('PROPOSAL_INVALID', 'SCOPE_QUOTE_DROPPED', 'BASIS_DROPPED', 'FACT_KEYS_FILTERED', 'ProviderFailure', 'ContextBudgetError',
                        'DOWNGRADED_BASIS_INCONSISTENT')]
        if scope_notes:
            parts.append('<details><summary>Uygulanabilirlik yargısının notları (reddedilen ya da düşürülen yanıtlar)</summary>')
            for note in scope_notes:
                parts.append(f'<p class="muted"><b>{esc(note["code"])}</b> · deneme {esc(note.get("attempt", "—"))}: {esc(str(note.get("detail", ""))[:600])}</p>')
                if note.get('response_excerpt'):
                    parts.append('<blockquote>'+esc(str(note['response_excerpt'])[:1500])+'</blockquote>')
            parts.append('</details>')
        parts.append(remediation_block(row))
        if any(c['relation'] == 'CONFLICTS' for c in proposal.get('policy_checks', [])):
            parts.append('<p class="notice"><b>Olası policy çelişkisi:</b> AI bazı pasajları bu yükümlülükle çelişkili buldu. Bu işaret kesin bir gap kararı değildir; ilgili alıntıları ve policy önceliğini incele.</p>')
        ranking = row.get('retrieval')
        checks = proposal.get('policy_checks', [])
        relations = {check['source_id']: check['relation'] for check in checks}
        # Packets before v0.12 have no judged_policy_ids: there the model read only what was "shown".
        judged = row.get('judged_policy_ids')
        filtered = proposal.get('filtered_passages') or []
        if judged is not None and checks:
            # Headings and crumbs are never read by design; only real passages count as unread.
            unread = sum(1 for chunk in policies.values() if judgeable(chunk)) - len(judged) - len(filtered)
            parts.append(f'<p class="muted">Yargı modeli bu yükümlülük için {esc(len(judged))} policy pasajını tek tek, diğerlerini görmeden okudu.'
                         + (f' <b>Policy setindeki {esc(unread)} pasaj okunmadı</b> (sıralamada geride kaldılar); onlardaki bir çelişki bu öneride yoktur.'
                            if unread > 0 else ' Yüklenen policy setinin tamamı okundu.')
                         + (f' {esc(len(filtered))} pasaj kural gereği yargıya gönderilmedi (aşağıda adlandırıldı).' if filtered else '') + '</p>')
        if filtered:
            parts.append('<details><summary>Kanıt kapısı — kural gereği yargıya gönderilmeyen pasajlar</summary>')
            for item in filtered:
                ref = policies.get(item['source_id'], {})
                parts.append(f'<p class="muted">{esc(ref.get("filename", ""))} · {esc(ref.get("locator", ""))} {esc(ref.get("number", ""))} · '
                             f'<b>{esc(GATES.get(item["reason"], item["reason"]))}</b> ({esc(item["reason"])})</p><blockquote>{esc(ref.get("text", ""))}</blockquote>')
            parts.append('</details>')
        signals = row.get('evidence_signals') or []
        if signals:
            # The retrieval debug view: every signal per passage beside the gate and the judgement.
            parts.append('<details><summary>Kanıt hattı — pasaj başına arama sinyalleri, kapı kararı ve yargı</summary>'
                         '<div class="scroll"><table><thead><tr><th>Pasaj</th><th>Sıra</th><th>Anlam (benzerlik · sıra)</th><th>Kelime (skor · sıra)</th>'
                         '<th>RRF</th><th>Reranker (skor · sıra)</th><th>Kapı</th><th>Yargı</th></tr></thead><tbody>')
            for item in sorted(signals, key=lambda s: (s.get('rank') is None, s.get('rank') or 0)):
                ref = policies.get(item['source_id'], {})
                gate = GATES.get(item.get('gate'), item.get('gate')) if item.get('gate') else ('yargılandı' if item.get('judged') else 'yargıya gitmedi')
                verdict = item.get('relation') or '—'
                rerank = (f'{item["rerank_score"]} · {item.get("rerank_rank") or "—"}' if item.get('rerank_score') else '—')
                lexical = f'{item.get("lexical_score", "—")} · {item.get("lexical_rank") or "—"}'
                semantic = f'{item.get("similarity", "—")} · {item.get("semantic_rank") or "—"}' if item.get('similarity') else '—'
                parts.append(f'<tr><td>{esc(ref.get("locator", ""))} {esc(ref.get("number", ""))} <span class="muted">{esc(item["source_id"][:8])}</span></td>'
                             f'<td>{esc(item.get("rank") or "—")}</td><td>{esc(semantic)}</td><td>{esc(lexical)}</td><td>{esc(item.get("rrf") or "—")}</td>'
                             f'<td>{esc(rerank)}</td><td>{esc(gate)}</td><td><b>{esc(verdict)}</b>{" · seçildi" if item.get("selected") else ""}</td></tr>')
            parts.append('</tbody></table></div><p class="muted">Benzerlik ve reranker skoru sıralama sinyalidir, kanıt değildir. '
                         '"Seçildi": yargı modeli pasajı destek, kısmi destek ya da çelişki olarak aldı.</p></details>')
        if ranking:
            # The reviewer must be able to see what the model never read.
            titles = ((('AI’a gösterilen pasajlar', 'shown'), ('Sırada olup AI’a gösterilmeyen pasajlar', 'not_shown')) if judged is None else
                      (('Sıralamanın başındaki pasajlar', 'shown'), ('Sıralamada geride kalan pasajlar', 'not_shown')))
            parts.append(('<details><summary>Policy araması — AI’ın okuduğu ve okumadığı pasajlar</summary>' if judged is None else
                          '<details><summary>Policy araması — pasaj sıralaması ve yargı modelinin her pasaj için kararı</summary>') +
                '<p class="muted">Sıra, anlam benzerliği ile kelime örtüşmesinin birleşimidir. Benzerlik bir kapsam kanıtı veya olasılık değildir.</p>')
            for title, key in titles:
                if ranking[key]:
                    parts.append(f'<h4>{title}</h4>')
                for item in ranking[key]:
                    ref = policies[item['source_id']]
                    lexical = 'yok' if item['lexical_rank'] is None else item['lexical_rank']
                    verdict = '' if judged is None else (f' · <b>yargı: {esc(relations[item["source_id"]])}</b>' if item['source_id'] in relations
                                                         else ' · <b>yargı modeli okumadı</b>')
                    parts.append(f'<p>#{esc(item["rank"])} · {esc(ref["filename"])} · {esc(ref["locator"])} {esc(ref["number"])} · '
                        f'benzerlik {esc(item["similarity"])} · anlam sırası {esc(item["semantic_rank"])} · kelime sırası {esc(lexical)}{verdict}</p>'
                        f'<blockquote>{esc(ref["text"])}</blockquote>')
            parts.append('</details>')
        parts.append('<details><summary>AI tarafından kullanılan alıntılar</summary>')
        for evidence in proposal['scope_evidence']+proposal['policy_evidence']:
            ref = sources.get(evidence['source_id']) or policies.get(evidence['source_id'], {})
            name = ref.get('printed_label') or f'{ref.get("filename", "")} / {ref.get("locator", "")} {ref.get("number", "")}'
            parts.append(f'<p>{esc(name)}</p><blockquote>{esc(evidence["quote"])}</blockquote>')
        reasons = {r['source_id']: r.get('reason', '') for d in row.get('diagnostics', []) if d.get('stage') == 'passages' for r in d['results']}
        unrelated = sum(1 for check in checks if check['relation'] == 'UNRELATED')
        for check in checks:
            if check['relation'] == 'UNRELATED':
                continue
            ref = policies[check['source_id']]
            kind = 'Kontrol kaydı satırı' if ref.get('locator') == 'control_row' else 'Pasaj kontrolü'
            parts.append(f'<p>{kind} · {esc(ref["filename"])} · {esc(ref["locator"])} {esc(ref["number"])} · <b>{esc(check["relation"])}</b></p><blockquote>{esc(check["quote"])}</blockquote>')
            if reasons.get(check['source_id']):
                heading = 'Çelişki gerekçesi (ikinci okumayla doğrulandı)' if check['relation'] == 'CONFLICTS' else 'Yargı modelinin gerekçesi'
                parts.append(f'<p class="muted">{heading}: {esc(reasons[check["source_id"]])}</p>')
        if unrelated:
            parts.append(f'<p class="muted">Tek tek okunan {esc(len(checks))} pasajın {esc(unrelated)} tanesi bu yükümlülükle ilgisiz bulundu.</p>')
        # v0.19 t7: which favourable readings the positive evidence gate rejected, and why, by code.
        gated = Counter(note.get('code') for d in row.get('diagnostics', []) if d.get('stage') == 'passages' for r in d['results']
                        for note in r.get('notes') or [] if isinstance(note, dict) and note.get('code') in POSITIVE_GATES)
        if gated:
            parts.append('<p class="muted">Olumlu kanıt kapısı: ' + esc('; '.join(f'{POSITIVE_GATES[code]} ({code}) × {count}'
                                                                                for code, count in sorted(gated.items()))) + '</p>')
        if saved:
            decision = saved_decisions[oid]
            heading = 'Kaydedilmiş otonom AI kararı — insan onayı yok' if 'AI' in saved['reviewer_role'] else 'Kaydedilmiş insan incelemesi'
            entry = saved_audit.get(oid)
            audit_line = ''
            if entry:
                audit_line = (f'<p class="notice"><b>Denetim kaydı:</b> eylem {esc(entry.get("action") or "—")} · AI önerisi '
                              f'{esc(entry["ai_proposal"]["applicability"])} / {esc(entry["ai_proposal"]["coverage"])} · insan kararı '
                              f'{esc(entry["human_decision"]["applicability"])} / {esc(entry["human_decision"]["coverage"])}'
                              + (' · değişen alan: ' + esc(', '.join(entry['changed_fields'])) if entry['changed_fields'] else '')
                              + (' · override gerekçesi: ' + esc(entry['override_reason']) if entry.get('override_reason') else '')
                              + (' · AI kararını veren: ' + esc(entry['ai_trace'].get('decided_by') or '—')
                                 + (' · AI işaretleri: ' + esc(', '.join(entry['ai_trace']['review_flags'])) if entry['ai_trace'].get('review_flags') else '')
                                 if entry.get('ai_trace') else '')
                              + ' · AI çıktısı silinmedi; bu karar ayrı bir kanıt olayıdır.</p>')
            parts.append('</details><h3>'+esc(heading)+'</h3><p>'+esc(saved['reviewer'])+' · '+esc(saved['reviewer_role'])+'</p>'+audit_line+trace_block(proposal)+'<blockquote>'+esc(json.dumps(decision, ensure_ascii=False, indent=2))+'</blockquote></article>')
            body.append(''.join(parts))
            continue
        # v0.17 review surface: regulation | policy evidence | AI proposal and the decision, side by side.
        cited = [c for c in checks if c['relation'] != 'UNRELATED'][:6]
        policy_column = ''.join(f'<p class="muted">{esc(policies[c["source_id"]]["filename"])} · {esc(policies[c["source_id"]]["locator"])} '
                                f'{esc(policies[c["source_id"]]["number"])} · <b>{esc(c["relation"])}</b></p><blockquote>{esc(c["quote"])}</blockquote>'
                                for c in cited) or '<p class="muted">AI bu yükümlülüğe policy alıntısı bağlamadı.</p>'
        ai_column = (f'<p><b>Uygulanabilirlik:</b> {esc(label(proposal["applicability"]))}</p><p><b>Kapsam:</b> {esc(coverage_label)}</p>'
                     + trace_block(proposal) + '<label>Karar<select data-field="action">' + options(ACTIONS, '') + '</select></label>'
                     '<label>Override gerekçesi<textarea data-field="override_reason" placeholder="AI önerisini değiştiriyorsan neden: hangi kanıt, hangi hüküm."></textarea></label>')
        parts.append('</details><fieldset><legend>İnceleyen kararı — otomatik onay verilmez</legend><div class="tri">'
                     f'<section><h4>Regülasyon</h4><blockquote>{esc(source["text"])}</blockquote></section>'
                     f'<section><h4>Policy kanıtı</h4>{policy_column}</section>'
                     f'<section><h4>AI önerisi ve karar</h4>{ai_column}</section></div><div class="columns">')
        for field, caption, choices, default in [
            ('extraction', 'Yükümlülük çıkarımı', [('UNCERTAIN', 'Henüz kararsız'), ('ACCEPT', 'Çıkarımı kabul et'), ('REJECT', 'Çıkarımı reddet')], 'UNCERTAIN'),
            ('applicability', 'Şirkete uygulanabilirlik', [('UNKNOWN', 'Belirsiz'), ('APPLIES', 'Uygulanıyor'), ('DOES_NOT_APPLY', 'Uygulanmıyor')], 'UNKNOWN'),
            ('coverage', 'Yazılı policy kapsamı', [('UNKNOWN', 'Belirsiz'), ('COVERS_TEXT', 'Metindeki gerekliliği karşılıyor'), ('PARTIAL', 'Kısmen karşılıyor'), ('CONFLICT', 'Açık çelişki var'), ('NO_EVIDENCE', 'Kanıt bulunamadı')], 'UNKNOWN')]:
            parts.append(f'<label>{caption}<select data-field="{field}">{options(choices, default)}</select></label>')
        if proposal.get('remediation'):
            parts.append('<label>AI önerisi<select data-field="remediation">'
                         + options([('UNCERTAIN', 'Henüz kararsız'), ('ACCEPT', 'Öneriyi kabul et'), ('REJECT', 'Öneriyi reddet')], 'UNCERTAIN')
                         + '</select></label>')
        parts.append('</div><details><summary>Karara dayanak şirket bilgileri ve alıntıları seç</summary><h4>Şirket bilgileri</h4>')
        for key in ('jurisdictions', 'activities', 'licences', 'products', 'customer_types', 'description'):
            if data['company'].get(key):
                parts.append(f'<label class="check"><input type="checkbox" data-fact="{key}">{esc(key)}: {esc(data["company"][key])}</label>')
        parts.append('<h4>Uygulanabilirliğin regülasyon dayanağı</h4>')
        for sid, source in sources.items():
            parts.append(f'<label class="check"><input type="checkbox" data-scope="{sid}">{esc(source["printed_label"])}</label><blockquote>{esc(source["text"])}</blockquote>')
        # Every passage under every card is fine for three short policies. A long policy
        # against a whole chapter would repeat thousands of passages in one page.
        ranked = [r['source_id'] for key in ('shown', 'not_shown') for r in (row.get('retrieval') or {}).get(key, [])]
        decided = [check['source_id'] for check in proposal.get('policy_checks', []) if check['relation'] != 'UNRELATED']
        offered = policies if len(policies) <= 12 else {sid: policies[sid] for sid in dict.fromkeys(
            [*decided, *row.get('retrieved_policy_ids', []), *ranked]) if sid in policies}
        parts.append('<h4>Policy kanıtları — bütün okunan pasajlar</h4>' if offered is policies else
                     '<h4>Policy kanıtları — bu yükümlülük için sıralanan pasajlar</h4><p class="muted">Policy uzun olduğu için '
                     'yalnızca aramanın öne çıkardığı pasajlar listelenir; başka bir pasaj için belgenin kendisine bak.</p>')
        for sid, chunk in offered.items():
            parts.append(f'<label class="check"><input type="checkbox" data-policy="{sid}">{esc(chunk["filename"])} · {esc(chunk["locator"])} {chunk["number"]}</label><blockquote>{esc(chunk["text"])}</blockquote>')
        parts.append('</details><label>Gerekçe<textarea data-field="rationale" required placeholder="Kararını, kapsam ve istisnaları dikkate alarak gerekçelendir."></textarea></label></fieldset></article>')
        body.append(''.join(parts))
    no_candidates = [c for c in data['cases'] if not c['output']['obligations']]
    for case in no_candidates:
        body.append(f'<article><h2>{esc(case["source"]["printed_label"])}</h2><p class="notice">Aday üretilemedi: {esc(case["reason"])}</p><blockquote>{esc(case["source"]["text"])}</blockquote><p>Bu sonuç, yükümlülük bulunmadığı anlamına gelmez; uzman etiketleme listesinde değerlendirilmelidir.</p></article>')
    inputs = {'analysis_head': packet['events'][0]['event_hash'], 'scopes': sources, 'policies': policies,
              'hosted': hosted, 'read_only': bool(saved),
              # What APPROVE approves: the AI proposal's verdicts and quotes, copied as they are.
              'proposals': {row['id']: {'applicability': row['proposal']['applicability'], 'coverage': row['proposal']['coverage'],
                                        'company_fact_keys': row['proposal'].get('company_fact_keys', []),
                                        'scope_evidence': row['proposal'].get('scope_evidence', []),
                                        'policy_evidence': row['proposal'].get('policy_evidence', [])} for row in data['obligations']}}
    embedded = json.dumps(inputs, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    company = data['company']
    retrieval = data.get('policy_retrieval') or {}
    if str(retrieval.get('method', '')).startswith('hybrid'):
        search = ('Policy araması: anlam benzerliği + kelime örtüşmesi (yerel model '
                  +str(retrieval['embedding']['model_version']).split('@')[0]+'). Benzerlik hukuki ilgililik değildir.')
    else:
        search = 'Policy araması: yalnızca kelime örtüşmesi. Farklı kelimelerle yazılmış ilgili bölümler kaçabilir.'
    regulation = data.get('regulation') or {}
    if regulation:
        named = (f'T.C. mevzuatı · {esc(regulation.get("title") or regulation["module"]+" "+regulation["chapter"])}' if regulation.get('regulator') == 'TR'
                 else f'FCA {regulation["module"]} {regulation["chapter"]}')
        search = (f'Regülasyon: {named} · seçilen {regulation["selected"]} hedef provision\'ın '
                  f'{regulation["processed"]} tanesi işlendi'+(' (analiz durduruldu)' if regulation['stopped_early'] else '')
                  + (f' · {regulation["helper_count"]} yardımcı provision context olarak okundu' if regulation.get('helper_count') is not None else '')
                  + '. '+search)
    reranker = (retrieval.get('reranker') or {})
    if reranker.get('status') == 'active':
        search += f' Yeniden sıralama: yerel cross-encoder ({reranker.get("model", "")}).'
    official = 'GERÇEK MEVZUAT METNİ' if regulation.get('regulator') == 'TR' else 'GERÇEK FCA KAYNAKLARI'
    banner = f'SENTETİK ŞİRKET / ÖRNEK POLICY — {official}' if company['synthetic'] else 'GERÇEK ŞİRKET VERİSİ — İNSAN İNCELEMESİ GEREKLİ'
    return '''<!doctype html><html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cardaman · FCA inceleme pilotu</title><style>
:root{font-family:system-ui,sans-serif;color:#18312e;background:#edf2ef}*{box-sizing:border-box}body{margin:0}main{max-width:1120px;margin:auto;padding:32px 24px 64px}h1{font-size:36px;letter-spacing:-1px;margin:18px 0}h2{margin:12px 0}h3{font-size:16px}p{line-height:1.65}a{color:#126653}article,.intro,.reviewer{background:white;border:1px solid #d4e0d9;border-radius:16px;padding:26px;margin:22px 0}header .tag,.banner{font-size:12px;font-weight:700;letter-spacing:.5px;color:#6b4a0b;background:#fff0c9;padding:8px 12px;border-radius:6px;display:inline-block}.action{font-size:21px;font-weight:600}.tri{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:12px}.tri section{background:#f6f9f7;padding:12px;border-radius:8px}.tri h4{margin:0 0 8px}.tri textarea{width:100%;min-height:56px}.trace{margin:8px 0;font-size:13px}.trace ol{padding-left:18px;margin:4px 0}.trace li{margin:6px 0;line-height:1.5}.badge{display:inline-block;font-size:11px;font-weight:700;padding:1px 6px;border-radius:4px;background:#e1efe6;color:#155f4d}.badge.no{background:#fbe4dc;color:#8a3b12}.badge.open{background:#fff0c9;color:#6b4a0b}.impact{border:2px solid #b7791f;background:#fff8e6;padding:12px 16px;border-radius:12px;margin:16px 0}@media(max-width:900px){.tri{grid-template-columns:1fr}}.columns{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:20px}blockquote{white-space:pre-wrap;margin:12px 0;padding:16px;background:#f3f6f4;border-left:3px solid #578a74;line-height:1.7;font-size:14px;overflow-wrap:anywhere}details{margin:16px 0}summary{cursor:pointer;font-weight:600;padding:8px 0}.notice{color:#755213;background:#fff7e5;padding:12px;border-radius:8px}fieldset{border:1px solid #d4e0d9;border-radius:10px;padding:18px;margin-top:24px}legend{padding:0 8px;font-weight:600}label{display:block;font-size:14px}input:not([type=checkbox]),select,textarea{display:block;width:100%;padding:10px;margin:7px 0 14px;border:1px solid #a6bdb0;border-radius:6px;font:inherit}textarea{min-height:95px}.check{margin:10px 0;line-height:1.5}.check input{margin-right:8px}button{background:#155f4d;color:white;border:0;padding:14px 22px;border-radius:8px;font-weight:700;cursor:pointer}.muted{color:#526c62;font-size:14px}.brand{font-weight:800;letter-spacing:3px;color:#155f4d}.remediation{border:1px dashed #b98a2b;background:#fffaf0;border-radius:10px;padding:14px 18px;margin:16px 0}.remediation h3{margin-top:0}.basis{font-size:14px;line-height:1.6;padding-left:18px}.counts{display:flex;gap:22px;flex-wrap:wrap}.counts strong{font-size:24px;display:block}footer{font-size:13px;color:#526c62;line-height:1.7}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid #e1e9e4;vertical-align:top}th{font-size:12px;letter-spacing:.4px;color:#526c62}tr.none td{color:#7a8a83}.flag{color:#8a3b12;white-space:nowrap}.scroll{overflow-x:auto}
</style><main><div class="brand">CARDAMAN / PILOT</div><h1>Regülasyondan policy incelemesine</h1><div class="banner">'''+esc(banner)+'''</div><section class="intro"><h2>'''+esc(company['name'])+'''</h2><p>Bu ekran AI önerilerini kaynaklarıyla incelemek içindir. Policy metninin yeterliliği, kontrolün fiilen uygulandığını kanıtlamaz. Aşağıdaki seçimler başlangıçta belirsizdir; otomatik onay verilmez.</p><div class="counts"><div><strong>'''+str(len(data['cases']))+'''</strong>'''+('madde' if regulation.get('regulator') == 'TR' else 'FCA provision')+'''</div><div><strong>'''+str(len(data['obligations']))+'''</strong>yükümlülük adayı</div><div><strong>'''+str(len(data['policies']))+'''</strong>policy belgesi</div></div><p class="muted">'''+esc(search)+'''</p><details><summary>Şirket profili</summary><blockquote>'''+esc(json.dumps(company, ensure_ascii=False, indent=2))+'''</blockquote></details></section>'''+impact_block(data)+overview(data)+'''
<form id="review"><section class="reviewer"><div class="columns"><label>İnceleyen adı<input id="reviewer" required></label><label>Görev / uzmanlık<input id="role" required></label></div><p class="muted">Yerel operatör beyanıdır; kimlik veya uzmanlık doğrulaması yapılmaz.</p></section>'''+''.join(body)+'''
<button type="submit">İnceleme kararlarını JSON olarak indir</button><p id="message" aria-live="polite"></p></form><footer>İndirilen karar dosyası özgün analiz hash'ine bağlıdır. “pilot review” komutuyla kanıt paketine yeni olay olarak eklenir. Bu HTML dosyasına yazılanlar sunucuya gönderilmez; sayfa kapatıldığında indirilmemiş kararlar kaybolur. Kaynakta veya şirkette değişiklik varsa yeni analiz gerekir. Blockchain kaydı veya güvenilir zaman damgası oluşturulmamıştır.</footer>
<script type="application/json" id="data">'''+embedded+'''</script><script>
const data=JSON.parse(document.getElementById('data').textContent);
const submitButton=document.querySelector('button[type=submit]');
if(data.read_only){document.querySelector('.reviewer').hidden=true;submitButton.hidden=true;document.querySelector('footer').textContent='Kaydedilmiş inceleme salt okunur. Yeni bir değerlendirme için özgün analizi açın. Kimlik/uzmanlık operatör beyanıdır; blockchain kaydı yoktur.';}
else if(data.hosted){submitButton.textContent='İncelemeyi kanıt paketine kaydet';document.querySelector('footer').textContent='Kaydet düğmesi kararları bu bilgisayardaki çalışma alanına gönderir. Kaydedilmemiş seçimler sayfa kapanınca kaybolur. Kimlik/uzmanlık operatör beyanıdır; blockchain kaydı yoktur.';
document.getElementById('review').addEventListener('input',()=>parent.postMessage({type:'cardaman-dirty',analysis_head:data.analysis_head},'*'));
window.addEventListener('message',event=>{if(event.source!==parent||event.data?.type!=='cardaman-review-result'||event.data.analysis_head!==data.analysis_head)return;submitButton.disabled=false;document.getElementById('message').textContent=event.data.message;});}
document.getElementById('review').addEventListener('submit',e=>{e.preventDefault();const cards=[...document.querySelectorAll('.obligation')];if(!cards.length){document.getElementById('message').textContent='Aday yok; labels.json çalışma listesini uzman incelemesine gönder.';return;}
const decisions=cards.map(card=>{const get=k=>{const el=card.querySelector('[data-field="'+k+'"]');return el?el.value:null;};
const d={obligation_id:card.dataset.id,extraction:get('extraction'),applicability:get('applicability'),coverage:get('coverage'),rationale:get('rationale'),remediation:get('remediation')||'UNCERTAIN',action:get('action')||null,override_reason:get('override_reason')||'',company_fact_keys:[...card.querySelectorAll('[data-fact]:checked')].map(x=>x.dataset.fact),scope_evidence:[...card.querySelectorAll('[data-scope]:checked')].map(x=>({source_id:x.dataset.scope,quote:data.scopes[x.dataset.scope].text})),policy_evidence:[...card.querySelectorAll('[data-policy]:checked')].map(x=>({source_id:x.dataset.policy,quote:data.policies[x.dataset.policy].text}))};
if(d.action==='APPROVE'){const ai=data.proposals[d.obligation_id];d.applicability=ai.applicability==='POSSIBLY_APPLIES'?'UNKNOWN':ai.applicability;d.coverage=ai.coverage;d.company_fact_keys=ai.company_fact_keys;d.scope_evidence=ai.scope_evidence;d.policy_evidence=ai.policy_evidence;if(d.extraction==='UNCERTAIN')d.extraction='ACCEPT';}
if(d.action!=='OVERRIDE')d.override_reason='';return d;});
for(const d of decisions){if(!d.action){document.getElementById('message').textContent='Her yükümlülük için karar seç: onayla, reddet, değiştir ya da daha fazla kanıt iste.';return;}if(d.action==='OVERRIDE'&&!d.override_reason.trim()){document.getElementById('message').textContent='Override için gerekçe zorunlu: AI önerisini neden değiştirdiğini yaz.';return;}if(d.action==='NEEDS_EVIDENCE')continue;if(d.applicability!=='UNKNOWN'&&(!d.company_fact_keys.length||!d.scope_evidence.length)){document.getElementById('message').textContent='Kesin uygulanabilirlik kararı için şirket bilgisi ve regülasyon dayanağı seç.';return;}if(['COVERS_TEXT','PARTIAL','CONFLICT'].includes(d.coverage)&&!d.policy_evidence.length){document.getElementById('message').textContent='Policy kapsamı/çelişkisi için policy kanıtı seç.';return;}}
const value={format:'regchain-pilot-review-v1',analysis_head:data.analysis_head,reviewer:document.getElementById('reviewer').value,reviewer_role:document.getElementById('role').value,decisions};
if(data.read_only)return;
if(data.hosted){submitButton.disabled=true;document.getElementById('message').textContent='Kaydediliyor…';parent.postMessage({type:'cardaman-review',value},'*');return;}
const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='pilot-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);document.getElementById('message').textContent='Karar dosyası indirildi. Kanıt paketine eklemek için review komutunu çalıştır.';});
</script></main></html>'''
