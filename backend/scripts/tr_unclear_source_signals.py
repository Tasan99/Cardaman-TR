"""Indicative text signals on the obligations whose addressee is UNCLEAR (one record per obligation). A signal says where
to read, not who is bound. argv: unknown-causes.json out.json"""
import collections
import json
import re
import sys

from regchain.tr.frames import fold

d = json.load(open(sys.argv[1], encoding='utf-8'))
recs = {r['obligation_id']: r for r in d['records'] if r['review_type'] == 'CLARIFY_REGULATORY_SCOPE'}
AUTHORITY_START = re.compile(r'^\s*(?:\(\d+\)\s*)?(?:[a-zçğıöşü]{1,2}\)\s*)?(?:bakanlık|bakan\b|kurum\b|kurul\b|il özel idare|belediye|valilik|'
                             r'maliye bakanlığı|yetkili merci|genel müdürlük)')
AGENTIVE = re.compile(r'(?:bakanlık|kurum|kurul|merci|idare)[a-zçğıöşü]*\s+tarafından|bakanlıkça|kurumca|kurulca')
ADMIN_VERB = re.compile(r'(?:belirlenir|verilir|alınır|yapılır|ödenir|ödenmez|karşılanır|düzenlenir|yayımlanır|tahsil edilir|'
                        r'iptal edilir|kaydedilir|incelenir|denetlenir)\s*\.?\s*$')
CONSUMER = re.compile(r'^\s*(?:\(\d+\)\s*)?(?:[a-zçğıöşü]{1,2}\)\s*)?tüketici(?:ler)?\b')
ANAPHORA = re.compile(r'(?:\bbu (?:etiket|belge|bilgi|ürün|mal|hizmet|kişi|işletme)|\btaraflar\b|\bilgililer\b|\bsorumlular\b)')
NAMED_PARTY = re.compile(r'^\s*(?:\(\d+\)\s*)?(?:[a-zçğıöşü]{1,2}\)\s*)?[^,;.]{0,120}?[a-zçğıöşü]+(?:lar|ler|ları|leri)\s*,')
NOT_A_DUTY = re.compile(r'(?:söz konusu olur|sayılır|hükümleri uygulanır|uygulanır)\s*\.?\s*$')

groups = collections.defaultdict(list)
for r in recs.values():
    text = fold(r['text'])
    chapeau = fold(r['chapeau'] or '')
    if AUTHORITY_START.search(text) or AGENTIVE.search(text) or (r['authorities'] and ADMIN_VERB.search(text)):
        key = 'AUTHORITY_TASK_READ_AS_COMPANY_DUTY'
    elif CONSUMER.search(text) or CONSUMER.search(chapeau):
        key = 'CONSUMER_RIGHT_OR_CONSUMER_SUBJECT'
    elif NOT_A_DUTY.search(text):
        key = 'LEGAL_CONSEQUENCE_OR_REFERENCE_NOT_A_DUTY'
    elif r['prior_sentence_actors'] and ANAPHORA.search(text):
        key = 'ADDRESSEE_IN_PRIOR_SENTENCE_BY_ANAPHORA'
    elif r['list_item']:
        key = 'LIST_ITEM_ADDRESSEE_IN_LEAD_IN_OR_CLOSING'
    elif NAMED_PARTY.search(text) and not r['authorities']:
        key = 'NAMED_PARTY_WITHOUT_VOCABULARY_CLASS'
    else:
        key = 'IMPERSONAL_NO_PARTY_FOUND'
    groups[key].append(r)
out = {k: {'obligations': len(v), 'records': 5 * len(v), 'examples': [{'ref': r['provision_ref'], 'marker': r['marker'], 'text': r['text'][:200]} for r in v[:4]]}
       for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))}
json.dump(out, open(sys.argv[2], 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for k, v in out.items():
    print(f"{k}: {v['obligations']} obligations")
    for e in v['examples'][:3]:
        print('   ', e['ref'], '|', e['marker'], '|', e['text'][:150].replace('\n', ' '))
