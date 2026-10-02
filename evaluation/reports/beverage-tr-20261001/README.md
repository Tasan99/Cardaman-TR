# Cardaman TR beverage pilots - selective adjudication round, 30 September - 1 October 2026

Etiketler geliştirici etiketleridir (DEVELOPMENT); her sayı **INDICATIVE**. Veri seti BEVERAGE_TR_DEV_V2 0.4.0. Kod: ölçüm öncesi dondurulan commit `6658175`, sonrası `3e36ebf`.

## Setler

- **DEV**: fit (rules were corrected against it)
- **HOLDOUT**: blind until 6658175; development data since (its errors were used for comparer v2)
- **VALIDATION**: unseen: 42 cases on a second register per pilot, written before the held-out results were read; scored once (run 8)

## 1. Hüküm okuma (kural okuyucu; 4B çıkarım modeli run-4)

| Görev | DEV (fit) | HOLDOUT (kör, ölçüm öncesi) | HOLDOUT (şimdi) |
|---|---|---|---|
| OBLIGATION_EXTRACTION/rule | 289/289 | 105/112 | 105/112 (93.8 %) |
| ADDRESSEE_MATCH/rule | 204/204 | 55/61 | 55/61 (90.2 %) |
| EXCEPTION_DETECTION/rule | 60/60 | 21/27 | 22/27 (81.5 %) |
| OBLIGATION_EXTRACTION/model | 231/289 | — | 89/112 (79.5 %) |

Not: the exemption wording "gerek yoktur" (found on a corpus false COVERED) turned one held-out exception label right.

## 2. Politika kapsaması, yalnızca kurallar (model yok)

| Set | Önce (6658175) | Sonra (3e36ebf) | Yanlış CONTRADICTED / kaçan CONTRADICTED |
|---|---|---|---|
| DEV | 33/45 | 36/45 (80.0 %) | 0 / 0 |
| HOLDOUT | 10/28 | 18/28 (64.3 %) | 0 / 2 |
| VALIDATION | 17/42 | 18/42 (42.9 %) | 1 / 3 |

## 3. Seçici hakemlik: kurallar + 8B (dondurulmuş ölçümler)

Otomatik karar = incelemeye gitmeyen vaka. İncelemeye giden vaka doğru sayılmaz (strict). Model önerisi ayrı sütundur.

| Bacak | Kural doğru | Otomatik doğru/verilen | Oto. yanlış | Oto. yanlış COVERED / CONTRADICTED | İnceleme | Strict doğruluk | Model önerisi doğru | Çağrı süresi |
|---|---|---|---|---|---|---|---|---|
| HOLDOUT, önce, 8B quick | 10/28 | 10/18 | 8 | 0 / 0 | 10 (10 kural hatalı) | 35.7 % | 10/11 | 3.8 sn |
| HOLDOUT, önce, 8B thinking | 10/28 | 11/23 | 12 | 0 / 0 | 5 (5 kural hatalı) | 39.3 % | 6/11 | 24.9 sn |
| VALIDATION, sonra, 8B thinking | 19/42 | 25/31 | 6 | 0 / 1 | 11 (10 kural hatalı) | 59.5 % | 18/21 | 17.9 sn |
| VALIDATION, sonra, 8B quick | 19/42 | 18/22 | 4 | 0 / 1 | 20 (19 kural hatalı) | 42.9 % | 16/22 | 3.2 sn |
| DEV, sonra, 8B thinking | 36/45 | 36/40 | 4 | 0 / 0 | 5 (2 kural hatalı) | 80.0 % | 9/18 | 49.4 sn |
| HOLDOUT, sonra, 8B thinking | 19/28 | 20/23 | 3 | 0 / 0 | 5 (5 kural hatalı) | 71.4 % | 10/13 | 37.8 sn |
| DEV, sonra, 8B quick | 36/45 | 27/29 | 2 | 0 / 0 | 16 (7 kural hatalı) | 60.0 % | 4/18 | 10.9 sn |
| HOLDOUT, sonra, 8B quick | 19/28 | 17/20 | 3 | 0 / 0 | 8 (6 kural hatalı) | 60.7 % | 8/13 | 8.0 sn |

## 4. Hüküm okuma hakemliği (HOLDOUT, dondurulmuş, 6658175)

- **clause-holdout-quick**: 28/112 hüküm eskale (%25.0); kural hatası 7, eskale edilen 1; model doğruluğu (eskale edilenlerde) 85.7 %; inceleme 5 (isabet 20.0 %); hüküm başına 0.6 sn.
- **clause-holdout**: 28/112 hüküm eskale (%25.0); kural hatası 7, eskale edilen 1; model doğruluğu (eskale edilenlerde) 100.0 %; inceleme 1 (isabet 100.0 %); hüküm başına 3.7 sn.

## 5. Gerçek korpus (2.55x yükümlülük × 3 pilot)

| Pilot | Satır | Eskale satır (önce → sonra) | Eskale yükümlülük | Hakemlik çağrısı | Başarısız çağrı | İnceleme satırı | Katalog kapsamıyla uygulanan | Model süresi toplam | Yükümlülük başına | Çağrı başına (medyan / en uzun) |
|---|---|---|---|---|---|---|---|---|---|---|
| tr-bev-pilot-alcohol-integrated (corpus) | 3218 | 202 → 616 (19.1 %) | 344 (13.5 %) | 0 | 0 | 10 | 666 | 0 sn | 0.00 sn | 0.0 / 0.0 sn |
| tr-bev-pilot-non-alcohol-bottler (corpus) | 3409 | 344 → 1065 (31.2 %) | 435 (17.1 %) | 0 | 0 | 14 | 929 | 0 sn | 0.00 sn | 0.0 / 0.0 sn |
| tr-bev-pilot-alcohol-import (corpus) | 2102 | 163 → 518 (24.6 %) | 312 (12.2 %) | 0 | 0 | 11 | 531 | 0 sn | 0.00 sn | 0.0 / 0.0 sn |
| tr-bev-pilot-alcohol-integrated (corpus-adjudicate-quick) | 3218 | 202 → 616 (19.1 %) | 344 (13.5 %) | 370 | 0 | 292 | 666 | 2429 sn | 0.95 sn | 5.6 / 20.0 sn |
| tr-bev-pilot-non-alcohol-bottler (corpus-adjudicate-quick) | 3409 | 344 → 1065 (31.2 %) | 435 (17.1 %) | 473 | 0 | 399 | 929 | 2804 sn | 1.10 sn | 4.5 / 18.9 sn |
| tr-bev-pilot-alcohol-import (corpus-adjudicate-quick) | 2102 | 163 → 518 (24.6 %) | 312 (12.2 %) | 320 | 0 | 222 | 531 | 1774 sn | 0.70 sn | 4.9 / 21.8 sn |


**Korpus bulguları (3e36ebf):**
- Gecikme: bira pilotunda 8B quick ile 370 çağrı, toplam 40 dk; çağrı başına 6,9 sn (medyan 5,6; en uzun 20), yükümlülük başına 0,95 sn; başarısız çağrı 0; doğrulanamayan satır 0. Eski motor 167–316 sn/satırdı.
- Eskalasyon oranı v2 ile 2–3 katına çıktı (satırların %19 / %31 / %25'i; yükümlülüklerin %13,5 / %17,1 / %12,2'si) ve satır bazında %5–15 hedefinin dışına taştı. Sebep: çapalı düşük eşik (sem ≥ 0,55 + 2 ortak kök) korpusta çok daha fazla çift kabul ediyor. Ölçülmüş alternatif: 3 ortak kök (HOLDOUT'ta altın 7/9, diğer çift 3/550). Değiştirilmedi; VALIDATION bir kez ölçülmüş set olarak kalsın diye.
- 8B quick önerilerinin 223/370'i CONTRADICTED; hiçbiri karar olmadı (inceleme bayrağı). 146 satır modelin tek başına verdiği PARTIAL ile otomatik PARTIALLY_COVERED oldu (SEMANTIC_ADJUDICATED); bu davranış etiketli setlerde doğruydu, korpus ölçeğinde doğrulanmadı.
- `corpus-adjudicate` (8B thinking) tamamlanmadı; bittiğinde satırları `out/queue.log`'dadır.

## 6. Regresyon

backend/tests (1640 tests): 242 bilinen hata; identities and reasons identical to the recorded baseline (FileNotFoundError artefacts 239, ModuleNotFoundError 1, AssertionError 1, argument error 1).

## 7. Yanlış otomatik kararlar ve incelemeye gidenler (3e36ebf, 8B thinking)

### adjudicate-validation
- İNCELEME COVV-01: beklenen COVERED, kural NOT_COVERED, karar NOT_COVERED, öneri COVERED; MODEL_PROPOSES_COVERED; MODEL_PROPOSES_COVERED
- YANLIŞ COVV-07: beklenen CONTRADICTED, kural NOT_COVERED, karar NOT_COVERED, öneri —; RULE_ONLY; —
- İNCELEME COVV-08: beklenen COVERED, kural NOT_COVERED, karar NOT_COVERED, öneri COVERED; MODEL_PROPOSES_COVERED; MODEL_PROPOSES_COVERED
- İNCELEME COVV-11: beklenen COVERED, kural NOT_COVERED, karar NOT_COVERED, öneri COVERED; MODEL_PROPOSES_COVERED; MODEL_PROPOSES_COVERED
- YANLIŞ COVV-13: beklenen COVERED/PARTIALLY_COVERED, kural NOT_COVERED, karar NOT_COVERED, öneri —; RULE_ONLY; —
- İNCELEME COVV-14: beklenen COVERED/PARTIALLY_COVERED, kural PARTIALLY_COVERED, karar PARTIALLY_COVERED, öneri COVERED; MODEL_PROPOSES_COVERED; MODEL_PROPOSES_COVERED
- İNCELEME COVV-19: beklenen COVERED, kural NOT_COVERED, karar NOT_COVERED, öneri COVERED; MODEL_PROPOSES_COVERED; MODEL_PROPOSES_COVERED
- İNCELEME COVV-24: beklenen CONTRADICTED, kural NOT_COVERED, karar NOT_COVERED, öneri CONTRADICTED; MODEL_CONFLICT_UNCONFIRMED; MODEL_CONFLICT_UNCONFIRMED
- YANLIŞ COVV-25: beklenen COVERED, kural NOT_COVERED, karar PARTIALLY_COVERED, öneri PARTIALLY_COVERED; SEMANTIC_ADJUDICATED; —
- YANLIŞ COVV-28: beklenen PARTIALLY_COVERED, kural CONTRADICTED, karar CONTRADICTED, öneri —; RULE_ONLY; —
- YANLIŞ COVV-29: beklenen COVERED, kural NOT_COVERED, karar NOT_COVERED, öneri —; RULE_ONLY; —
- İNCELEME COVV-31: beklenen COVERED, kural NOT_COVERED, karar NOT_COVERED, öneri COVERED; MODEL_PROPOSES_COVERED; MODEL_PROPOSES_COVERED
- YANLIŞ COVV-34: beklenen COVERED, kural PARTIALLY_COVERED, karar PARTIALLY_COVERED, öneri PARTIALLY_COVERED; BOTH_READINGS; —
- İNCELEME COVV-36: beklenen COVERED/PARTIALLY_COVERED, kural NOT_COVERED, karar NOT_COVERED, öneri CONTRADICTED; MODEL_CONFLICT_UNCONFIRMED; MODEL_CONFLICT_UNCONFIRMED
- İNCELEME COVV-38: beklenen COVERED/PARTIALLY_COVERED, kural NOT_COVERED, karar NOT_COVERED, öneri UNKNOWN; RULE_ONLY; ADJUDICATION_UNRESOLVED
- İNCELEME COVV-39: beklenen COVERED, kural NOT_COVERED, karar NOT_COVERED, öneri COVERED; MODEL_PROPOSES_COVERED; MODEL_PROPOSES_COVERED
- İNCELEME COVV-41: beklenen COVERED, kural PARTIALLY_COVERED, karar PARTIALLY_COVERED, öneri COVERED; MODEL_PROPOSES_COVERED; MODEL_PROPOSES_COVERED

### adjudicate-dev
- İNCELEME COV-09: beklenen NOT_COVERED, kural NOT_COVERED, karar NOT_COVERED, öneri CONTRADICTED; MODEL_CONFLICT_UNCONFIRMED; MODEL_CONFLICT_UNCONFIRMED
- İNCELEME COV-12: beklenen PARTIALLY_COVERED, kural NOT_COVERED, karar NOT_COVERED, öneri COVERED; MODEL_PROPOSES_COVERED; MODEL_PROPOSES_COVERED
- YANLIŞ COV-14: beklenen COVERED, kural NOT_COVERED, karar PARTIALLY_COVERED, öneri PARTIALLY_COVERED; SEMANTIC_ADJUDICATED; —
- İNCELEME COV-15: beklenen COVERED/PARTIALLY_COVERED, kural PARTIALLY_COVERED, karar PARTIALLY_COVERED, öneri CONTRADICTED; MODEL_CONFLICT_UNCONFIRMED; MODEL_CONFLICT_UNCONFIRMED
- YANLIŞ COV-16: beklenen COVERED, kural NOT_COVERED, karar NOT_COVERED, öneri NOT_COVERED; BOTH_READINGS; —
- İNCELEME COV-25: beklenen NOT_COVERED, kural NOT_COVERED, karar NOT_COVERED, öneri CONTRADICTED; MODEL_CONFLICT_UNCONFIRMED; MODEL_CONFLICT_UNCONFIRMED
- YANLIŞ COV-30: beklenen COVERED, kural PARTIALLY_COVERED, karar PARTIALLY_COVERED, öneri PARTIALLY_COVERED; BOTH_READINGS; —
- YANLIŞ COV-40: beklenen COVERED, kural PARTIALLY_COVERED, karar PARTIALLY_COVERED, öneri PARTIALLY_COVERED; BOTH_READINGS; —
- İNCELEME COV-43: beklenen COVERED, kural NOT_COVERED, karar NOT_COVERED, öneri COVERED; MODEL_PROPOSES_COVERED; MODEL_PROPOSES_COVERED

### adjudicate-holdout
- İNCELEME COVH-06: beklenen COVERED, kural NOT_COVERED, karar NOT_COVERED, öneri COVERED; MODEL_PROPOSES_COVERED; MODEL_PROPOSES_COVERED
- İNCELEME COVH-15: beklenen COVERED, kural NOT_COVERED, karar NOT_COVERED, öneri COVERED; MODEL_PROPOSES_COVERED; MODEL_PROPOSES_COVERED
- YANLIŞ COVH-17: beklenen COVERED/PARTIALLY_COVERED, kural NOT_COVERED, karar NOT_COVERED, öneri NOT_COVERED; BOTH_READINGS; —
- İNCELEME COVH-19: beklenen COVERED, kural NOT_COVERED, karar NOT_COVERED, öneri COVERED; MODEL_PROPOSES_COVERED; MODEL_PROPOSES_COVERED
- İNCELEME COVH-20: beklenen CONTRADICTED, kural NOT_COVERED, karar NOT_COVERED, öneri CONTRADICTED; MODEL_CONFLICT_UNCONFIRMED; MODEL_CONFLICT_UNCONFIRMED
- İNCELEME COVH-21: beklenen COVERED, kural NOT_COVERED, karar NOT_COVERED, öneri COVERED; MODEL_PROPOSES_COVERED; MODEL_PROPOSES_COVERED
- YANLIŞ COVH-25: beklenen NOT_COVERED, kural NOT_COVERED, karar PARTIALLY_COVERED, öneri PARTIALLY_COVERED; SEMANTIC_ADJUDICATED; —
- YANLIŞ COVH-26: beklenen COVERED, kural NOT_COVERED, karar PARTIALLY_COVERED, öneri PARTIALLY_COVERED; SEMANTIC_ADJUDICATED; —


## 8. Devam (1 Ekim gece): sayısal motor, model-only PARTIAL, eskalasyon eşiği, çok motorlu katman

- **Sayısal motor (tr-compare-rules-v3):** yasak biçiminde yazılmış limit ("pH 4,8'i geçemez", "100 mg/L'yi geçemez") artık olumsuzlama değil limit; metinde adı geçen nitelik ("kinin miktarı", "karbondioksit") limitin niteliği; pH kendi birimi; niteliği çözülemeyen limit kesin karar vermez (UNCLEAR → inceleme); art arda limitlerde ("etil alkol 3,0 g/L, laktik asit 0,6 g/L") sözlük niteliği sonraki sayıya yapışmıyor. Yalnız kurallar, VALIDATION: 19 → **22/42**; yanlış CONTRADICTED 1 → **0**; DEV 36, HOLDOUT 19 değişmedi. Korpus hüküm anlık görüntüsü değişmedi.
- **Model-only PARTIAL:** artık öneri + REVIEW_REQUIRED (MODEL_PROPOSES_PARTIAL); karar kuralın sözünde kalıyor, kanıt satırda.
- **Eskalasyon eşiği:** kaçırılan altın ifade ve inceleme yükü birlikte ölçüldü (korpus tablosu, model yok): 2 kök → bira %19,1 / alkolsüz %31,4 satır (HOLDOUT altın 8/9); **3 kök → %9,7 / %17,0 (7/9)**; çapasız → %6,4 / %10,1 (6/9). Üç kök seçildi.
- **Kayıtlı koşunun (2 kök, run 8) bugünkü kodla tekrar oynatılması:** DEV otomatik 30 doğru / 2 yanlış, inceleme 13; HOLDOUT 19 / 2, inceleme 7; VALIDATION 21 / 5, inceleme 16; yanlış COVERED / CONTRADICTED her sette 0 / 0. Üç kökle aday kümesi değiştiği için bazı eskale vakaların kaydı yok; bunlar incelemeye gider (ADJUDICATION_UNRESOLVED), asla geçmez. Taze bir model koşusu `scripts/Test-TR.ps1 -WithModel` ile alınır.
- **Çok motorlu katman (`tr/engines.py`):** profilin düştüğü her sektör paketi için bir SectorEngine (paketin taşıdığı mevzuat, çıkarım, yönlendirme, değerlendirme), ortak ExpertServices (korpus, sözlük, kayıtlı benzerlikler, hakem, doğrulama). Mevcut fonksiyonlar değişmedi; CLI: `python -m regchain.tr assess --profile ...`.
- **Doğrulama seti bir kez ölçüldü** kuralı bu geliştirmelerle bozuldu (hataları okunarak düzeltildi); sonraki doğruluk iddiası için yeni, bağımsız uzman etiketli örnekler gerekir.

## 9. Devam (2 Ekim): gerçek kamuya açık belgeler, karşılaştırıcı v4, satış kanalı ve ambalaj boyutları

- **Gerçek belgeler (test girdisi, depoya alınmadı; `scripts/tr_external_test.py`):** iki bira üreticisinin etik/yönetim-sistemi belgeleri ve bir şişeleyicinin insan hakları belgesi, kamuya açık PDF'lerden numaralı cümlelere çevrilip pilot profillerin kaydı olarak değerlendirildi (model yok). İlk okuma: bira pilotunda **26 CONTRADICTED, hepsi yanlış** (rüşvet maddesindeki "hediye, ağırlama" bedelsiz dağıtım yasağına; STK projelerine "sponsor olabilir" marka sponsorluğu yasağına; "çeşitliliğe değer verir" "pH değeri"ne; "alkollü araç kullanmamalıdırlar" "bira suyu mevzuata uygun olmalıdır"a karşı okundu) ve kalite politikası maddelerinin genel yükümlülükleri "kapsadığı" iddiası. Şişeleyici belgesinde ilgili satır 0 (belge konusu dışı; doğru).
- **tr-compare-rules-v4 (4745065):** izin ifadesi mutlak yasakla ancak yasağın konusu hakkındaysa çelişir (ürünleri, şirketin markası/logosu/ticari unvanı ya da en az iki ayırt edici sözcük); olumsuzlama için yükümlülüğün sözcüklerinin çoğu, en az üçü ve yükümlülüğün yönetmediği bir fiil olmaması gerekir; tek ayırt edici sözcüklü yükümlülük yalnız öğeleriyle ilişkilenir; mevzuat kalıp sözcükleri ("gerekli, uygun, mevzuat, prosedür, esas, değer, yönetmelik, madde, sağlanır") ayırt edici sayılmaz. Yalnız kurallar: DEV 36, HOLDOUT 19, VALIDATION 22 (değişmedi); korpus CONTRADICTED bira 23 / alkolsüz 1 / ithalatçı 7 (değişmedi; marka kuralı olmadan gerçek sponsorluk çelişkisi COVH-02 kayboluyordu, 7 → 1). Gerçek belgelerin v4 ile yeniden değerlendirmesi kullanıcı tarafında koşulacak (ilk v4 koşusu 30 dakikada bitmedi; `compare.passage_frames` artık önbellekli).
- **Satış kanalı ve ambalaj boyutları (adım 3):** `ObligationScope.sales_channels` yeni boyut (LEGAL_ENTITY / FACILITY / ACTIVITY); satışı yöneten ya da mesafeli satış konulu hükümdeki kanal sözcükleri (bakkal, büfe → TRADITIONAL_RETAIL; süpermarket, zincir market → MODERN_RETAIL; mesafeli sözleşme, e-ticaret, internet üzerinden satış → ONLINE) hükmü o kanallardan satan tüzel kişi/faaliyete bağlar; hiçbirinden satmayan bağlı değil, kanalı belirtilmemiş UNKNOWN. Belge adındaki ("toptan satış belgesi") ya da web adresindeki ("Kurumun internet sitesinde") kanal sözcüğü kanal değildir; istisna içindeki sözcük sayılmaz. Ambalaj (depozito → RETURNABLE, cam şişe → GLASS, pet/plastik → PET, metal/alüminyum/teneke kutu → CAN) yalnız ürün düzeyindeki hükümlerde `product_attributes` olur; tüzel kişi düzeyinde uygulanınca 5 etiketli uygulanabilirlik vakası PARTIAL'a kaydığı için daraltıldı. Korpus: kanal taşıyan hüküm kapsamı **83** (Kanun 6563: 36, Yönetmelik 20237: 34, Kanun 6502: 9, Yönetmelik 20435: 2, 14646 md. 22/4: 1, 15592 md. 11/3: 1), ambalaj taşıyan **4** (7510 md. 34/4-c ve 23282 md. 19/1 cam; 38745 md. 14/3 ve 15/4 depozito). Etiketli setler ve regresyon değişmedi (TR 343 test, tam paket 242 başarısızlık kimlik+neden aynı). Bilinen sınır: 15592 md. 11/3'te "büfe" bir gıda işletmesi türüdür, kanal değil; satan tüzel kişiye daraltılması sorgulanabilir (1 hüküm).

## 10. Devam (2 Ekim): kurul kararı katmanı (Rekabet Kurulu taahhüt kararı)

- **Ne:** `tr/decisions.py`. Bir yönetmelik bir şirket sınıfını bağlar; bir kurul kararı adını verdiği teşebbüsü bağlar. DECISION_PRECEDENT katmanı paketlere girmez (registry reddeder); karar, profilin muhatap dediği tüzel kişiler için (`LegalEntity.bound_by_decisions` ya da komut satırında `--addressee`) sektör motorlarının yanında okunur, başka hiçbir tüzel kişiye bağlanmaz. Üretim mantığında şirket adı yok: kayıt kararın kendi kimliğini taşır, muhatabı profil söyler.
- **Akış:** `scripts/tr_decision_import.py` PDF/metni `text.txt + decision.json` olarak saklar (SHA-256; her yüklemede yeniden doğrulanır; kurum `adapters.INSTITUTIONS['REKABET']`, yalnız rekabet.gov.tr) → numaralı taahhüt maddeleri bölüm olur (`commitment_sections`: sıra dışı numara madde açmaz, alt maddeler lead-in altında okunur) → yönetmelik kural okuyucusu (frames/extraction) → her yükümlülük muhatap tüzel kişiye bağlanır (DECISION_ADDRESSEE kapısı; diğer tüzel kişiler DOES_NOT_APPLY / DECISION_ADDRESSEE_MISMATCH; temellendirilemeyen madde UNKNOWN kalır) → aynı karşılaştırıcı, aday ifadeler, hakem ve doğrulama → **her satır REVIEW_REQUIRED (DECISION_PRECEDENT_REVIEW)**: karar okuyucusunun etiketli vakası yok, hiçbir satırı otomatik karar değildir.
- **Okuyucu eklemeleri (genel kurallar, marka/şirket yok):** taahhüt dili gelecek zaman ("açık tutacaktır" MUST, "bağlamayacaktır" MUST_NOT) ve "… taahhüt eder" / "… -mayacağını taahhüt eder"; kurumun gözden geçirme/inceleme/izleme işi (DELEGATION_VERB: gözden geçirilir, incelenir, izlenir) şirket yükümlülüğü değil. `RegulationMeta`: kurul kararı `interprets` boş olabilir (kendi dayanağıyla muhatabına yükümlülük yaratır; resmî rehber hâlâ yorumladığı metni adlandırmak zorunda).
- **Korpus etkisi (anlık görüntü farkı, 7 cümle):** 3 OTHER → OBLIGATION (6207 md. 15/18 c.1 "projeler … içerecektir", c.2 "onayı aranacaktır"; 7510 md. 28/7 "araç ve gereçler … yapılmış olacaktır" — üçü gerçek yükümlülük), 2 OBLIGATION → DELEGATION (15594 md. 6/2 "Yetkili merci tarafından … incelenir", 4821 md. 8/1 c.3 "… incelenir" — kurum işi), 2 yalnız işaret değişimi. Yükümlülük sayısı 2550 → 2551. Etiketli setler değişmedi.
- **Sentetik gösterim (`tests/test_tr_decisions.py`, gerçek karar değil):** 5 madde → 2 OBLIGATION, 2 PROHIBITION (biri "Ancak … uygulanmaz" istisnasıyla), 1 DELEGATION (Kurul'un gözden geçirmesi); muhatap NONALC-SALES için 4 satır, hepsi NOT_COVERED / NO_EVIDENCE / RULE_ONLY, inceleme nedeni DECISION_PRECEDENT_REVIEW, doğrulanmayan 0; diğer dört tüzel kişi DOES_NOT_APPLY. Komut: `python -m regchain.tr decisions --decision <id> --profile <profil> --addressee <tüzel kişi> [--decisions-root …] [--out …]`.
- **Gerçek karar (CCSD, 04.06.2026, 26-20/614-243):** taahhüt metni PDF'i depoda yok; kullanıcı `evaluation/external/pdf/` altına indirince `scripts/tr_decision_import.py` ile saklanır ve aynı komutla değerlendirilir. Bu adımın gerçek-belge sonucu henüz ölçülmedi.

## 11. Devam (2 Ekim): QDMS dışa aktarımı (mevzuat → Cardaman gerekçesi → etkilenen politika/kontrol → insan onayı → QDMS eylemi)

- **Ne:** `tr/qdms.py` ve `python -m regchain.tr qdms export|approve`. Her boşluk satırı bir QDMS satırı olur: tüzel kişi (ürün satırında ürünü işleyen tüm tüzel kişiler), hüküm (metin, sürüm, bağlayıcılık, alıntı), uygulanabilirlik (durum, dayanak), gerekçe (Cardaman kararı AUTO/REVIEW_REQUIRED, kapsama, dayanak, inceleme nedenleri, model önerisi, ilgili cümleler alıntılarıyla), etkilenen süreç (sorumlu birimler, yürürlükteki/dayanılan belgeler, kontroller), politika durumu, gereken eylemler, kanıt (dönem içi/dışı) ve insan onayı.
- **Eylem sözlüğü (genel, ürün bağımsız):** çelişki → DOCUMENT_CHANGE_REQUEST/REVISE_CONFLICTING_STATEMENT (hedef: çelişen cümle ve belgesi, öncelik HIGH); kapsanmıyor → DOCUMENT_CHANGE_REQUEST/ADD_STATEMENT; kısmi → COMPLETE_STATEMENT; kontrol yok → CONTROL_DEFINITION_REQUEST; kanıt yok → EVIDENCE_REQUEST; belirsiz → COMPLIANCE_REVIEW_TASK; ürün değeri limit dışı → NONCONFORMITY. İncelemeye giden satırda ilk eylem her zaman COMPLIANCE_REVIEW_TASK.
- **Onay kapısı:** dışa aktarımda her eylem DRAFT, her satır PENDING; hiçbir şey bir yere gönderilmez (dosyalar: export.json, export.csv UTF-8 BOM ve noktalı virgül, approvals-template.json, ready-actions.json). Onay dosyasındaki her karar satırın parmak izine bağlıdır: gerekçe, kapsama, durum, eylemler ya da metnin sürümü değiştiyse onay STALE_APPROVAL olarak reddedilir ve satır PENDING kalır; bilinmeyen satır ve aynı satıra iki karar da reddedilir; onaylayan adı zorunlu. Yalnız APPROVED satırların eylemleri READY_FOR_QDMS olur ve `ready-actions.json`a girer; REJECTED → CANCELLED; RETURNED → DRAFT.
- **Ölçülen (sentetik şişeleyici kaydı, kurallar + kayıtlı servis yok, 1 dk 3 s):** 3348 düzenleme satırı + sentetik kararın 4 satırı = 3352 satır; politika durumu CONTRADICTED 1, NOT_COVERED 3326, PARTIALLY_COVERED 21, UNKNOWN 4; Cardaman kararı AUTO 3188, REVIEW_REQUIRED 164; 3514 eylem, hepsi DRAFT, hazır eylem 0. Onay denemesi: çelişen satır (Tebliğ 23706 md. 5/1-b, enerji içeceği şartnamesi, LIMIT_WEAKER) onaylandı → 1 eylem READY_FOR_QDMS (Ar-Ge, SPEC-ENERGY-01#1); bir satır reddedildi → CANCELLED; yanlış parmak izli onay STALE_APPROVAL.
- **Bilinen sınırlar:** (1) her yükümlülük × hedef ayrı bir değişiklik talebi: sentetik kayıtta 3326 ADD_STATEMENT taslağı, gerçek bir QDMS'e bu hâliyle verilmez; belge/birim bazında gruplama gerekir. (2) Belirli bir QDMS ürününün içe aktarma biçimine eşleme ve gönderim yapılmadı (entegratörün adımı). (3) Ürün satırının tüzel kişileri profildeki `product_ids`den gelir; ürünü işleyen her tüzel kişi listelenir.
- **Kendi testiniz:** `scripts/Test-TR.ps1` artık 4b adımında şişeleyici pilotunun QDMS dışa aktarımını da üretir (`4b-qdms/`), özet satırı SUMMARY.txt'de.
