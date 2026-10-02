# Uygulanabilirlik güvenliği — teslim raporu (2–3 Ekim 2026)

Kapsam: kanal tamlığı hatası, UNKNOWN uygulanabilirliğin görünürlüğü, satır sayısı uzlaştırması, HOLDOUT ölçüm ayrımı.
Yeni özellik, model değişikliği, yeni mevzuat ailesi ve canlı QDMS bağlantısı yok. Etiketler INDICATIVE; sentetik pilot
profilleri ve sentetik politika kayıtları kullanıldı. Tüm commit'ler yereldir, push edilmedi.

Kod sürümleri: aa60e92 (düzeltmeler), 27dfc7a (ölçüm betikleri), 7188feb (inceleme kayıtlarında kapı kanıtı).
Karşılaştırılan sürümler: 2c3d56a (1 Ekim self-test), 39a9659 (2 Ekim self-test). Girdiler üç sürümde aynı (korpus,
pilot profilleri, sentetik kayıtlar, `recorded/20261001` benzerlik tablosu ve hakem kayıtları; `git diff` boş). Model
çağrılmadı: hakem kararları kayıttan yeniden oynatıldı.

## 1. Doğrulanan kök nedenler

| # | Bulgu | Doğrulama | Durum |
|---|---|---|---|
| 1 | Kanal listesi, genel profil tamlık bayrağından (`profile_complete`) "tam" sayılıyordu; faaliyet kanal listesi her zaman "tam" sayılıyordu | `routing._entity_facts` ve `_targets` (ACTIVITY) kodu; yeni testler düzeltmeden önce 2 başarısız + 8 hata verdi | düzeltildi (aa60e92) |
| 2 | UNKNOWN uygulanabilirlik gap raporuna ve QDMS'e hiç girmiyordu | `assess_profile` / `compare_profile` yalnız APPLIES/PARTIAL alıyordu; 39a9659'da 2427 / 7319 / 3091 UNKNOWN karar hiçbir çıktıda yoktu | düzeltildi (aa60e92) |
| 3 | "mesafeli sözleşme" ONLINE kanalı sayılıyordu | 20237 md. 6/3 ("sesli iletişim yoluyla") ve md. 8/2 ("telefonla aranması") mesafeli sözleşmenin telefonla da kurulduğunu söylüyor | sözlükten çıkarıldı |
| 4 | "depozito" iade edilebilir ambalaj (RETURNABLE_PACKAGING) sayılıyordu | 38745 md. 14/3 ve 15/4 "depozito yönetim sistemine dâhil olan ambalajlar": sistem tek kullanımlık kutu/PET'i de kapsar; tam da bu ürünler dışlanıyordu | sözlükten çıkarıldı |
| 5 | "büfe" satış kanalı sayılıyordu | 15592 md. 11/3: "Çadır, büfe ve seyyar satış araçları gibi … gıda işletmeleri" — işletme türü | sözlükten çıkarıldı |
| 6 | Bira pilotundaki "3218 − 96 + 2 = 3124, raporda 3172" farkı | 96 iki paketin toplamıydı (alkol 48 + alkolsüz 48); paket bazında 3218 − 48 + 2 = 3172 ve 2462 − 48 = 2414 | sayım birimi hatası (raporda); kod hatası değil |
| 7 | Aynı yükümlülük×hedef iki paketten gelince QDMS'te iki satır oluyordu | bira pilotunda alkolsüz paketin her satırı alkol paketindeki bir satırın kopyası (5586 ham satır = 3172 benzersiz eşleşme) | birleştirildi + kayıt defteri (aa60e92) |
| 8 | İnceleme kayıtlarında eksik bilginin ayrıntısı yoktu (yalnız kapı adı) | `pilot.applicability.gate` kanıtı `evidence` altında tutuyor; 27dfc7a self-test'inde görüldü | düzeltildi (7188feb) |

Doğrulanıp doğru bulunan dışlamalar: cam şişe koşulu (23282 md. 19/1 "tekrar kullanılabilen cam şişeler"; 7510 md. 34/4-c
"Cam ambalajlarda … şişe üzerine baskı"), elektronik ticaret (6563: online kanal gerçek koşul), 15594 md. 6/2 ("Yetkili
merci tarafından başvuru … incelenir": idarenin işi, şirket yükümlülüğü değil).

## 2. Kanal tamlığı: yeni davranış

- Alanlar: `LegalEntity.sales_channels_complete` + `sales_channels_basis`, `Activity.channels_complete` + `channels_basis`
  (şemadaki `tags_complete`, `products_complete`, `activities_complete` adlandırmasıyla uyumlu). Varsayılan `False`.
- Listede adı geçen kanal: biliniyor (liste eksik olsa da). Listede olmayan kanal: yalnız liste açıkça tamsa "yok".
- Liste açık ve kanal hükmü belirliyorsa: UNKNOWN (`PROFILE_INCOMPLETE`, kanal kapısı UNDETERMINED).
- Başka bağımsız ve karara bağlanmış kapı (ör. tam faaliyet listesinde faaliyet yok) hükmü dışlıyorsa: DOES_NOT_APPLY,
  gerekçesi o kapı; kanal kapısı UNDETERMINED kalır.
- Çelişki (tam liste + listede olmayan bir kanalı adlandıran faaliyet kaydı, ya da ECOMMERCE_SALE faaliyeti varken ONLINE
  yok): `SALES_CHANNEL_PROFILE_CONFLICT`, UNKNOWN; hiçbir zaman dışlama değil.
- Faaliyetin açık listesi, tüzel kişinin tam listesiyle sınırlanır (tüzel kişi kullanmadığını söylüyorsa faaliyet de kullanamaz).
- Eski profiller "tam" diye taşınmadı. Pilot profillerin hiçbirine kanal tamlığı eklenmedi: sentetik listelerin tam
  olduğuna dair bir dayanak yok. Test: tam işaretli her listede dayanak alanı dolu olmalı.

## 3. Değiştirilen dosyalar

| Dosya | Değişiklik |
|---|---|
| `backend/src/regchain/tr/profile.py` | kanal tamlık ve dayanak alanları |
| `backend/src/regchain/tr/routing.py` | `_channel_facts`, `IMPLIED_CHANNELS`, çelişki kodu, faaliyet tavanı |
| `backend/src/regchain/tr/data/lexicon.json` | mesafeli sözleşme → ONLINE, depozito → RETURNABLE, büfe → TRADITIONAL_RETAIL çıkarıldı |
| `backend/src/regchain/tr/compare.py` | `ApplicabilityReview`, `applicability_review`, `review_type`, `flat_gate`, `GapReport.applicability_reviews` |
| `backend/src/regchain/tr/adjudicate.py` | `assess_profile` UNKNOWN kayıtlarını toplar; istatistiklere `applicability_unknown*` |
| `backend/src/regchain/tr/decisions.py` | karar katmanında temellendirilemeyen madde de inceleme kaydı |
| `backend/src/regchain/tr/qdms.py` | ayrı uygulanabilirlik bölümü (`PROFILE_DATA_REQUEST` / `APPLICABILITY_REVIEW_TASK`), paketler arası birleştirme, kayıt defteri, `applicability-reviews.csv` |
| `backend/src/regchain/tr/__main__.py` | `assess --out` inceleme kayıtlarını da yazar |
| `backend/scripts/tr_selftest_summary.py` | özet satırına UNKNOWN sayıları |
| `backend/scripts/tr_rows_dump.py`, `tr_holdout_ab.py`, `tr_reconcile.py`, `tr_dropped_rows.py` | ölçüm betikleri (yeni) |

## 4. Eklenen testler ve sonuçları

| Test | Kapsam |
|---|---|
| `tests/test_tr_channel_completeness.py` (10) | eksik alan, boş liste (tam/açık), kısmi liste, tam liste, açık kanal varlığı, bağımsız kapsam dışılık, çelişkili profil (2 biçim), faaliyet listesi ve tavanı, pilot profillerinde dayanaksız tamlık yok |
| `tests/test_tr_extraction.py` `ExclusionAgainstSourceTests` (4) | mesafeli sözleşme ≠ online; depozito ≠ iade edilebilir; büfe kanal değil; cam koşulu korunur. Düzeltmeden önce 3 başarısız |
| `tests/test_tr_extraction.py` kanal testi (güncellendi) | `profile_complete` kanal listesini tamlamaz; tam liste o kapıyla dışlar |
| `tests/test_tr_qdms.py` `UnknownApplicabilityTests` (3) | yönlendirmedeki her UNKNOWN karar → rapor → istatistik → QDMS inceleme bölümü → CSV → onay; politika eylemi yok; eksik bilgide `required` ve `complete` var |
| `tests/test_tr_qdms.py` `LedgerTests` (1) | iki paket → tek satır; her analiz satırı kayıt defterinde bir kez; sayılar tutar |

Sonuçlar (kullanıcının koştuğu self-test, commit 7188feb, `evaluation/selftest/20261002-193234`):

| Paket | Çalışan | Başarılı | Başarısız | Hata | Atlanan |
|---|---|---|---|---|---|
| TR | 378 | 378 | 0 | 0 | 0 |
| Tam | 1692 | 1377 | 2 | 240 | 73 |

Tam paketin 242 başarısız/hatası v018/v019 modüllerinde (239'u `FileNotFoundError`: kurtarılamayan eski artefaktlar);
TR modüllerinde 0. Kimlik ve neden olarak kayıtlı baseline ile aynı.

## 5. Satır sayısı uzlaştırması

Birimler: ham satır = (paket, yükümlülük, hedef), self-test'in "rows" sayısı, paket başına. Bira pilotunda iki paket aynı
yükümlülük×hedef eşleşmesini ayrı ayrı sayar.

| Pilot / paket | 1 Eki (2c3d56a) | 2 Eki (39a9659) | şimdi (7188feb) | 1 Eki'den düşen | eklenen | denetim |
|---|---|---|---|---|---|---|
| Bira / alkol | 3218 | 3172 | 3209 | 11 | 2 | 3218 − 11 + 2 = 3209 |
| Bira / alkolsüz | 2462 | 2414 | 2451 | 11 | 0 | 2462 − 11 = 2451 |
| Şişeleyici | 3409 | 3379 | 3396 | 14 | 1 | 3409 − 14 + 1 = 3396 |
| İthalatçı | 2102 | 2088 | 2100 | 4 | 2 | 2102 − 4 + 2 = 2100 |

2 Ekim'de düşen ama şimdi geri gelen satırların hepsi (bira 37 + 37, şişeleyici 17, ithalatçı 12) yanlış sözlük eşlemelerinden
(depozito, mesafeli sözleşme, büfe) dışlanmıştı. 1 Ekim'e göre hâlâ dışarıdaki 40 satır (`dropped-rows-0110-to-7188feb.csv`,
her satırda hüküm, kapsam koşulu, profil kanıtı ve metin):

| Nereye gitti | Satır | Hüküm | Gerekçe |
|---|---|---|---|
| UNKNOWN incelemesi | 21 | Kanun 6563 (e-ticaret) | online kanal koşulu; pilot kanal listeleri açık, ONLINE yok → bilinmiyor |
| UNKNOWN incelemesi | 1 | Yönetmelik 23282 md. 19/1 | cam şişe koşulu; ürünün etiket listesi tam değil |
| DOES_NOT_APPLY | 11 | Yönetmelik 23282 md. 19/1 | cam şişe koşulu; ürün etiketleri tam, cam yok |
| DOES_NOT_APPLY | 1 | Yönetmelik 7510 md. 34/4-c | cam ambalaj koşulu; su ürünü yalnız PET, liste tam |
| Yükümlülük değil | 6 | Yönetmelik 15594 md. 6/2 | "Yetkili merci tarafından … incelenir": idarenin işi |

Eklenen 5 satır: Yönetmelik 6207 md. 15/18 (c.1–2) ve 7510 md. 28/7 — gelecek zamanla yazılmış yükümlülükler ("içerecektir",
"aranacaktır", "olacaktır"). Aynı anahtarda durumu değişen 3 satır (7510 md. 17/1; 34147 md. 8/1-c iki hedef):
NOT_COVERED → PARTIALLY_COVERED, üçü de REVIEW_REQUIRED; neden karşılaştırıcı v4 (4745065'te başlıyor, 6264324'te yok),
kanal/ambalaj değil.

Önce/sonra yönlendirme durumları (tüm yükümlülük×hedef kararları, iki paket ayrı):

| Pilot | sürüm | APPLIES | PARTIAL | DOES_NOT_APPLY | UNKNOWN |
|---|---|---|---|---|---|
| Bira | 1 Eki | 5170 | 510 | 10612 | 7319 |
| Bira | 2 Eki | 5074 | 512 | 10706 | 7319 |
| Bira | şimdi | 5660 (APPLIES+PARTIAL, iki paket) | | ölçülmedi | 7331 |
| Şişeleyici | 1 Eki | 3082 | 327 | 4465 | 3088 |
| Şişeleyici | 2 Eki | 3051 | 328 | 4487 | 3091 |
| Şişeleyici | şimdi | 3396 (APPLIES+PARTIAL) | | ölçülmedi | 3095 |
| İthalatçı | 1 Eki | 2092 | 10 | 1370 | 2424 |
| İthalatçı | 2 Eki | 2078 | 10 | 1384 | 2427 |
| İthalatçı | şimdi | 2100 (APPLIES+PARTIAL) | | ölçülmedi | 2428 |

## 6. UNKNOWN inceleme ve dışa aktarım sayıları (7188feb)

| Pilot / paket | UNKNOWN | kapsam belirsiz (CLARIFY_REGULATORY_SCOPE) | eksik profil bilgisi (COMPLETE_PROFILE_FACT) | kanal kapısı açık | çelişki |
|---|---|---|---|---|---|
| Bira / alkol | 3959 | 2916 | 1043 | 44 | 0 |
| Bira / alkolsüz | 3372 | 2640 | 732 | 43 | 0 |
| Şişeleyici | 3095 | 2200 | 895 | 6 | 0 |
| İthalatçı | 2428 | 1458 | 970 | 41 | 0 |

Eksik bilgi kayıtlarının tamamı hükmün istediğini, profilin söylediğini ve listenin tam olup olmadığını taşıyor (bira 1850/1850,
şişeleyici 893/893, ithalatçı 1006/1006 kapı kaydı).

QDMS (şişeleyici, self-test 4b): analiz 3396 satır = 3365 dışa aktarılan + 31 `EXCLUDED_COVERED_AUTO` (kapsanmış, otomatik,
eylem yok); 3095 uygulanabilirlik incelemesi ayrı bölümde (`applicability-reviews.csv`), hepsi DRAFT / PENDING; hazır eylem 0;
yinelenen satır kimliği 0. Uygulanabilirlik durumu ile politika/kontrol kapsama durumu ayrı alanlarda; inceleme kayıtlarında
kapsama değerlendirilmez (`coverage_assessed: false`), uygunsuzluk veya belge değişikliği üretilmez.

## 7. HOLDOUT: benzerlik tablosu açık / kapalı

Metrik ("kural kapsaması"): pay = kural karşılaştırıcısının vaka için verdiği sözcüğün (vakanın hükmünün etiketli hedefteki
satırlarından en ağırı: CONTRADICTED > UNKNOWN > NOT_COVERED > PARTIALLY_COVERED > COVERED; satır yoksa NO_ROW) vakanın
beklenen sözcüklerinden biri olduğu vaka sayısı; payda = bölümdeki tüm vakalar. Hakem kararı ve inceleme hesaba katılmaz.

| Bölüm (n) | sürüm | `score_coverage` (tablo yok) | kurallar, tablo yok | kurallar, tablo var |
|---|---|---|---|---|
| DEV (45) | 2c3d56a / 39a9659 / aa60e92 | 36 / 36 / 36 | 36 / 36 / 36 | 36 / 36 / 36 |
| HOLDOUT (28) | 2c3d56a / 39a9659 / aa60e92 | 18 / 18 / 18 | 18 / 18 / 18 | 19 / 19 / 19 |
| VALIDATION (42) | 2c3d56a / 39a9659 / aa60e92 | 21 / 21 / 21 | 21 / 21 / 21 | 22 / 22 / 22 |

7188feb: self-test `ai evaluate` HOLDOUT 0,6429 (18/28, tablo yok); `RecordedPipelineTests` testi tablolu kurallar sayısını
36 / 19 / 22 olarak sabitliyor ve geçti.

18 ile 19 farkı tek vaka: **COVH-16**, Yönetmelik 6203 md. 15/2 @ ALC-IMP-TRADING, beklenen CONTRADICTED. Hüküm aylık satış
raporunu "ayı takip eden ayın en geç 20 nci günü" ister; ifade SOP-LOG-12#2 "takip eden ayın 25'ine kadar" der. Sözcük
örtüşmesi 0,26 (eşik 0,5), kayıtlı benzerlik 0,753 (eşik `SEMANTIC_RELEVANCE` 0,70): son tarih karşılaştırması
(DEADLINE_LATER) yalnız tablo varken çalışır. VALIDATION'daki fark da tek vaka (COVV-16, 15594 md. 6/5: tablo varken CONTRADICTED, yokken NOT_COVERED); mekanizması ayrıca incelenmedi.

Tablonun üretimi: run 8 (commit 3e36ebf), bge-m3@79076464, kosinüs; etiketli her kapsama vakasının hükmündeki yükümlülük
metinleri × o vakanın kaydındaki tüm ifadeler. Beklenen cevaplar okunmaz, ama hangi hükümlerin tabloya gireceğini
değerlendirme vakalarının listesi belirler. Bu yüzden tablolu sayılar bağımsız doğruluk kanıtı değildir; HOLDOUT 3e36ebf'ten
beri geliştirme verisidir ve VALIDATION v3 düzeltmeleri için okunmuştur. Referans cevaplar ve örnekler değiştirilmedi.

## 8. Çözülemeyen veya çalıştırılmayan kontroller

- 7188feb için tam satır dökümü (`tr_rows_dump.py`) ve kayıt defterli QDMS dışa aktarımı bira ve ithalatçı pilotları için
  koşulmadı; şişeleyicinin QDMS'i self-test'ten. "Şimdi" sütununda DOES_NOT_APPLY bu yüzden ölçülmedi.
- Düşen satır tablosunda DOES_NOT_APPLY satırlarının profil kanıtı 2 Ekim dökümünden (kapı adı ve durumu; o dökümün betiği
  `required/stated` alanlarını almıyordu). Bu satırların dışlama kuralı 2 Ekim'den beri değişmedi.
- İnceleme yükü: kapsamı belirsiz her hüküm her tüzel kişi için ayrı kayıt (şişeleyicide 2200). Yükümlülük başına toplama
  yeni bir özellik olur; yapılmadı.
- Pilot profillerinde kanal tamlığı yok (dayanak yok): kanal koşullu hükümler, kanalı listelenmemiş şirketler için UNKNOWN.
  Gerçek müşteri profilinde tamlık ve dayanağı şirketten gelmeli.
- 14646 md. 22/4 ("… bakkal, market … işyerlerinde satışa sunulamaz"): kanal okuması metne dayanıyor, ama bu kanallara satan
  dağıtıcının mı yoksa işyeri işletmecisinin mi bağlandığı çözülmedi (bu bir dışlama değil, daha önce de APPLIES idi).
- Canlı model koşusu yapılmadı (kayıttan yeniden oynatma); gerçek politika belgeleri ve Rekabet Kurulu kararı bu görevin dışında.
- Geçici git worktree'leri `C:\dev\cw-2c3d56a`, `cw-39a9659`, `cw-6264324`, `cw-4745065`, `cw-aa60e92` duruyor.
