# Uçtan uca doğrulama ve UNKNOWN inceleme yükü (2–3 Ekim 2026)

Davranış değişikliği yok: bu görevde kod yalnız ölçüm betiklerinde değişti. Canlı QDMS'e bir şey yazılmadı; çıktılar
yerel dosyalardır. Etiketler INDICATIVE; pilot profilleri ve politika kayıtları sentetiktir.

## 1. Kod ve koşu izlenebilirliği

- Kullanıcının self-test'i commit **7188feb** üzerinde koştu (`evaluation/selftest/20261002-193234`).
- 7188feb → 89a380a: yalnız `evaluation/reports/applicability-safety-20261003/` (README, CSV, JSON). `backend/` farkı 0 satır,
  çalışma ağacı temiz. 89a380a çalışma davranışını değiştirmez.
- Bu görevin ölçümleri 89a380a üzerinde koştu (kod 7188feb ile aynı); karşılaştırma 2c3d56a (1 Ekim self-test) üzerinde aynı
  girdilerle yeniden koşuldu. Her koşunun kaydı: `manifest-head.json`, `manifest-2c3d56a.json` (`backend/scripts/tr_run_manifest.py`).
- Kayıt içeriği: commit, kirli/temiz, Python sürümü, 34 girdi dosyasının SHA-256'sı, 23 saklı kaynağın sürüm kimliği ve
  ayrıştırılmış metin özeti, eşikler, kural sürüm etiketleri, benzerlik tablosu.
- Model: yok. Hakem kararları `recorded/20261001/adjudicate-{dev,holdout,validation}.jsonl` kayıtlarından yeniden oynatıldı.
- Benzerlik tablosu: `recorded/20261001/similarities.json`, SHA-256 `9be4250d…cdbda`, `bge-m3:latest@79076464…`, **91 yükümlülük**
  (yalnız etiketli vakaların hükümleri). Pilot koşularında korpusun geri kalanı tabloya hiç girmez.
- 2c3d56a ile 89a380a arasında girdi farkı: kaynak sürümleri 0, profiller 0, politika kayıtları 0, kayıtlı koşular 0;
  `lexicon.json` ve `vocabulary.json` farklı (bunlar kural verisi, kod değişikliklerinin parçası: kanal/ambalaj kalıpları,
  REKABET kurumu). **Düzeltme:** önceki raporda "girdiler aynı" ifadesi bu iki dosyayı atlıyordu.
- Bulgu: kural sürüm etiketleri davranışı tanımlamıyor. `tr-frames-v1` ve `tr-scope-rules-v1` gelecek zaman, kanal ve ambalaj
  değişikliklerinden sonra da aynı. Davranışı yalnız commit kimliği tanımlıyor.

## 2. Gerçekten çalıştırılan komutlar ve sonuçları

Hepsi `C:\dev\Cardaman\backend` içinden, `PYTHONIOENCODING=utf-8 PYTHONUTF8=1`, çıktı `evaluation/runs/20261003-verification/`
(gitignored). Hepsi çıkış kodu 0.

| Komut | Kod | Sonuç |
|---|---|---|
| `scripts/tr_run_manifest.py manifest-head.json` / `manifest-2c3d56a.json` | 89a380a / 2c3d56a | 34 girdi, 23 kaynak |
| `scripts/tr_rows_dump.py rows-head.json` | 89a380a | ithalatçı 2100 satır / 5899 karar; bira 5660 / 23611; şişeleyici 3396 / 10957 |
| `scripts/tr_rows_dump.py rows-2c3d56a.json` | 2c3d56a | ithalatçı 2102 / 5896; bira 5680 / 23611; şişeleyici 3409 / 10962 |
| `python -m regchain.tr qdms export --profile <pilot> --similarities … --adjudications …×3 --out qdms-<pilot>` | 89a380a | üç pilot, aşağıda |
| `scripts/tr_row_fates.py rows-2c3d56a.json rows-head.json . fates` | — | her satırın akıbeti (`row-fates.csv`) |
| `scripts/tr_unknown_causes.py <pilot> unknown-causes-<pilot>.json` | 89a380a | üç pilot |
| `scripts/tr_unclear_source_signals.py unknown-causes-bottler.json …` | 89a380a | kaynak işaretleri (gösterge) |
| `scripts/tr_review_grouping.py unknown-causes-<pilot>.json …` | — | gruplama önerisi sayıları |

Çalıştırılmayanlar: testler (bu görevde istenmedi; test sonuçları kullanıcının 7188feb self-test'inden), HOLDOUT ölçümü (bu
görevde yeniden koşulmadı; 7188feb self-test'i ve önceki aa60e92 ölçümü), canlı model, canlı QDMS.

## 3. Üç pilot: ayrı ayrı (89a380a)

| | Şişeleyici | Bira (iki paket) | İthalatçı |
|---|---|---|---|
| Yönlendirme kararı (yükümlülük × hedef × paket) | 10957 | 23611 | 5899 |
| APPLIES / PARTIAL | 3068 / 328 | 5148 / 512 | 2090 / 10 |
| DOES_NOT_APPLY | 4466 | 10620 | 1371 |
| UNKNOWN (ham) | 3095 | 7331 | 2428 |
| UNKNOWN (tekil yükümlülük × hedef) | 3095 | 3959 | 2428 |
| Gap satırı (ham) | 3396 | 5660 (3209 + 2451) | 2100 |
| Gap satırı (tekil yükümlülük × hedef) | 3396 | 3209 | 2100 |
| Politika/kontrol kapsaması: NOT_COVERED / PARTIALLY / CONTRADICTED / COVERED / UNKNOWN | 3339 / 21 / 1 / 31 / 4 | 5602 / 22 / 23 / 13 / 0 | 2058 / 15 / 8 / 19 / 0 |
| Otomatik / incelemede (politika satırı) | 3232 / 164 | 5513 / 147 | 2046 / 54 |
| QDMS: dışa aktarılan politika satırı | 3365 | 3199 | 2081 |
| QDMS: eylem gerektirmeyen (kapsanmış, otomatik) | 31 | 13 | 19 |
| QDMS: tekilleştirilen (aynı yükümlülük × hedef, başka paket) | 0 | 2448 politika + 3372 inceleme | 0 |
| QDMS: uygulanabilirlik incelemesi (tekil) | 3095 | 3959 | 2428 |
| QDMS eylemleri / hazır eylem | 3527 DRAFT / 0 | 3302 DRAFT / 0 | 2136 DRAFT / 0 |

Kayıt defteri her pilotta kapanıyor: şişeleyici 3365 + 31 = 3396; bira 3199 + 13 + 2448 = 5660 politika satırı ve
3959 + 3372 = 7331 inceleme; ithalatçı 2081 + 19 = 2100. Yinelenen satır kimliği 0.

## 4. Sayımların kapanışı

Birim: paket başına ham gap satırı (self-test'in "rows" sayısı). Toplam 1 Ekim 11191 (3218 + 2462 + 3409 + 2102), şimdi 11156
(3209 + 2451 + 3396 + 2100), net −35. Aynı birim; fark tam kapanıyor:

| Akıbet | Satır | Açıklama |
|---|---|---|
| Aynı kaldı | 11148 | anahtar, kapsama ve karar aynı |
| Yalnız durumu değişti | 3 | 7510 md. 17/1 (BOTTLING), 34147 md. 8/1-c (BOTTLING, SALES): NOT_COVERED → PARTIALLY_COVERED, üçü de REVIEW_REQUIRED; neden karşılaştırıcı v4 (4745065) |
| İncelemeye taşındı | 22 | 21 Kanun 6563 (kanal listesi açık), 1 Yönetmelik 23282 md. 19/1 (ürün etiketleri tam değil) |
| Kapsam dışı | 12 | 11 Yönetmelik 23282 md. 19/1, 1 Yönetmelik 7510 md. 34/4-c (cam koşulu; ürün etiketleri tam, cam yok) |
| Silindi (artık yükümlülük okunmuyor) | 6 | Yönetmelik 15594 md. 6/2 c.1 ("Yetkili merci tarafından … incelenir") |
| Yeni eklendi | 5 | aşağıda |

1 Ekim: 11148 + 3 + 22 + 12 + 6 = 11191. Şimdi: 11148 + 3 + 5 = 11156. −40 + 5 = −35.

Yeni eklenen 5 satır (hepsi yeni okunan yükümlülük, 1 Ekim'de karar yoktu; QDMS'te EXPORTED):

| Pilot | Hüküm | Hedef |
|---|---|---|
| İthalatçı | Yönetmelik 6207 md. 15/18 c.1 ("… içerecektir") | ALC-IMP-BREWING |
| İthalatçı | Yönetmelik 6207 md. 15/18 c.2 ("… onayı aranacaktır") | ALC-IMP-BREWING |
| Bira | Yönetmelik 6207 md. 15/18 c.1 | ALC-INT-BREWING |
| Bira | Yönetmelik 6207 md. 15/18 c.2 | ALC-INT-BREWING |
| Şişeleyici | Yönetmelik 7510 md. 28/7 c.1 ("… yapılmış olacaktır") | NONALC-BOTTLING |

Tekilleştirme birimi ayrı: bira pilotunda 5660 ham satır = 3209 tekil yükümlülük × hedef (alkolsüz paketin 2451 satırının
2448'i alkol paketindeki satırla aynı sonuçla birleşir; 3'ü kapsanmış-otomatik olarak iki pakette de dışarıda). Her eski ve yeni
satırın akıbeti ve yeni satırın QDMS sonucu: `row-fates.csv` (11196 satır).

## 5. UNKNOWN neden dağılımı (şişeleyici, 3095 kayıt)

Politika kanıtı bu kayıtlarda rol oynamaz: uygulanabilirlik bilinmediği için kapsama değerlendirilmez.

| Neden | Kayıt | Tekil hüküm | Tüzel kişi eşleşmesi | Kaynak |
|---|---|---|---|---|
| Kaynakta muhatap yok, kanun metni | 2180 | 436 | 436 × 5 tüzel kişi | 6502 (1110), 5996 (620), 4760 (265), 6563 (185) |
| Kaynakta taraf adı var, sözlükte sınıfı yok ("işyeri") | 20 | 4 | 4 × 5 | 5996 |
| Şirket bilgisi eksik | 895 | 887 | NONALC-BRAND 887, DISTRIBUTION 3, SALES 3, ürün 2 | eksik kapı: faaliyet 872, şirket sınıfı 15, kanal 6, ürün etiketi 2 |
| Çelişkili profil bilgisi | 0 | | | |
| Başka (kaynak temellendirilemedi vb.) | 0 | | | |

2200 "kapsamı belirsiz" kaydın tamamı dört kanundan; her hüküm beş tüzel kişinin her birine ayrı kayıt olarak yazılıyor
(`records_per_obligation`: 5). Kanun metinlerinde katalog yedeği bilerek kullanılmıyor (`extraction.scopes_of`, `fallback`:
bir kanun birden çok tarafa konuşur). Şirket bilgisi eksikliğinin 887'si tek nedene bağlı: NONALC-BRAND'in profili tam
işaretli değil, faaliyetleri bilinmiyor.

Kaynakta muhatap var mı? 440 belirsiz hüküm üzerinde metin işaretleri (gösterge, karar değil; `tr_unclear_source_signals.py`):

| İşaret | Hüküm | Örnek | Ne anlama geliyor |
|---|---|---|---|
| İdarenin görevi şirket yükümlülüğü okunmuş | 55 | 5996 md. 4/1-a "Bakanlık, … tedbirleri almakla yükümlüdür"; md. 4/8 "İl özel idareleri ve belediyeler … yükümlüdür"; md. 5/5 c.2 "Maliye Bakanlığının görüşü alınır" | tür okuma hatası: muhatap var, şirket değil |
| Kişisiz / taraf bulunamadı | 314 | 5996 md. 9/2 "… itlafı … yerine getirilir" | kaba sayım: içinde adı geçen taraflar da var (md. 4/6 "haberdar olan ilgililer") |
| Adı geçen taraf, sözlükte sınıfı yok | 20 | 5996 md. 9/1 "Hayvan sahipleri …"; md. 33/4 "Laboratuvarlar …" | muhatap metinde var, içecek şirketi değil |
| Tüketici özne / tüketici hakkı | 19 | 6502 md. 26/2 c.4 "Tüketici, … ödediği takdirde …" | şirket yükümlülüğü değil |
| Hukuki sonuç / atıf | 17 | 5996 md. 14/3 "… veteriner hekim … tarafından uygulanır" | şirket yükümlülüğü değil |
| Liste maddesi, muhatap kapanış cümlesinde | 14 | 6502 md. 11/1 a–ç: maddeler tüketicinin seçimlik hakları ("… seçimlik haklarından birini kullanabilir."); yükümlülük kapanıştaki ayrı cümlede: "Satıcı, tüketicinin tercih ettiği bu talebi yerine getirmekle yükümlüdür." | maddeler şirket yükümlülüğü değil; satıcının yükümlülüğü ayrı cümlede, okuyucu liste maddelerine bu cümlenin yüklemini veriyor |
| Muhatap önceki cümlede (gönderme) | 1 | 6502 md. 10/3 c.2 "Bu etiketin tüketiciye verilmesi … zorunludur" | muhatap önceki cümlede (üretici, ithalatçı, satıcı) |

Kanıt olmadan muhatap atanmadı; UNKNOWN sayısını düşürecek bir varsayım eklenmedi.

Diğer iki pilot (aynı yöntem): bira 7331 kayıt (iki paket) = 5508 kanunda muhatap yok + 48 sınıfsız taraf + 1775 şirket bilgisi
eksik; ithalatçı 2428 = 1446 + 12 + 970. İki alkol pilotu aynı kanunları taşıdığı için kaynak tarafındaki inceleme konuları aynı
(486 hüküm).

## 6. İnceleme gruplama önerisi (uygulanmadı)

Kurallar: bir ana görev yalnız aynı belirsizliği, aynı sorumluyu ve aynı son tarihi (bugün hiçbir kayıtta son tarih yok)
paylaşan kayıtları toplar; her tüzel kişinin alt kararı ve kanıtı alt kayıt olarak kalır; nedenler birleştirilmez.

- Kaynak sorusu (kim bağlı?): metinden cevaplanır, her tüzel kişi için aynıdır → konu = hüküm; görev = madde (Hukuk, aynı
  metin okunur). Alt kayıtlar: her tüzel kişi, kendi diğer kapı kanıtıyla (ör. istisna kapısı).
- Şirket bilgisi sorusu: profilden cevaplanır, engellediği her hüküm için aynıdır → konu = görev = (tüzel kişi, eksik kapı)
  (Uyum); alt kayıtlar: o bilginin açacağı hükümler.

| Pilot | Ham kayıt | Tekil inceleme konusu | Kullanıcı görevi |
|---|---|---|---|
| Şişeleyici | 3095 | 446 (440 kaynak + 6 şirket) | 141 (135 madde + 6 şirket) |
| Bira (iki paket) | 7331 | 491 (486 + 5) | 160 (155 + 5) |
| İthalatçı | 2428 | 491 (486 + 5) | 160 (155 + 5) |

En büyük görev: "NONALC-BRAND'in faaliyet listesini tamamla" — 872 kaydı tek başına açar. Örnek ana görev çıktıları (alt
kayıtları kısaltılmış): `review-grouping-*.json` → `examples`.

## 7. Test sorunları (7188feb self-test, kullanıcının koşusu)

| Paket | Çalışan | Başarılı | Başarısız | Hata | Atlanan |
|---|---|---|---|---|---|
| TR | 378 | 378 | 0 | 0 | 0 |
| Tam | 1692 | 1377 | 2 | 240 | 73 |

242 başarısız/hata kimliği baseline (`evaluation/reports/regression-baseline/full-output.txt`) ile birebir aynı; TR modüllerinde 0.
239 `FileNotFoundError`: eksik klasörler `regulations/tedbirler-200713012` (191), `evaluation/datasets` (34), `scripts/launcher`
(6), `evaluation` (4), `regulations/kanun-5549` (2), `evaluation/independent` (1), `scripts` (1). Diğer 3:

| Test | Tür | Neden |
|---|---|---|
| `test_hardening_prompt_budget` (modül yüklenemedi) | ERROR | `ModuleNotFoundError: hardening_prompt_budget` — modül depoda yok |
| `test_turkiye.WorkspaceTurkiyeTests.test_sample_scenarios_ship_with_their_policies_and_the_page_offers_them` | FAIL | `'tr-masak-5549' not found in []` — MASAK 5549 örnek senaryosu yok (Kanun 5549 dosyaları eksik) |
| `test_v019_version.BuildScriptTests.test_build_info_carries_the_backend_version_and_the_source_hash` | FAIL | `scripts/Build-Launcher.ps1` yok |

Üçü de eksik artefakt; hiçbiri uydurulmadı, susturulmadı.

## 8. COVH-16: son tarih karşılaştırması neden tabloya bağlı

`backend/src/regchain/tr/compare.py` `_relate_frame`, satır 722–727: daha geç son tarih (`DEADLINE_LATER`) ancak
`common >= 2` ve (`overlap >= 0.5` veya `similarity >= SEMANTIC_RELEVANCE` (0,70)) ise çelişki sayılır. `overlap`, yükümlülüğün
tüm ayırt edici köklerinin ifadede bulunan payıdır (satır 637–638). COVH-16'da yükümlülük cümlesi uzun (form, elektronik ortam,
mesai bitimi …), ifade kısa: overlap 0,26; tablo değeri 0,753. Sonuç: tablo yoksa UNRELATED, varsa CONFLICTS.

Genel çözüm planı (davranış değişmedi):
1. Son tarihin ilişkisini cümlenin tamamıyla değil, son tarihin yönettiği öğelerle kur: aynı son tarih türü (DAY_OF_MONTH /
   WITHIN_DAYS), aynı alıcı (ikisinde de adı geçen kurum, ör. "Kurum"), aynı gönderme/bildirme eylemi sınıfı (sözlük: bildirmek,
   iletmek, intikal ettirmek, sunmak) ve son tarihe bağlı nesnede en az bir ortak ayırt edici kök ("satış").
2. Örtüşmeyi son tarihin cümle parçası üzerinde ölç (yüklem + nesne + alıcı), form/ortam ayrıntılarını değil.
3. Türler ve alıcı uyuşup nesne uyuşmuyorsa UNRELATED değil UNCLEAR (incelemeye) — sessiz geçiş yok.
4. Ölçüm: DEV/HOLDOUT/VALIDATION tablosuz ve tablolu, korpus CONTRADICTED sayısı önce/sonra; HOLDOUT ve VALIDATION bağımsız
   kanıt sayılmaz (ikisi de geliştirmede okundu).

## 9. Sıradaki en küçük düzeltme paketi (öneri)

1. **İdare görevlerini şirket yükümlülüğünden ayır** (55 hüküm × tüzel kişi): özne veya ajan kurum olan cümleler ("Bakanlık …
   yükümlüdür", "Bakanlıkça/… tarafından … alınır") DELEGATION. Yalnız metindeki kurum adına dayanır.
2. **Liste maddesi ile kapanıştaki ayrı cümleyi ayır** (14 hüküm): 6502 md. 11/1'de maddeler tüketici hakkı, satıcının yükümlülüğü kapanıştaki ikinci cümle; bugün maddeler o cümlenin yüklemini alıyor ve muhatapsız kalıyor.
3. **Tüketici özneli cümleler şirket yükümlülüğü değil** (19 hüküm): hak/sonuç cümlesi olarak işaretle.
4. **Kural sürüm etiketlerini artır** (`tr-frames-v2`, `tr-scope-rules-v2`): davranış değişiklikleri etiketten görünsün.
5. Her biri için önce hatayı gösteren test; etiketli setler ve korpus UNKNOWN/APPLIES sayıları önce/sonra; muhatap yalnız
   metindeki kanıtla.

Gruplama (6. bölüm) ve COVH-16 planı (8. bölüm) bu paketten sonra ayrı adımlar.

## 10. Dosyalar

`manifest-*.json`, `row-fates.csv`, `row-fates-summary.json`, `unknown-causes-summary.json`, `unknown-source-signals-bottler.json`,
`review-grouping-*.json`, `qdms-*.summary.json`. Ham dökümler (≈40 MB) ve tam QDMS dışa aktarımları `evaluation/runs/20261003-verification/`
altında (gitignored).
