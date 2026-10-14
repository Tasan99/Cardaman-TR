# Düzeltme turu — 12–14 Ekim 2026 (test + inceleme bulguları)

Başlangıç: **b664a9f**. Bitiş: **9c391c3** + bu rapor. Kaynak: 12 Ekim test ve çok ajanlı inceleme turunda doğrulanan bulgular
(`evaluation/selftest/20261012-164309/review-findings-raw.txt`, yerel, git dışı). Her düzeltmede önce başarısız test yazıldı,
sonra düzeltme yapıldı. Etiketler INDICATIVE; profiller ve politika kayıtları sentetik.

## Self-test (modelsiz), önce / sonra

| Adım | 20261012-164309 (b664a9f) | 20261014-160814 (9c391c3) |
|---|---|---|
| TR testleri | 403 OK | **458 OK** (55 yeni test) |
| Tam paket regresyonu | IDENTICAL | IDENTICAL (yeni hata 0, düzelen 0, neden değişen 0) |
| 3 pilot (replay), değerlendirme, QDMS özeti | — | önceki koşuyla birebir aynı (SUMMARY.txt satır 3–4b) |
| Kural sürümleri | 8/8 ok | 8/8 ok |

Değerlendirme sayıları değişmedi: kural okuma DEV 1.0 / HOLDOUT 0.9375, kural kapsaması DEV 0.8 / HOLDOUT 0.6786, strict_accuracy
DEV 0.6667 / HOLDOUT 0.6786. Yanlış otomatik COVERED 0, yanlış otomatik CONTRADICTED 0. Düzeltilen kalıplar sentetik pilot
kayıtlarında geçmediği için pilot satırları değişmedi; düzeltmeler kendi testleriyle ölçüldü.

## Yapılanlar

| Commit | Etiket | Sonuç |
|---|---|---|
| 5149c2c | tr-frames-v4, tr-scope-rules-v3, tr-compare-rules-v7 | Kural okuyucu ve karşılaştırıcı için tek bir sayı okuyucu. **Sayı sözcükleri:** "on dört gün" artık 4 değil 14 gün (20237'nin 14 günlük altı yükümlülüğü; yasal 14 günü yazan politika artık CONTRADICTED çıkmıyor). Bitişik yazılan "onbeş", "kırkbeş" ve "yüzseksen" okunuyor. **Ondalık:** "%94.8" 94.8 (önce 948), "0.5 mg/L" 0.5, "1.000.000 litre" bir milyon; tarihler sayı sanılmıyor. **Saat:** "24 saat" bir son tarih birimi oldu ve bir hükmün tüm son tarihleri okunuyor; eşleştirme belirsizse sonuç UNCLEAR. **Yaş:** "18 yaşını doldurmamış" da MINOR. 40 korpus çerçevesinin sayısı düzeldi; hiçbir yükümlülük grubu değişmedi. |
| 3bd7783 | tr-qdms-v5 | Onay fingerprint'i eylemi yapacak tüzel kişileri (entity_ids) ve kanıtı da kapsıyor. Kim yapacak değişirse eski onay geçersiz (STALE_APPROVAL). |
| 68c9648 | tr-qdms-v6 | `qdms approve`, export.json'daki durum ve fingerprint'lere güvenmiyor; her fingerprint'i yeniden hesaplıyor, elle düzenlenmiş satır EXPORT_MODIFIED olarak reddediliyor. Bir satırı iki farklı kararla veren onay dosyası bütünüyle reddediliyor (önce ilk karar geçerli sayılıyordu). Birden çok dosyada en geç `decided_at` geçerli; önceki kararlar `history`'de saklanıyor. BOM'lu JSON kabul ediliyor, bozuk dosya traceback yerine tek satır mesajla exit 2 veriyor. |
| 7db6ed4 | tr-qdms-v7 | İnceleme düzeltmeleri. Rakam ya da gelecek tarihli `decided_at` reddediliyor (1 saat saat-kayması payı var). Fingerprint satırın tamamını, impacted_process dahil, kapsıyor. Eski kararlar dosyası verilmeden yeniden onaylamak geçmişi düşüremiyor. Bozuk entity_ids traceback vermiyor. Ayakta kalan mutantlar için testler eklendi. |
| cbd5b84 | — (kural dosyası değişmedi) | Anket Excel'den geri gelebiliyor. Kodlama sırası: UTF-8 (BOM'lu/BOM'suz), UTF-16 ("Unicode metin"), Windows-1254. Ayraç başlık satırından seçiliyor (`;`, `,` ya da TAB). Eksik sütun adıyla bildiriliyor. Çelişen yinelenen satırlar (aynı değere hem E hem H) reddediliyor. `questions` dolu questionnaire.csv / questions.json'ın üzerine `--force` olmadan yazmıyor. `questions` ve `answer` hatalı girdide traceback yerine tek satır verip exit 2 dönüyor. stdout/stderr UTF-8'e çevrildi; çıktı yönlendirildiğinde PYTHONUTF8 olmadan da çökmüyor. |
| 9c391c3 | — | İnceleme düzeltmeleri. Aynı sütun iki kez verilirse (örneğin "CEVAP" ve "Cevap") reddediliyor. Tek harfli "Y"/"N" reddediliyor (Türkçede "Y" = Yok olabilir). Hata mesajları hangi dosyanın bozuk olduğunu söylüyor. Uzantısız JSON cevap dosyası tanınıyor. |

B (QDMS) ve C (anket) şeritlerini ikişer bağımsız inceleyici (doğruluk / proje kuralları) çürütmeye çalıştı; buldukları
düzeltildi. **5149c2c bağımsız incelemeden geçmedi:** tur kullanıcı isteğiyle bu noktada kapatıldı.

Not: QDMS fingerprint'leri değişti (v4 → v7). v4 ile verilmiş onaylar tasarım gereği geçersiz; depoda doldurulmuş onay dosyası yok.

## Açık kalanlar (önerilen sırayla)

1. **Kapsama anlamı (yarıda kaldı, commit edilmedi):** politikadaki istisna yok sayılıyor (yanlış COVERED); "06.00 ile 22.00
   arasında" CONTRADICTED çıkıyor; "tavsiye edilir" MUST okunuyor. Taslak değişiklikler `C:\dev\cw-fa` çalışma kopyasında,
   commit'lenmemiş, testten geçmemiş.
2. **Şirketin kendi politika kaydı:** `--register PATH` seçeneği; "1)" numaralı ya da numarasız belgede ifadeler kayboluyor;
   kayıt metni Unicode normalize edilmiyor; BOM'lu register.json reddediliyor.
3. 5149c2c'nin bağımsız incelemesi; canlı model bacağı (`Test-TR.ps1 -WithModel`).
4. 12 Ekim raporundaki diğer bulgular: alıcı kurum ve aktör denetimi, ters niteleyici, uygulanabilirlik
   (SELLER/WHOLESALE 100 satır, ambalaj atığı ve GEKAP), VALIDATION'ın self-test'te puanlanmaması, `changes` komutunun mülgaları
   görmemesi.
