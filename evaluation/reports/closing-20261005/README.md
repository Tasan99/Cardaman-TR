# Kapanış raporu — TR içecek pilotu, 3–5 Ekim 2026 turu

Son temiz sürüm: **6e57a81** (yerel, push edilmedi). Kullanıcının self-test'i (`evaluation/selftest/20261004-230251`):
TR 403 test, 403 başarılı; tam paket 1717 test, 242 başarısız/hata = kayıtlı baseline (kimlik ve neden), 73 atlanan;
kural sürüm kaydı 8/8 eşleşiyor. Etiketler INDICATIVE; profiller ve politika kayıtları sentetik.

## Bu turda yapılanlar (sırayla, her biri önce başarısız test → düzeltme)

| Adım | Commit | Sonuç |
|---|---|---|
| 1. Dört okuma düzeltmesi | 3caea6b, 314bded | idarenin görevi (özne yalnız kurum) DELEGATION; tüketici özneli cümle yükümlülük değil (edilgen yasak hariç); liste kapanışındaki ikinci cümle ayrı hüküm; kural sürümleri dosya özetiyle kayıtlı (`data/rule_versions.json`, `scripts/tr_rule_versions.py`) |
| 2. Üç pilot yeniden | 20261003-step2b | kaynak kaynaklı UNKNOWN: şişeleyici 2200 → 2160, bira 5556 → 5460, ithalatçı 1458 → 1434; düşen her karar metinden doğrulandı; yeni otomatik kararların tamamı 6502 md. 11/1 c.2'den |
| 3. Profil soruları | 7acf3e7, 7ff89fa, 8e52683 | `questions` → `questionnaire.csv` (şirket doldurur: değer başına E/H, liste tam mı, cevaplayan, dayanak); `answer --answers questionnaire.csv` yalnız etkilenen hedefleri yeniden yönlendirir; boyut başına tamlık bayrağı (tr-routing-v2) |
| 4. İnceleme gruplama | 7ff89fa (tr-qdms-v3/v4) | şişeleyici 3056 kayıt → 438 konu → 141 görev (Hukuk 135, Uyum 6); alt kayıtlar karar ve kanıtını korur; onay görevde; alt kayıt tek başına onaylanamaz; değişen alt kayıt görev onayını geçersiz kılar |
| 5. Yapısal son tarih çelişkisi | 29193e1, 6e57a81 (tr-compare-rules-v6) | aynı son tarih türü + aynı alıcı kurum (kök + yönelme/bulunma eki) + teslim fiili + ortak nesne sözcüğü; kurum ve fiil aynı, nesne farklı → UNCLEAR. HOLDOUT tablosuz 18 → **19/28**, tablolu 19; DEV 36, VALIDATION 22 (tablolu/tablosuz aynı). Korpus: 11164 satırın hepsi değişmedi (yeni çelişki 0) |

Ara hatalar ve düzeltmeleri: tüketici kuralı iki gerçek yasağı düşürdü (314bded ile geri alındı); v5'teki alıcı tespiti "Kuruma"yı
görmedi, kural hiç çalışmadı (6e57a81); ankette kodlar yerine etiketler (8e52683).

## Açık kalanlar

- **6502 md. 11/1 c.2 "Satıcı, … yükümlüdür":** yalnız toptan satış yapan tüzel kişilere APPLIES diyor (mevcut SELLER eşlemesi
  WHOLESALE'i de kapsıyor; kanun satıcıyı tüketiciye mal sunan diye tanımlıyor). Değiştirilmedi; incelenmeli.
- **Şişeleyicide 2160 kaynak kaynaklı UNKNOWN:** dört kanunun 432 hükmü × 5 tüzel kişi; 132 Hukuk görevi olarak gruplanmış.
  Kanıt olmadan muhatap atanmadı; düşürmenin yolu hüküm okuyucusuna yeni genel kurallar, varsayım değil.
- **Profil soruları şirketin cevabını bekliyor:** 5 soru, 895 değerlendirme (873'ü tek soru: marka şirketinin faaliyet listesi).
- **HOLDOUT ve VALIDATION bağımsız kanıt değil:** ikisi de geliştirmede okundu. Doğruluk iddiası için yeni uzman etiketli set gerekir.
- Gerçek belge testi (`scripts/tr_external_test.py`, v4+ ile) ve Rekabet Kurulu kararı bu turda koşulmadı.

## Gerçek CCI pilotuna geçiş için gerekenler

1. Rekabet Kurulu taahhüt metni PDF'i (`evaluation/external/pdf/`), `scripts/tr_decision_import.py` ile saklanır.
2. Şirketin doldurduğu `questionnaire.csv` ve gerçek profil (tüzel kişiler, faaliyetler, kanallar, ürünler; tamlık dayanaklarıyla).
3. Şirketin politika/prosedür kayıtları (kayıt formatı `cardaman-tr-policy-register/1`).
4. Çıktı: `qdms export` (politika satırları + inceleme görevleri), onay şablonu, hazır eylem listesi; hiçbir eylem insan onayı
   olmadan hazır olmaz.

Ölçüm dosyaları: `evaluation/runs/20261004-step5b/` (döküm, akıbet tablosu), `20261004-step5b/holdout.json` kullanıcı koşusu.
Geçici git worktree'leri `C:\dev\cw-*` silinebilir (`git worktree remove`).
