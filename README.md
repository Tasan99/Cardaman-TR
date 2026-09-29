# RegChain

FCA Handbook ve T.C. mevzuatı (mevzuat.gov.tr) değişikliklerini şirket yükümlülükleri ve policy/control kanıtlarına bağlayan
RegTech platformunun **v0.16: hedef madde filtresi ve analiz kapsamı ekranı, kanıt kapısı, ikinci okumayla doğrulanan çelişki, ağırlıklı kapsam, provizyon başına uygulanabilirlik, yerel reranker, yanıt önbelleği ve zengin AI çağrı kaydı** geliştirmesi.

Yeni: [v0.16 kök nedenler, önce/sonra canlı ölçüm (150 dk → 7,6 dk), reranker ve kalan borçlar](docs/PHASE16.md).
Önceki: [v0.15 kök neden analizi, ön sınıflandırma, uygulanabilirlik dayanağı, öneri ve AI kaydı](docs/PHASE15.md).
Önceki: [v0.14 Cardamon'un iş akışlarının karşılıkları ve neyin değişmediği](docs/PHASE14.md).
Önceki: [v0.13 Türk mevzuatı: kanun/yönetmelik/tebliğ seçimi, Türkçe dayanak katmanı, ufuk taraması](docs/PHASE13.md).
Önceki: [v0.12 ayrı yargı modeli, düşünme modu ve pasaj başına karar](docs/PHASE12.md).
Önceki: [v0.11 herhangi bir FCA Handbook bölümünü seçip bütünüyle inceleme](docs/PHASE11.md).
Önceki: [v0.10 anlamsal policy araması, yalıtılmış çelişki kontrolü ve uzun yol düzeltmesi](docs/PHASE10.md).
Önceki: [v0.9 tarayıcıdan şirket/policy yükleme, analiz ve inceleme](docs/PHASE9.md).
Önceki: [v0.8 pilotu çalıştırma, inceleme ekranı ve uzman değerlendirmesi](docs/PHASE8.md).
Önceki: [v0.7 değişiklikleri, test durumu ve çalıştırma](docs/PHASE7.md).
Önceki: [kaynak/parser değişimi ayrıştırma ve gerçek kaynaklarda model ölçümü](docs/PHASE6.md).
Önceki: [konsolide Handbook kaynakları, belgeler arası referans çözümü ve kapsama raporu](docs/PHASE5.md).
Önceki: [kalite iyileştirmeleri, yeniden işleme ve test komutları](docs/PHASE4.md).

Güncel kullanım: [FCA belgelerini toplama ve sürümleri inceleme](docs/PHASE2.md).

Başlangıç: [tam mimari, şema açıklaması ve veri akışı](docs/ARCHITECTURE.md).
[Aşama planı](docs/ROADMAP.md) · [Doğrulama durumu](docs/VALIDATION.md).

## Hazır olanlar

**En kolay başlatma: `Cardaman.exe`'ye çift tıkla.** Sunucuyu başlatır, Ollama kapalıysa açmayı
dener ve tarayıcıyı doğru oturum bağlantısıyla açar. Pencere açık kaldıkça uygulama çalışır;
pencereyi kapatınca sunucu da kapanır. Sekmeyi kapattıysan tekrar çift tıkla, aynı oturum açılır.
Kaynağı `scripts/launcher/CardamanLauncher.cs`; `.\scripts\Build-Launcher.ps1` Windows'un kendi
C# derleyicisiyle yeniden üretir. Oturum anahtarı diske yazılmaz.

Terminalden başlatmak istersen: Cardaman kökünde `.\scripts\Run-Workspace.ps1` çalıştırıp
terminalin verdiği oturum bağlantısını aç. **Terminalde Ctrl+C kopyalamaz, sunucuyu kapatır.** Şirket profilini doldur, belgeleri yükle, analizi başlat ve insan
incelemesini aynı ekrandan kaydet. Geçmiş analizler korunur; kanıt paketi ZIP olarak
indirilebilir. Tek operatörlü bu yerel ekran Docker gerektirmez. Gerçek AI analizi
için Ollama gerekir; kural tabanlı akış testi ve geçmişe erişim Ollama olmadan çalışır.
Modeller: çıkarım `qwen3:4b`, yargı `qwen3:8b` (düşünme açık), arama `bge-m3` — üçü de kuruluysa
başlatıcı kendiliğinden seçer. Yargı modeli her policy pasajını tek tek okur; 18 pasajlık bir policy
setinde yükümlülük başına 6–7 dakika sürer, bu yüzden bölümün tamamı yerine kısım seçmek beklenen
kullanımdır. Ölçümler ve sınırlar: [docs/PHASE12.md](docs/PHASE12.md).

**Türkiye:** 03. adımda "Türkiye" seçili gelir; düzenlemeyi adıyla ara (ör. "tedbirler hakkında",
"ödeme hizmetleri"), listeden seç, istersen madde numaralarını yaz. Metin mevzuat.gov.tr'den
kendiliğinden iner ve madde madde ayrıştırılır; Resmî Gazete'nin son günlerdeki başlıkları ana
ekranda taranır. Hazır örnek: "Anadolu Ödeme" senaryosu (5549 sayılı Kanun md. 3/4/8 + üç Türkçe
policy). Ne kadarının ölçüldüğü ve sınırlar: [docs/PHASE13.md](docs/PHASE13.md).

**Cardamon eşleniği (v0.14):** CSV kontrol kaydı yükle (her satır bir kontrol) → başlıkta uygulanabilir / policy
kapsaması / kontrol kapsaması sayıları; her yükümlülüğe AI özeti ve şirketin risk taksonomisinden kategori;
Türk mevzuatında ceza maddelerinden yaptırım bağlantısı; analiz sayfasında "AI'a sor" (alıntıları
doğrulanmış yanıt), policy maddesi taslağı ve sorumlu atama. Hepsi yerel modelle; süreler ve sınırlar
[docs/PHASE14.md](docs/PHASE14.md).

**v0.15 — öneri ve dayanak:** Her madde alt paragrafı modele gitmeden önce kuralla sınıflanır (yükümlülük /
yasak / tanım / kapsam / yetki devri "yönetmelikle belirlenir" / yaptırım / izin / istisna); yalnızca yükümlülük
taşıyanlar çıkarıma girer, diğerleri sınıfıyla raporlanır. Uygulanabilirlik dört durumludur (uygulanıyor / olası
uygulanabilir / uygulanmıyor / belirsiz) ve her öneri, yargı modelinin karşılaştırdığı **şirket bilgisi ↔ kapsam
ifadesi** çiftlerini (backend doğrulamalı) taşır. Policy'nin karşılamadığı ya da çeliştiği her yükümlülük için
**AI önerisi** (tür, öncelik ve eylem kuralla; policy ifadesi taslağı yerel modelle) üretilir; hepsi "AI-GENERATED
PROPOSAL" damgalıdır ve inceleme kararında kabul/ret alanı vardır. Her model çağrısı `ai-calls.jsonl`'a (görev,
model, istem hash'i, token, süre; metin yok) yazılır; iş kaydında `ai_usage` görünür. Ayrıntı ve canlı ölçüm:
[docs/PHASE15.md](docs/PHASE15.md).

**v0.16 — kapsam ve hız:** "Yalnızca şu maddeler" alanı virgül/boşluk/"md." ile yazılır, canlı önizlenir ("Hedef maddeler: 3, 4, 8")
ve boş bırakılırsa bütün düzenleme için onay ister; iş kaydı ve rapor **hedef provision / context olarak okunan yardımcı provision**
ayrımını ve A–K özetini (aday, dört durumlu uygulanabilirlik, beş durumlu kapsam, kontrol kaydı ayrı, öneri durumu, insan onayı
bekleyen, çağrı ve süre) gösterir. Yargı: kanıt kapısı (sayfa numarası, başlık ve kopuk satır kırıntıları yargıya gitmez),
CONFLICT için ikinci doğrulayıcı okuma, destek belirsizlikten ağır basar, uygulanabilirlik provizyon başına bir kez, bütçeyi aşan
onarım istemi kırpılır, dayanakla çelişen durum UNKNOWN olarak kaydedilir. **Bent düzeyi entity gate (v0.16.1):** bendin
bağladığı yükümlü türü ve muhatap türü (dernek, vakıf, yabancı dernek şubesi, gerçek/tüzel kişi, banka, merkezi yurt dışında
bulunan yükümlü…) şirket profiliyle kuralla karşılaştırılır; profil dışıysa provizyon APPLIES olsa bile bent DOES_NOT_APPLY
olur, model çağrılmaz ve altı alanlı dayanak (ana provision, alt bent, gerekli varlık, şirket varlığı, eşleşme, sonuç)
raporda görünür; dayanaktaki bir NO yalnızca açık istisna ise APPLIES'ı veto eder ve kaynağıyla yazılır. İsteğe bağlı yerel cross-encoder reranker
(`RERANK_MODEL=BAAI/bge-reranker-v2-m3`, model yoksa RRF'e düşer). Aynı senaryoda 674 çağrı / 150 dk → 39 çağrı / 7,6 dk;
ölçüm ve sınırlar [docs/PHASE16.md](docs/PHASE16.md).

**Otonom karar modu:** 04. adımda "Otonom AI kararı" seçilirse AI, kanıt kapısından geçen yükümlülükleri
kendisi karara bağlayıp imzalar (ayrı kanıt olayı), geçemeyenleri insana bırakır; "yargı tekrarı" 2–3
seçilirse uyuşmayan okumalar otomatik karar dışında kalır. Ölçülen hata oranı ekranda yazar; sorumluluk
yazılıma geçmez.

Docker gerektirmeyen yerel FCA CONC pilotu: şirket profili ve TXT/MD/PDF/DOCX policy
metinlerini alır; AI uygulanabilirlik/policy önerilerini kaynak alıntılarıyla üretir.
Tarayıcı inceleme ekranı, hash ile bağlı karar dosyası, versiyon karşılaştırması ve
uzman etiketleme çalışma listesi vardır. İlk canlı deneme gerçek FCA metinleri ve
sentetik şirket/policy dosyalarıyla yapıldı; gerçek uzman onayı henüz yoktur.
Repo kökünde `.\scripts\Run-Concpilot.ps1` yeni bir pilot paketi üretir.

PostgreSQL/pgvector DDL, tenant RLS ve composite foreign key'ler, tarihsel kayıt
koruması, checksum migration runner, FastAPI operasyon endpoint'leri, Docker
geliştirme tanımı ve test edilebilir audit hash çekirdeği.
FCA HTML/PDF çekimi, kaynak hash/bytes arşivi, duplicate tespiti, append-only sürümler,
paragraf farkları ve salt okunur regulatory API eklendi.

Şema ve alıntı doğrulamalı çıkarım pipeline'ı, rules baseline ve Ollama adaptörü eklendi.
Yerel Qwen3 4B/8B modelleri dört sentetik ve iki arşivlenmiş FCA örneğinde denendi;
bu küçük geliştirme seti hukuki doğruluk ölçümü değildir. Tüm adaylar insan incelemesi gerektirir.
Konsolide FCA Handbook adaptörü (basılı etiket, R/G tipi, provision yürürlük tarihi,
glossary bağları), sabitlenmiş corpus üzerinde belgeler arası referans çözümü,
audit için corpus pinning, eksik kaynak kapsama raporu ve amendment enstrümanını
konsolide provision'lara bağlayan tarih doğrulamalı eşleştirme eklendi.
Arşivlenmiş bytes'ı tek parser ile yeniden ayrıştırarak kaynak değişikliğini parser
değişikliğinden ayıran karşılaştırma ve gerçek kaynaklarda salt okunur model ölçüm
harness'ı eklendi. Model denemeleri ve ret nedenleri `docs/model-comparison-v07.json` dosyasındadır.
Yerel şirket/policy inceleme komutu, inceleyen kararlarını ve kaynak alıntılarını
hash doğrulamalı pakette tutar; otomatik uygulanabilirlik veya compliance kararı vermez.
Yerel pilot, şirket policy pasajlarını yerel embedding modeliyle (varsayılan bge-m3) anlamca
sıralar ve olumlu hükümleri pasaj başına yalıtılmış bir çelişki kontrolünden geçirir; bunlar
sıralama ve ihtiyat mekanizmalarıdır, doğruluk ölçümü değildir. Regülasyon tarafında semantik
embedding RAG, authenticated şirket API'leri, applicability/gap/impact motorlarının üretim
API'leri, Next.js dashboard ve testnet anchoring henüz uygulanmamıştır.
İskelet üzerinden gerçek compliance kararı üretilmez.

## Docker ile geliştirme

Docker Desktop + Compose gerektirir. Repo kökünde PowerShell:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Yeni kurulumda .env parolalarını ve DATABASE_URL değerini tutarlı biçimde değiştir.
docker compose up --build -d
docker compose logs migrate api
Invoke-RestMethod http://localhost:8000/health/live
Invoke-RestMethod http://localhost:8000/health/ready
```

Konsolide kaynakları almak için `docker compose --profile ingest run --build --rm ingest handbook`,
eksik bölümleri görmek için [PHASE5 kapsama komutu](docs/PHASE5.md#çalıştırma).

API/OpenAPI: http://localhost:8000/docs. Resmi FCA kaynakları için okuma API'leri vardır;
bu PostgreSQL API'sinde kimlikli şirket ve analiz endpoint'leri yoktur. v0.9 çalışma
alanı ayrı, yalnızca yerel erişime açık bir süreçtir. DB/Redis host'a port
açmaz. API yalnızca localhost'ta dinler. Migration owner rolüyle; API salt okunur
non-owner rolüyle çalışır. .env yerel geliştirme içindir, üretim secret deposu değildir.
Parolada URI özel karakteri varsa DATABASE_URL içinde percent-encode edin.

## Testler

Python 3.12+ ile repo kökünde:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e './backend[test]'
.\.venv\Scripts\python -m unittest discover -s backend/tests -v
```

PostgreSQL integration testleri yalnızca disposable test DB üzerinde yürütülür:

```powershell
docker compose --profile test run --build --rm test
```

Bu komut izole db-test servisine bağlanır. Testler kendi transaction'larını geri alır.
Dependency aralıkları başlangıç uyumluluğu içindir; üretim/release öncesi lock ve
image digest pinleme zorunlu kabul ölçütüdür.

## Audit çekirdeği sınırı

`make_event` sabit canonical JSON formatıyla hash üretir. `verify_chain` güvenilir
expected_head ve expected_count ister. Dışarıya sabitlenmemiş bir zincir baştan
yeniden yazılabilir; hash çekirdeği tek başına blockchain entegrasyonu değildir.
Model zamanı ve hukuki doğruluk blockchain tarafından doğrulanmaz.
