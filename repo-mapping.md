# SurgTwin-GS — Repo Haritası

Bu belge, SurgTwin-GS projesinde kullanılacak GitHub repolarının hangi amaçla ve hangi modülde kullanılacağını haritalandırır.

İlgili kilitli konsept dokümanı: [[charters/SurgTwin-GS_Detailed_Concept_Paper.docx|SurgTwin-GS Konsept Dokümanı]]

---

## 1. Belirsizlik (Uncertainty) Modeli ve Çapraz-Görünüm Optimizasyonu
**Kullanılacak Repo:** `UC-NeRF` (https://github.com/wrld/UC-NeRF)
- **Projedeki Yeri:** Faz-1, Belirsizlik Loss Formülasyonu ve Cross-View Pseudo-GT üretimi.
- **Kullanım Amacı:** UC-NeRF, endoskopik seyrek görünümlerde (sparse views) belirsizlik tahmini (uncertainty estimation) ve fotometrik tutarsızlıkları modellemek için tasarlanmıştır. Konsept belgemizde belirttiğimiz "geometrik belirsizlik ve oklüzyon/halüsinasyon belirsizliği" sinyallerini matematiksel olarak formüle ederken (örneğin kayıp fonksiyonunu güven skoru ile ağırlıklandırma) bu reponun belirsizlik (uncertainty) tahmin mimarisi doğrudan referans alınacaktır.

## 2. Gerçek Zamanlı Rendering ve Hafif Çıkarım (Lightweight Inference) Altyapısı
**Kullanılacak Repolar:**
`Instant-NGP` (https://github.com/nvlabs/instant-ngp)
`tiny-cuda-nn` (https://github.com/nvlabs/tiny-cuda-nn)
- **Projedeki Yeri:** Faz-1, "Near-real-time update" ve Çıkarım Aşamasındaki Hafif Belirsizlik Eşleyicisi (Lightweight Uncertainty Mapper).
- **Kullanım Amacı:** Konsept belgesinde DeepSeek'in uyarısıyla "gradyan açlığını (gradient starvation) önlemek" ve "çapraz-görünüm tutarlılığını çıkarım aşamasından çıkarıp hızı korumak" kararı alınmıştı. `tiny-cuda-nn`, C++ ve CUDA ile yazılmış, son derece düşük bellek/zaman maliyeti olan bir sinir ağı iskeletidir. Çıkarım anında çalışacak olan "hafif belirsizlik eşleyici (confidence head)" ve GS yoğunluk kontrol (densification/pruning) mekanizmaları bu CUDA tabanı üzerinde inşa edilerek FPS hedefleri güvence altına alınacaktır.

## 3. Deforme Doku Rekonstrüksiyonu ve Karşılaştırmalı (Baseline) Modeller
**Kullanılacak Repolar:**
`EndoNeRF` (https://github.com/med-air/EndoNeRF)
`Forplane` (https://github.com/Loping151/ForPlane)
- **Projedeki Yeri:** Başarı Ölçütleri (Evaluation) ve Mimari Geliştirme.
- **Kullanım Amacı:** Konsept belgesi, SurgTwin-GS'in performansını kanıtlamak için çeşitli testler öngörür. EndoNeRF, cerrahi alet oklüzyonlarını maskeleme ve dinamik doku render etme konusunda alanın temel standartlarındandır. Forplane ise doku deformasyonlarını statik/dinamik ortogonal düzlemlere ayırarak optimizasyon süresini 100 kat hızlandıran yeni bir SOTA yöntemidir. Bu repoların kaynak kodları, SurgTwin-GS'in (PSNR, SSIM, LPIPS, Depth RMSE) metriklerinde ne kadar üstün olduğunu kanıtlamak için "Baseline (Kıyaslama)" modeli olarak çalıştırılacaktır.

## 4. Speküler Yansıma Testleri ve Veri Bozulması İzolasyonu (Robustness Testing)
**Kullanılacak Repo:** `Endoscopy Corruptions` (https://github.com/Ivanrs297/endoscopycorruptions)
- **Projedeki Yeri:** Risk Azaltma Stratejisi (Speküler Yansıma Veri Bozulması) ve Çapraz-Görünüm Tutarlılığı.
- **Kullanım Amacı:** Konsept belgesinde, cerrahi ortamlardaki parlak doku yansımalarının (specular reflections) üretilen sözde (pseudo-GT) hata sinyallerini bozabileceği teknik bir risk olarak belirlenmiştir. EndoDepth ekibi tarafından sunulan bu kütüphane, endoskopik görüntülere spesifik olarak "oklüzyon, ışık yansıması, doku deformasyonu ve bulanıklık" gibi yapay bozulmalar (corruptions) ekleyebilmektedir. Geliştireceğimiz "yansıma filtreleme ve robüst hata fonksiyonlarının" gerçekten işe yarayıp yaramadığını laboratuvar ortamında simüle etmek için doğrudan bu kütüphaneyi kullanacağız.

## 5. Cerrahi Alet Maskeleme (Görüş Temizliği)
**Kullanılacak Repo:** `SAM3D` (veya yeni nesil `SAM2` entegrasyonları)
- **Projedeki Yeri:** Faz-1, Girdi ve Ön İşleme (Input and preprocessing).
- **Kullanım Amacı:** SurgTwin-GS mimarisindeki "belirsizlik skorlarının" üretilebilmesi için cerrahi aletlerin ve oklüzyonların nerede olduğunun sisteme bildirilmesi gerekir. SAM tabanlı modeller, endoskopik videolardaki aletleri piksel bazında hızlıca segmente ederek (maskeleyerek), Gaussian Splatting motoruna "bu bölgedeki pikseller aletten kaynaklanıyor, fotometrik loss'u burası için düşür" komutunun verilmesini (Loss Decoupling) sağlayacak alt veriyi (mask_id) üretecektir.

## 6. Gelecek Fazlar İçin (Faz-2/Faz-3) Trajektori ve Poz Doğrulama
**Kullanılacak Repo:** `EVO` (http://github.com/MichaelGrupp/evo)
- **Projedeki Yeri:** Faz-2 (SfM-free ve Kamera Poz İnce Ayarı).
- **Kullanım Amacı:** Konsept belgemizde Faz-1 için "kamera poz tahmin (pose estimation) probleminin SERV-CT gibi veri setleriyle izole edileceği" açıkça yazılmıştır. Ancak proje Faz-2'ye (kendi kamera pozunu bulan monocular sisteme) geçtiğinde, endoskobun uzaydaki hareketinin (trajectory) doğruluğunu hesaplamak için (ATE RMSE - Absolute Trajectory Error vb.) EVO kütüphanesini doğrudan kalite ölçüm aracı olarak kullanacağız.

---

**Özetle:** Projenin kodlama aşamasına geçerken **UC-NeRF**'ten belirsizlik matematiğini, **tiny-cuda-nn**'den donanım hızlandırmasını, **Endoscopy Corruptions**'tan robüstlük testlerini alacak ve kurduğumuz mimariyi **EndoNeRF** ve **Forplane**'e karşı yarıştıracağız.
