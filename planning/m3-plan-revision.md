M3 planı Gemini ve DeepSeek değerlendirmeleri sonrası nihai revizyonla onaylandı.

M3 başlığı:
M3 — Uncertainty-Weighted Photometric Loss for RGB-Depth Conflict Resolution

Ana karar:
İlk önerilen w_depth ağırlıklandırması kullanılmayacak. M3’te depth loss RGB residual’a göre kısılmayacak; depth loss valid GT depth maskesi üzerinde geometrik anchor olarak korunacak. Ağırlıklandırma photometric loss üzerinde yapılacak.

Loss:
L_total =
mean(w_photo * |rgb_pred - rgb_gt|)

* lambda_depth * mean(|depth_pred - depth_gt|[valid_depth_mask])

Başlangıç ayarları:
lambda_depth = 0.2
lambda_reg = 0.0
densification = false
w_photo_min = 0.15
alpha = 2.0 for p95-normalized H1

M3-H1:
w_photo residual-based olacak. u_photo hesaplanırken rgb_pred kesinlikle detach edilecek:

rgb_residual = abs(rgb_pred.detach() - rgb_gt).mean(dim=-1)
scale = quantile(rgb_residual.flatten(), 0.95).detach()
scale = clamp(scale, min=1e-4)
u_photo = clamp(rgb_residual / scale, 0.0, 1.0)
w_photo = clamp(exp(-alpha * u_photo), w_photo_min, 1.0)

Static max=1.0 normalization kullanılmayacak.
Per-iteration min-max normalization kullanılmayacak.
Robust detached p95 normalization kullanılacak.

PyTorch kritik kural:
w_photo hesaplamasında rgb_pred.detach() zorunludur. Aksi halde model kendi loss ağırlığını manipüle edebilir ve loss weighting optimizasyon grafiğinin parçası haline gelir.

M3-H2:
Mask-aware photometric weighting eklenecek. Specular/tool/occlusion maskeleri öncelikle w_photo hesabına dahil edilecek; depth loss’u otomatik kısmak için kullanılmayacak. Depth loss yalnızca valid_depth_mask ile sınırlandırılacak.

M3-H3:
H1/H2 sonuçlarına göre gerekirse lambda_depth=0.1 denenir; en iyi weighting stratejisi korunur.

Acceptance — Minimum PASS:

1. 1000 iterasyon tamamlanır.
2. depth_semantics = metric_meters doğrulanır.
3. densification = false.
4. scale regularizer kullanılmaz.
5. val_psnr >= 56.20 dB.
6. val_depth_rmse_m_raw <= 0.030 m.
7. M2-B R1’e göre PSNR artar.
8. mean_w_photo >= 0.15 ve <= 0.95.
9. w_photo dağılımı raporlanır: min, max, p10, p50, p90.
10. normalization mode = p95_detached olarak raporlanır.

Acceptance — Strong PASS:

1. val_psnr >= 56.20 dB.
2. val_depth_rmse_m_raw <= 0.023 m veya M2-B R1’den kötü değil.
3. val_abs_rel <= 0.18.
4. 0.3 <= mean_w_photo <= 0.8.
5. M2-B R1’e göre RGB-depth trade-off daha iyi olur.

M2-B controlled negative result olarak kapalı kalacak. Daha fazla fixed-weight M2-B tuning yapılmayacak. Learned mapper M3 kapsamına alınmayacak. Densification M4’e bırakılacak.
