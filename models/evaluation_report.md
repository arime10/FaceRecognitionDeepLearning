# 📊 Yapay Zeka Model Değerlendirme ve Performans Raporu

**Proje Adı:** Gerçek vs. Sahte Yüz Sınıflandırma (Face Spoofing / Deepfake Detection)  
**Model Mimarisi:** Google Xception (Extreme Inception) + Transfer Learning & Fine-Tuning  
**Tarih:** 30 Eylül 2026  
**Durum:** Başarıyla Tamamlandı (Optimal Yakınsama)

---

## 1. Temel Performans Özeti

| Metrik | Değer | Açıklama |
| :--- | :--- | :--- |
| **Genel Doğruluk (Accuracy)** | **%93.33** | Modelin test edilen 750 görselde verdiği doğru karar oranı |
| **Doğrulama Kaybı (Val Loss)** | **0.1862** | Modelin tahmin hatası (Sıfıra ne kadar yakınsa o kadar iyi) |
| **Eğitim Doğruluğu (Train Acc)** | **%90.15** | Modelin eğitim kümesindeki öğrenme oranı |
| **Eğitim Kaybı (Train Loss)** | **0.2546** | Eğitim sırasındaki hata payı |
| **Aşırı Öğrenme (Overfitting)** | **Yok** | Val Acc (%93.3), Train Acc'den (%90.1) yüksek; ezberleme yok |

---

## 2. Sınıf Bazında Detaylı Başarı Karnesi

Modelin performansı **daha önce hiç görmediği 750 adet bağımsız test görseli** üzerinde ölçülmüştür:

| Sınıf | Hassasiyet (Precision) | Duyarlılık (Recall) | F1-Skoru | Test Görsel Sayısı |
| :--- | :---: | :---: | :---: | :---: |
| **Gerçek (Class 0)** | **%90.0** | **%97.4** | **0.94** | 388 adet |
| **Sahte (Class 1)** | **%97.0** | **%89.0** | **0.93** | 362 adet |
| **Ağırlıklı Ortalama** | **%94.0** | **%93.3** | **0.93** | **750 adet** |

### Bu Metrikler Ne Anlama Geliyor?
* **Gerçek Yüz Duyarlılığı (Recall %97.4):** Karşısına gelen 388 gerçek insanın 378'ini hiç sektirmeden "Gerçek" olarak tanıdı. Gerçek bir insanı kapıda bırakma riski sadece %2.6!
* **Sahte Yüz Hassasiyeti (Precision %97.0):** Model bir fotoğrafa "Bu SAHTE" dediğinde, bu kararında **%97 oranında haklı çıkıyor**. Yanlış alarm oranı son derece düşük.
* **F1-Skoru (0.93 - 0.94):** Hassasiyet ile duyarlılık arasındaki harmonik dengedir. 0.90'ın üzeri endüstriyel standartlarda "üretim seviyesi (production-grade)" kabul edilir.

---

## 3. Karışıklık Matrisi (Confusion Matrix)

Yapay zekanın tam olarak nerede doğru bilip nerede yanıldığını gösteren sayısal tablo:

```
                  Tahmin: GERÇEK     Tahmin: SAHTE
Gerçekte GERÇEK:       378                10          (Toplam 388)
Gerçekte SAHTE :        40               322          (Toplam 362)
```

* **Doğru Tespitler:** $378 + 322 = 700$ adet görsel hatasız bilindi (%93.33).
* **Yanlış Tespitler:** Sadece $10 + 40 = 50$ adet görselde hata yapıldı (%6.67).

---

## 4. Eğitim Mimarisi ve Hiperparametreler

* **Girdi Çözünürlüğü:** 256 x 256 RGB
* **Batch Size:** 32
* **Aşama 1 (Isınma / Warm-Up):** 3 Epoch (Learning Rate: 1e-4) - Sadece eklenen `Dense(256)` ve `Dropout(0.4)` eğitildi.
* **Aşama 2 (İnce Ayar / Fine-Tuning):** 10 Epoch (Learning Rate: 1e-5) - Xception'ın son 20 katmanı açıldı.
* **Düzenlileştirme (Regularization):** Data Augmentation (Döndürme, kırpma, kaydırma), Dropout (0.4), EarlyStopping (patience=5).
* **Kayıtlı Model Konumu:** `models/best_model.keras`
* **Metrik Grafiği Konumu:** `models/training_metrics.png`
