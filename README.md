# 🧠 Biyometrik Güvenlik: Sahte Yüz ve Deepfake Tespit Sistemi

**Geliştirici:** Emirhan Dağ  
**Temel Teknolojiler:** Python 3.12, TensorFlow / Keras 3, Xception CNN, Grad-CAM, OpenCV, Gradio  
**Tarih:** 2026  

---

## 📌 1. Giriş ve Problem Tanımı (Motivation)

Biyometrik kimlik doğrulama sistemleri (yüz tanıma ile kilit açma, uzaktan bankacılık müşteri edinimi, sınır güvenlik kapıları), fiziksel ve dijital saldırılara karşı son derece savunmasızdır. Saldırganlar sıklıkla:
1. **Ekran Yansıtma Saldırıları (Replay Attacks):** Başka bir ekranda oynatılan yüz videosunu kameraya tutmak.
2. **Yazdırılmış Fotoğraf Saldırıları (Print Attacks):** Yüksek çözünürlüklü basılı fotoğrafları kullanmak.
3. **Yapay Zeka Destekli Sahtecilik (Deepfake & AI Generation):** Üretken ağlarla (GAN / Diffusion) sentetik yüzler üretmek.

Bu çalışmada amaç; **herhangi bir ek donanım veya derinlik sensörü (LiDAR / IR) gerektirmeksizin**, sadece standart RGB kameralardan alınan görüntüler üzerinde **mikro doku bozulmalarını, ekran yansımalarını ve sentetik artefaktları** tespit eden yüksek başarımlı bir derin öğrenme modeli geliştirmektir.

---

## 🏗️ 2. Yöntem ve Model Mimarisi

Sistem, transfer öğrenme (Transfer Learning) ve aşamalı ince ayar (Progressive Fine-Tuning) yaklaşımıyla tasarlanmıştır:

```
[ Girdi: 256x256x3 RGB Yüz Görseli ]
                   │
                   ▼
  [ Xception Omurgası (ImageNet Pre-trained) ]
    * Depthwise Separable Convolutions
    * Derinlik Ayrılabilir Evrişimler ile Düşük Parametre Sayısı
                   │
                   ▼
     [ Global Average Pooling 2D ]  --> Parametre patlamasını engeller
                   │
                   ▼
       [ Dense (256, ReLU) ]         --> Göreve özel ara öznitelik öğrenimi
                   │
                   ▼
        [ Dropout (Rate = 0.4) ]     --> Ezberlemeyi (Overfitting) engeller
                   │
                   ▼
     [ Dense (1, Sigmoid) Çıktı ]   --> 0: GERÇEK | 1: SAHTE
```

### ⚙️ İki Aşamalı Eğitim Protokolü
1. **Aşama 1 (Isınma / Warm-up - 3 Epoch):**  
   Xception omurgası dondurulmuş (`trainable = False`), sadece yeni eklenen sınıflandırma başlığı `Adam(lr = 1e-4)` optimizasyonu ile eğitilmiştir. Böylece hazır ağırlıklar rastgele gradyan şokundan korunmuştur.
2. **Aşama 2 (İnce Ayar / Fine-tuning - 10 Epoch):**  
   Xception mimarisinin son 20 katmanının kilidi açılarak çok düşük bir öğrenme oranı (`lr = 1e-5`) ile hedef veri setine hassas adaptasyon sağlanmıştır.

---

## 👁️ 3. Açıklanabilir Yapay Zeka (Explainable AI - Grad-CAM Entegrasyonu)

Derin öğrenme modelleri genellikle "kara kutu" (black-box) olarak nitelendirilir. Kritik güvenlik sistemlerinde modelin bir yüze neden "sahte" dediğinin gerekçelendirilmesi zorunludur.

Bu projede **Grad-CAM (Gradient-weighted Class Activation Mapping)** algoritması uygulanmıştır:
* Modelin son evrişim katmanı olan `block14_sepconv2_act` katmanının aktivasyon haritaları, sınıf çıktısının gradyanları ile ağırlıklandırılır.
* Çıktı olarak görselin üzerine **termal ısı haritası (JET Colormap)** basılarak yapay zekanın yüzdeki hangi kusura, yansımaya veya kenar keskinliğine odaklandığı şeffaf bir şekilde görselleştirilir.

---

## 📊 4. Deneysel Bulgular ve Performans Metrikleri

Model, eğitim sürecinde **hiç görmediği 750 adet bağımsız doğrulama görseli** üzerinde test edilmiş ve aşağıdaki sonuçlara ulaşılmıştır:

### Temel Performans Göstergeleri

| Metrik | Değer | Akademik Değerlendirme |
| :--- | :---: | :--- |
| **Genel Doğruluk (Validation Accuracy)** | **%93.33** | 750 test görselinde 700 doğru sınıflandırma |
| **Doğrulama Kaybı (Validation Loss)** | **0.1862** | Düşük kayıp, yüksek model güvenilirliği |
| **Eğitim Doğruluğu (Training Accuracy)** | **%90.15** | Sağlam öğrenme |
| **Aşırı Uyum (Overfitting) Riski** | **Yok** | Val Acc > Train Acc (Model genelleştirme yeteneğine sahip) |

### Sınıf Bazında Metrikler (Classification Report)

| Sınıf | Hassasiyet (Precision) | Duyarlılık (Recall) | F1-Skoru | Test Örnek Sayısı |
| :--- | :---: | :---: | :---: | :---: |
| **Gerçek (`gercek`)** | **%90.0** | **%97.4** | **0.94** | 388 |
| **Sahte (`sahte`)** | **%97.0** | **%89.0** | **0.93** | 362 |
| **Ağırlıklı Ortalama** | **%94.0** | **%93.3** | **0.93** | **750** |

### Karışıklık Matrisi (Confusion Matrix)

$$\begin{pmatrix} 378 & 10 \\ 40 & 322 \end{pmatrix}$$

* **Gerçek Yüzlerin Tespiti:** 388 gerçek bireyin **378'i doğru tespit edilmiştir** (Hata oranı sadece %2.6).
* **Sahte Yüz Alarmı Güvenilirliği:** Modelin "Sahte" kararı verdiği durumlarda doğruluk oranı **%97.0**'dır.

---

## 💻 5. Sistem Mimarisi ve Dosya Hiyerarşisi

```
├── dataset/                     # Eğitim ve test veri havuzu
│   ├── train/                   # 2.996 eğitim görseli (%50-%50 dengeli)
│   └── validation/              # 750 test görseli
├── models/                      # Çıktılar ve ağırlıklar
│   ├── best_model.keras         # Eğitilmiş en iyi derin öğrenme ağırlıkları
│   ├── labels.json              # Sınıf indeks haritası
│   ├── training_metrics.png     # Epoch bazlı Accuracy & Loss grafikleri
│   └── evaluation_report.md     # Sayısal metrik dökümü
├── config.py                    # Merkezi hiperparametre ve dizin yönetimi
├── train.py                     # 2 aşamalı model eğitim boru hattı
├── gradcam.py                   # Grad-CAM ısı haritası üretim motoru
├── predict.py                   # Bağımsız çıkarım ve tahmin modülü
├── app.py                       # Modern Gradio web arayüzü ve sunum paneli
├── live_cam.py                  # OpenCV gerçek zamanlı kamera HUD sistemi
├── baslat.bat                   # Web arayüzü tek tık başlatıcı
└── requirements.txt             # Bağımlılıklar
```

---

## 🚀 6. Çalıştırma Talimatları

### Web Arayüzünü Başlatma:
```bash
python app.py
```
veya Windows'ta **`baslat.bat`** dosyasına çift tıklayın. Tarayıcınızda açın:  
👉 **`http://127.0.0.1:7860`**

### Canlı Web Kamerası ile Test:
```bash
python live_cam.py
```
*(Kameradan anlık video akışı üzerinde gerçek zamanlı FPS ve güven oranı çizilir)*

---

## 💻 Geliştirici

**Emirhan Dağ**  
