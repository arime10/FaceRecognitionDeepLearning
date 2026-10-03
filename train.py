import os
import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import config
from create_sample_dataset import setup_demo_dataset_if_empty

def train_pipeline():
    print("=" * 60)
    print("    DERİN ÖĞRENME MODEL EĞİTİM BORU HATTI (PIPELINE)")
    print("=" * 60)

    # 1. Veri seti kontrolü
    if not os.path.exists(config.VAL_DIR):
        # Alternatif isimleri kontrol et (val, test)
        for alt in ["val", "test"]:
            alt_path = os.path.join(config.DATASET_DIR, alt)
            if os.path.exists(alt_path):
                config.VAL_DIR = alt_path
                break

    train_classes = [d for d in os.listdir(config.TRAIN_DIR) if os.path.isdir(os.path.join(config.TRAIN_DIR, d))] if os.path.exists(config.TRAIN_DIR) else []
    
    if len(train_classes) < 2:
        print(f"\n[HATA] '{config.TRAIN_DIR}' dizininde en az 2 sınıf klasörü bulunmalıdır.")
        print("Lütfen veri setinizi şu yapıda yerleştirin:")
        print("dataset/")
        print("  ├── train/")
        print("  │     ├── sinif_1/ (Görseller)")
        print("  │     └── sinif_2/ (Görseller)")
        print("  └── validation/ (veya test/)")
        print("        ├── sinif_1/ (Görseller)")
        print("        └── sinif_2/ (Görseller)\n")
        return

    print(f"\n[OK] Bulunan Sınıflar: {train_classes}")

    # 2. Veri Artırma ve Yükleyiciler
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    # Doğrulama verisinde artırma yapılmaz!
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode="binary",
        shuffle=True
    )

    validation_generator = val_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode="binary",
        shuffle=False  # Metriklerin karışmaması için shuffle=False zorunlu!
    )

    # Sınıf etiketlerini kaydet
    class_indices = train_generator.class_indices
    # {0: "sinif_a", 1: "sinif_b"} formatına çevir
    labels_map = {str(v): k for k, v in class_indices.items()}
    with open(config.LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels_map, f, indent=4, ensure_ascii=False)
    print(f"\n[OK] Sınıf haritası kaydedildi: {labels_map}")

    # 3. Model Mimarisi (Xception ile Transfer Learning)
    print("\n[INFO] Xception omurgası yükleniyor (ImageNet ağırlıkları)...")
    base_model = Xception(
        weights="imagenet",
        include_top=False,
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)
    )

    # Aşama 1 için temel modeli donduruyoruz (Warm-up)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = Dense(256, activation="relu", name="dense_head")(x)
    x = Dropout(config.DROPOUT_RATE, name="dropout_head")(x)
    predictions = Dense(1, activation="sigmoid", name="output_classification")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Aşama 1: Sadece yeni başlığı eğit
    print("\n" + "-" * 50)
    print("AŞAMA 1: Yeni Katmanların Isınma Eğitimi (Warm-Up)")
    print("-" * 50)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE_WARMUP),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=config.EPOCHS_WARMUP,
        verbose=1
    )

    # Aşama 2: İnce Ayar (Fine-Tuning)
    print("\n" + "-" * 50)
    print("AŞAMA 2: Xception Son Katmanlarıyla İnce Ayar (Fine-Tuning)")
    print("-" * 50)
    base_model.trainable = True
    # Son 20 katman hariç geri kalanını dondur
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE_FINETUNE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=config.MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=config.EPOCHS_FINETUNE,
        callbacks=callbacks,
        verbose=1
    )

    # 4. Doğrulama ve Raporlama
    print("\n" + "=" * 50)
    print("MODEL DEĞERLENDİRME VE METRİKLER")
    print("=" * 50)
    
    val_loss, val_acc = model.evaluate(validation_generator, verbose=0)
    print(f"\nFinal Doğrulama Kaybı (Val Loss)     : {val_loss:.4f}")
    print(f"Final Doğrulama Doğruluğu (Val Acc) : {val_acc:.4f} (%{val_acc*100:.2f})")

    # Karışıklık Matrisi ve Sınıflandırma Raporu
    validation_generator.reset()
    raw_predictions = model.predict(validation_generator, verbose=0)
    predicted_classes = (raw_predictions > config.THRESHOLD).astype("int32").flatten()
    true_classes = validation_generator.classes

    unique_labels = np.unique(np.concatenate([true_classes, predicted_classes]))
    target_names = [labels_map.get(str(i), f"Sınıf {i}") for i in unique_labels]

    cm = confusion_matrix(true_classes, predicted_classes)
    cr = classification_report(
        true_classes,
        predicted_classes,
        labels=unique_labels,
        target_names=target_names
    )

    print("\nKarışıklık Matrisi (Confusion Matrix):")
    print(cm)
    print("\nSınıflandırma Raporu (Classification Report):")
    print(cr)

    # Rapor dosyasını kaydet
    report_file = os.path.join(config.MODELS_DIR, "evaluation_report.md")
    with open(report_file, "w", encoding="utf-8") as rf:
        rf.write("# 📊 Model Değerlendirme Raporu\n\n")
        rf.write(f"- **Doğrulama Kaybı (Val Loss):** `{val_loss:.4f}`\n")
        rf.write(f"- **Doğrulama Doğruluğu (Val Accuracy):** `%{val_acc * 100:.2f}`\n\n")
        rf.write("### Karışıklık Matrisi (Confusion Matrix)\n```\n")
        rf.write(str(cm))
        rf.write("\n```\n\n### Sınıflandırma Raporu (Classification Report)\n```\n")
        rf.write(cr)
        rf.write("\n```\n")
    print(f"[OK] Metrik raporu dosyaya kaydedildi: {report_file}")

    # Modelin her durumda kaydedildiğinden emin ol
    model.save(config.MODEL_PATH)
    print(f"\n[OK] Model başarıyla kaydedildi: {config.MODEL_PATH}")

    # 5. Eğitim Metriklerini Görselleştir ve Kaydet
    plt.figure(figsize=(12, 5))
    
    # Doğruluk grafiği
    plt.subplot(1, 2, 1)
    acc = history.history.get("accuracy", [])
    val_acc_hist = history.history.get("val_accuracy", [])
    epochs_range = range(1, len(acc) + 1)
    plt.plot(epochs_range, acc, 'b-', label='Eğitim Başarımı')
    plt.plot(epochs_range, val_acc_hist, 'r--', label='Doğrulama Başarımı')
    plt.title('Model Doğruluğu (Accuracy)')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()
    plt.grid(True)

    # Kayıp grafiği
    plt.subplot(1, 2, 2)
    loss = history.history.get("loss", [])
    val_loss_hist = history.history.get("val_loss", [])
    plt.plot(epochs_range, loss, 'b-', label='Eğitim Kaybı')
    plt.plot(epochs_range, val_loss_hist, 'r--', label='Doğrulama Kaybı')
    plt.title('Model Kaybı (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(config.METRICS_PLOT_PATH)
    plt.close()
    print(f"\n[OK] Metrik grafiği kaydedildi: {config.METRICS_PLOT_PATH}")
    print(f"[OK] En iyi model kaydedildi: {config.MODEL_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    train_pipeline()
