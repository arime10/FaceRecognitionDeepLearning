import os
import sys
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input

import config
import cv2
import config
from gradcam import make_gradcam_heatmap, superimpose_gradcam

# Global model ve etiket önbelleği (Hızlı çağrılar için)
_MODEL = None
_LABELS = None
_FACE_CASCADE = None

def get_face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _FACE_CASCADE

def detect_and_crop_face(pil_image, margin_ratio=0.25):
    """
    Çoklu Açı & Strateji ile Yüz Tespiti:
    1. Standart düz yüz taraması (Frontal Default)
    2. Detaylı yüz taraması (Frontal Alt2)
    3. Eğik ve yan yüz taraması (Profile Face - hem normal hem ayna)
    4. Bulunamazsa uzun dikey fotoğraflar için Akıllı Kafa Odaklanması (Smart Head Crop)
    """
    rgb = np.array(pil_image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = rgb.shape[:2]

    cascade_files = [
        ("haarcascade_frontalface_default.xml", False),
        ("haarcascade_frontalface_alt2.xml", False),
        ("haarcascade_profileface.xml", False),
        ("haarcascade_profileface.xml", True) # Ayna çevrilmiş (diğer yana eğik)
    ]

    found_faces = []
    for c_file, do_flip in cascade_files:
        test_gray = cv2.flip(gray, 1) if do_flip else gray
        cas = cv2.CascadeClassifier(cv2.data.haarcascades + c_file)
        faces = cas.detectMultiScale(test_gray, scaleFactor=1.08, minNeighbors=3, minSize=(50, 50))
        
        if len(faces) > 0:
            if do_flip:
                # Koordinatları orijinal yöne geri çevir
                faces = [[W - (x + w), y, w, h] for (x, y, w, h) in faces]
            
            # Yüzün görüntünün üst 75'lik diliminde olduğunu doğrula
            valid = [f for f in faces if f[1] < H * 0.75]
            if valid:
                found_faces = valid
                break

    if len(found_faces) > 0:
        # En büyük yüzü seç
        found_faces = sorted(found_faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = found_faces[0]

        pad_x = int(w * margin_ratio)
        pad_y = int(h * margin_ratio * 1.3)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(W, x + w + pad_x)
        y2 = min(H, y + h + pad_x)

        return pil_image.crop((x1, y1, x2, y2)), True

    # 4. Hiçbir algoritma bulamazsa ama fotoğraf uzun dikey (9:16) bir portreyse:
    # Üst %48'lik kafa/omuz bölgesine akıllı odaklanma uygula
    if H > W * 1.25:
        crop_h = int(H * 0.48)
        return pil_image.crop((0, 0, W, crop_h)), True

    return pil_image, False

def load_resources():
    global _MODEL, _LABELS
    if _MODEL is None:
        if not os.path.exists(config.MODEL_PATH):
            raise FileNotFoundError(
                f"Model bulunamadı: '{config.MODEL_PATH}'. Lütfen önce 'python train.py' çalıştırarak modeli eğitin."
            )
        print(f"[INFO] Model yükleniyor: {config.MODEL_PATH}")
        _MODEL = tf.keras.models.load_model(config.MODEL_PATH)

    if _LABELS is None:
        if os.path.exists(config.LABELS_PATH):
            with open(config.LABELS_PATH, "r", encoding="utf-8") as f:
                _LABELS = json.load(f)
        else:
            _LABELS = {"0": "Sınıf 0", "1": "Sınıf 1"}
    return _MODEL, _LABELS

def predict_image(image_input, with_gradcam=True, auto_crop=False):
    """
    Tek bir görseli analiz eder.
    
    Args:
        image_input: Dosya yolu (str), PIL.Image nesnesi veya RGB numpy dizisi.
        with_gradcam: True ise Grad-CAM ısı haritası görüntüsünü de döndürür.
        auto_crop: True ise OpenCV ile önce yüzü tespit edip kırpar.
    """
    model, labels = load_resources()

    # PIL Image formatına dönüştür
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise ValueError("Desteklenmeyen görsel formatı!")

    face_detected = False
    if auto_crop:
        pil_img, face_detected = detect_and_crop_face(pil_img)

    # 256x256 boyutlandırma
    resized_pil = pil_img.resize((config.IMG_WIDTH, config.IMG_HEIGHT))
    img_array = np.array(resized_pil, dtype=np.float32)
    original_rgb = np.array(resized_pil, dtype=np.uint8)

    # Xception ön işleme [-1, 1]
    preprocessed_array = preprocess_input(img_array)
    batch_array = np.expand_dims(preprocessed_array, axis=0)

    # Model Tahmini
    pred_prob = float(model.predict(batch_array, verbose=0)[0][0])
    
    prob_class_1 = pred_prob
    prob_class_0 = 1.0 - pred_prob

    label_0 = labels.get("0", "Sınıf 0")
    label_1 = labels.get("1", "Sınıf 1")

    if prob_class_1 >= config.THRESHOLD:
        predicted_label = label_1
        confidence = prob_class_1 * 100
    else:
        predicted_label = label_0
        confidence = prob_class_0 * 100

    gradcam_overlay = None
    if with_gradcam:
        try:
            heatmap = make_gradcam_heatmap(batch_array, model)
            gradcam_overlay = superimpose_gradcam(original_rgb, heatmap)
        except Exception as e:
            print(f"[UYARI] Grad-CAM üretilemedi: {e}")

    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "probabilities": {
            label_0: prob_class_0,
            label_1: prob_class_1
        },
        "gradcam_image": gradcam_overlay,
        "cropped_face": np.array(pil_img),
        "face_detected": face_detected
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python predict.py <resim_yolu>")
        sys.exit(1)
        
    test_img_path = sys.argv[1]
    result = predict_image(test_img_path, with_gradcam=True)
    print("\n--- TAHMİN SONUCU ---")
    print(f"Tahmin Edilen Sınıf : {result['predicted_label']}")
    print(f"Güven Oranı          : %{result['confidence']:.2f}")
    print(f"Detaylı Olasılıklar  : {result['probabilities']}")
    if result["gradcam_image"] is not None:
        out_name = "gradcam_output.jpg"
        Image.fromarray(result["gradcam_image"]).save(out_name)
        print(f"Grad-CAM görseli kaydedildi: {out_name}")
