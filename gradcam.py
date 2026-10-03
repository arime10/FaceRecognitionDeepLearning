import cv2
import numpy as np
import tensorflow as tf
import config

def get_last_conv_layer(model, preferred_name=config.LAST_CONV_LAYER_NAME):
    """
    Model içindeki son evrişim (conv) katmanını bulur.
    İsme göre arar; bulamazsa son 4D tensör üreten katmanı seçer.
    """
    try:
        return model.get_layer(preferred_name)
    except (ValueError, KeyError):
        pass

    # Bulunamazsa tersten tarayarak 4D çıktı veren katmanı bul
    for layer in reversed(model.layers):
        if hasattr(layer, "output_shape") and len(layer.output_shape) == 4:
            return layer
        # Eğer layer bir submodel ise içinde ara
        if hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if hasattr(sub_layer, "output_shape") and len(sub_layer.output_shape) == 4:
                    return sub_layer
    raise ValueError("Model içinde uygun bir evrişim (conv) katmanı bulunamadı.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=config.LAST_CONV_LAYER_NAME):
    """
    Görüntü ve model için Grad-CAM aktivasyon ısı haritasını üretir.
    """
    last_conv_layer = get_last_conv_layer(model, last_conv_layer_name)

    # Girdiden son evrişim katmanı aktivasyonuna ve son tahmine giden yardımcı model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_value = predictions[0][0]
        # İkili sınıflandırma: O anki tahmin yönünde gradyan al
        if pred_value >= config.THRESHOLD:
            score = pred_value
        else:
            score = 1.0 - pred_value

    # Son evrişim katmanının aktivasyonlarına göre gradyanları hesapla
    grads = tape.gradient(score, conv_outputs)
    
    # Her kanal için ortalama gradyan önemini çıkar (Global Average Pooling)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Ağırlıklı kombinasyon
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU ve Normalizasyon (0 ile 1 arası)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def superimpose_gradcam(original_img_np, heatmap, alpha=0.45):
    """
    Isı haritasını orijinal fotoğrafın üzerine yarı şeffaf renkli olarak bindirir.
    """
    # Isı haritasını orijinal görselin boyutuna yeniden boyutlandır
    h, w = original_img_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # 0-255 aralığına çek ve JET renk paleti uygula (Sıcak renkler: Kırmızı/Sarı)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    jet = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)

    # Orijinal resim ile birleştir
    superimposed = (jet * alpha) + (original_img_np * (1.0 - alpha))
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    return superimposed
