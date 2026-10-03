import os
import gradio as gr
from PIL import Image

import config
from predict import predict_image

def crop_image_by_pct(img, top, bottom, left, right):
    if img is None:
        return None
    w, h = img.size
    t = max(0, min(100, int(top)))
    b = max(0, min(100, int(bottom)))
    l = max(0, min(100, int(left)))
    r = max(0, min(100, int(right)))
    if t >= b:
        b = min(100, t + 5)
    if l >= r:
        r = min(100, l + 5)
    x1 = int(w * l / 100.0)
    y1 = int(h * t / 100.0)
    x2 = int(w * r / 100.0)
    y2 = int(h * b / 100.0)
    x2 = max(x1 + 10, x2)
    y2 = max(y1 + 10, y2)
    return img.crop((x1, y1, x2, y2))

def on_crop_mode_change(mode, img, t, b, l, r):
    is_manual = (mode == "✂️ Manuel Kırpma (Sürgüler)")
    preview = crop_image_by_pct(img, t, b, l, r) if is_manual and img is not None else None
    return gr.update(visible=is_manual), preview

def set_center_preset(img):
    t, b, l, r = 10, 65, 20, 80
    preview = crop_image_by_pct(img, t, b, l, r) if img is not None else None
    return t, b, l, r, preview

def set_reset_preset(img):
    t, b, l, r = 0, 100, 0, 100
    return t, b, l, r, img

def on_image_uploaded(img, mode, t, b, l, r):
    is_manual = (mode == "✂️ Manuel Kırpma (Sürgüler)")
    if is_manual and img is not None:
        return crop_image_by_pct(img, t, b, l, r)
    return None

def analyze_uploaded_image(img, crop_mode_choice, t, b, l, r):
    if img is None:
        return "Lütfen analiz edilecek bir görsel yükleyin.", None, None, None

    if crop_mode_choice == "✂️ Manuel Kırpma (Sürgüler)":
        img_to_use = crop_image_by_pct(img, t, b, l, r)
        auto_crop = False
        mode_text = f"*(✂️ Manuel Kırpılan Alan Analiz Edildi: Y:%{t}-%{b}, X:%{l}-%{r})*"
    elif crop_mode_choice == "🤖 Otomatik Yüz Kırpma (Önerilen)":
        img_to_use = img
        auto_crop = True
        mode_text = "*(✅ Yüz otomatik tespit edildi ve odaklanıldı)*"
    else:
        img_to_use = img
        auto_crop = False
        mode_text = "*(🖼️ Orijinal kadraj kırpılmadan analiz edildi)*"

    if img_to_use is None:
        return "Görsel işlenemedi.", None, None, None

    res = predict_image(img_to_use, with_gradcam=True, auto_crop=auto_crop)

    if auto_crop and not res.get("face_detected"):
        mode_text = "*(Yüz algoritması bulunamadı, orijinal görsel işlendi)*"

    summary_text = (
        f"### 🎯 Sonuç: **{res['predicted_label'].upper()}**\n"
        f"**Güven Oranı:** `%{res['confidence']:.2f}`\n\n"
        f"{mode_text}\n"
    )

    gradcam_img = res["gradcam_image"]
    cropped_face = res.get("cropped_face")
    return summary_text, res["probabilities"], cropped_face, gradcam_img

def analyze_webcam_image(cam_img, auto_crop=True):
    if cam_img is None:
        return "Lütfen kameradan bir fotoğraf çekin.", None, None, None
    res = predict_image(cam_img, with_gradcam=True, auto_crop=auto_crop)
    summary_text = (
        f"### 🎯 Sonuç: **{res['predicted_label'].upper()}**\n"
        f"**Güven Oranı:** `%{res['confidence']:.2f}`\n"
    )
    return summary_text, res["probabilities"], res.get("cropped_face"), res["gradcam_image"]

def get_metrics_image():
    if os.path.exists(config.METRICS_PLOT_PATH):
        return Image.open(config.METRICS_PLOT_PATH)
    return None

def get_readme_content():
    readme_path = os.path.join(config.BASE_DIR, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "README dokümantasyonu bulunamadı."

# Özel CSS: Sade ve temiz stil
custom_css = """
footer { display: none !important; visibility: hidden !important; }
a[href*="gradio.app"], a[href*="api"] { display: none !important; }
"""

with gr.Blocks(title="Sahte Yüz Tespit Sistemi") as demo:
    gr.Markdown("# 🛡️ Sahte Yüz ve Deepfake Tespit Sistemi")

    with gr.Tabs():
        # TAB 1: Görsel Yükleme ve Analiz
        with gr.TabItem("🖼️ Fotoğraf Analizi ve Grad-CAM"):
            with gr.Row():
                with gr.Column(scale=1):
                    crop_mode = gr.Radio(
                        choices=[
                            "🤖 Otomatik Yüz Kırpma (Önerilen)",
                            "✂️ Manuel Kırpma (Sürgüler)",
                            "🖼️ Kırpma Yapma (Orijinal)"
                        ],
                        value="🤖 Otomatik Yüz Kırpma (Önerilen)",
                        label="Kırpma Yöntemi"
                    )

                    input_img = gr.Image(
                        type="pil",
                        label="Fotoğraf Yükleyin"
                    )

                    # Manuel Kırpma Paneli
                    with gr.Group(visible=False) as manual_crop_group:
                        gr.Markdown(
                            "💡 **Manuel Kırpma:** Aşağıdaki sürgüleri kaydırarak alanı ayarlayabilirsiniz. "
                            "Kırpılan yüz anında alttaki önizlemede görünür."
                        )
                        preview_cropped = gr.Image(
                            type="pil",
                            label="✂️ Kırpılan Alan Önizlemesi",
                            interactive=False
                        )
                        with gr.Row():
                            slider_top = gr.Slider(0, 100, value=10, step=1, label="⬆️ Üst Sınır (%)")
                            slider_bottom = gr.Slider(0, 100, value=65, step=1, label="⬇️ Alt Sınır (%)")
                        with gr.Row():
                            slider_left = gr.Slider(0, 100, value=20, step=1, label="⬅️ Sol Sınır (%)")
                            slider_right = gr.Slider(0, 100, value=80, step=1, label="➡️ Sağ Sınır (%)")
                        with gr.Row():
                            btn_center_face = gr.Button("🎯 Standart Yüz Kadrajı (%10-%65)", size="sm")
                            btn_reset_crop = gr.Button("🔄 Tam Görsel (%0-%100)", size="sm")

                    btn_predict = gr.Button("🔍 Analiz Et", variant="primary", size="lg")

                with gr.Column(scale=1):
                    output_summary = gr.Markdown("Analiz sonuçları burada görüntülenecektir.")
                    output_labels = gr.Label(num_top_classes=2, label="Tahmin Olasılıkları")
                    with gr.Row():
                        output_cropped = gr.Image(type="numpy", label="✂️ Analiz Edilen Yüz / Alan")
                        output_gradcam = gr.Image(type="numpy", label="👁️ Grad-CAM Isı Haritası")

            # Kırpma modu değişimi
            crop_mode.change(
                fn=on_crop_mode_change,
                inputs=[crop_mode, input_img, slider_top, slider_bottom, slider_left, slider_right],
                outputs=[manual_crop_group, preview_cropped]
            )

            # Fotoğraf yüklendiğinde önizlemeyi besle
            input_img.change(
                fn=on_image_uploaded,
                inputs=[input_img, crop_mode, slider_top, slider_bottom, slider_left, slider_right],
                outputs=preview_cropped
            )

            # Sürgüler değiştikçe önizlemeyi anında güncelle
            for s in [slider_top, slider_bottom, slider_left, slider_right]:
                s.change(
                    fn=crop_image_by_pct,
                    inputs=[input_img, slider_top, slider_bottom, slider_left, slider_right],
                    outputs=preview_cropped
                )

            # Hızlı butonlar
            btn_center_face.click(
                fn=set_center_preset,
                inputs=[input_img],
                outputs=[slider_top, slider_bottom, slider_left, slider_right, preview_cropped]
            )
            btn_reset_crop.click(
                fn=set_reset_preset,
                inputs=[input_img],
                outputs=[slider_top, slider_bottom, slider_left, slider_right, preview_cropped]
            )

            # Analiz butonu
            btn_predict.click(
                fn=analyze_uploaded_image,
                inputs=[input_img, crop_mode, slider_top, slider_bottom, slider_left, slider_right],
                outputs=[output_summary, output_labels, output_cropped, output_gradcam]
            )

        # TAB 2: Kamera / Canlı Çekim
        with gr.TabItem("📷 Web Kamerası ile Test"):
            with gr.Row():
                with gr.Column(scale=1):
                    cam_img = gr.Image(sources=["webcam"], type="pil", label="Kameradan Fotoğraf Çek")
                    chk_cam_crop = gr.Checkbox(label="🎯 Yüzü Otomatik Kırp ve Odaklan", value=True)
                    btn_cam = gr.Button("📸 Çekilen Görseli Analiz Et", variant="primary")
                with gr.Column(scale=1):
                    cam_summary = gr.Markdown("Kamera sonuçları burada görüntülenecektir.")
                    cam_labels = gr.Label(num_top_classes=2, label="Kamera Olasılıkları")
                    with gr.Row():
                        cam_cropped = gr.Image(type="numpy", label="✂️ Kırpılan Yüz")
                        cam_gradcam = gr.Image(type="numpy", label="👁️ Grad-CAM Odak Haritası")

            btn_cam.click(
                fn=analyze_webcam_image,
                inputs=[cam_img, chk_cam_crop],
                outputs=[cam_summary, cam_labels, cam_cropped, cam_gradcam]
            )

        # TAB 3: Model Metrikleri ve Başarım
        with gr.TabItem("📊 Sonuçlar ve Metrikler"):
            gr.Markdown(
                """
                ### 📈 750 Bağımsız Test Görseli Üzerindeki Performans
                | Metrik | Gerçek (`gercek`) | Sahte (`sahte`) | Genel Ortalama |
                | :--- | :---: | :---: | :---: |
                | **Hassasiyet (Precision)** | %90.0 | **%97.0** | %94.0 |
                | **Duyarlılık (Recall)** | **%97.4** | %89.0 | %93.3 |
                | **F1-Skoru** | **0.94** | **0.93** | **0.93** |
                | **Genel Doğruluk (Accuracy)** | - | - | **%93.33** |
                """
            )
            metrics_display = gr.Image(
                value=get_metrics_image(),
                label="Eğitim ve Doğrulama Grafiği (Accuracy & Loss)"
            )
            btn_refresh_metrics = gr.Button("🔄 Grafiği Yenile")
            btn_refresh_metrics.click(fn=get_metrics_image, outputs=metrics_display)

        # TAB 4: Proje Dokümantasyonu
        with gr.TabItem("📄 Dokümantasyon"):
            gr.Markdown(get_readme_content())

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        css=custom_css
    )
