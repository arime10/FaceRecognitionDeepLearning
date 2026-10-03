import time
import cv2
import numpy as np
from PIL import Image

from predict import predict_image

def run_live_camera(camera_index=0, frame_skip=2):
    """
    Web kamerasından gerçek zamanlı video akışı alarak yapay zeka çıkarımı yapar.
    
    Args:
        camera_index: Kamera aygıt kimliği (varsayılan 0)
        frame_skip: Her N karede bir derin öğrenme modelini çalıştırır (performans optimizasyonu)
    """
    print("=" * 60)
    print("      CANLI KAMERA YAPAY ZEKA ÇIKARIM SİSTEMİ")
    print("=" * 60)
    print(">> Kameraya bağlanılıyor...")
    print(">> Çıkmak için pencere seçiliyken 'Q' veya 'ESC' tuşuna basın.\n")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[HATA] {camera_index} indeksli kameraya erişilemedi!")
        return

    # Kamera çözünürlüğünü 640x480 olarak ayarla
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_label = "Bekleniyor..."
    last_confidence = 0.0
    frame_count = 0
    fps = 0
    fps_start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[UYARI] Kameradan kare okunamadı!")
            break

        frame_count += 1
        h, w = frame.shape[:2]

        # Her N karede bir tahmin çalıştır
        if frame_count % frame_skip == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                res = predict_image(rgb_frame, with_gradcam=False)
                last_label = res["predicted_label"]
                last_confidence = res["confidence"]
            except Exception as e:
                last_label = f"Hata: {e}"

        # FPS hesaplama
        if time.time() - fps_start_time >= 1.0:
            fps = frame_count / (time.time() - fps_start_time)
            frame_count = 0
            fps_start_time = time.time()

        # HUD (Bilgi Paneli) Çizimi
        # Üst şerit için yarı şeffaf arka plan
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Renk belirleme: Yüksek güvenilirlikte yeşil, düşükte turuncu/kırmızı
        if last_confidence >= 80.0:
            status_color = (0, 220, 0)
        elif last_confidence >= 50.0:
            status_color = (0, 165, 255)
        else:
            status_color = (0, 0, 255)

        # Metinleri yaz
        cv2.putText(frame, f"Tahmin: {last_label}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(frame, f"Guven: %{last_confidence:.1f}", (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

        # Güven çubuğu (Progress bar)
        bar_x = 220
        bar_w = int((w - bar_x - 140) * (last_confidence / 100.0))
        cv2.rectangle(frame, (bar_x, 50), (w - 140, 68), (60, 60, 60), 1)
        if bar_w > 0:
            cv2.rectangle(frame, (bar_x, 50), (bar_x + bar_w, 68), status_color, -1)

        cv2.imshow("Canli Yapay Zeka Gozu (Xception AI)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(">> Kamera oturumu kapatıldı.")

if __name__ == "__main__":
    run_live_camera()
