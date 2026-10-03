import os
import numpy as np
from PIL import Image, ImageDraw
import config

def generate_sample_images(output_dir, category, count=30):
    """
    Test amaçlı yapay (sentetik) görseller üretir:
    - Kategori 0 ('sinif_a'): Kırmızı zemin üzerinde beyaz daireler
    - Kategori 1 ('sinif_b'): Mavi zemin üzerinde sarı kareler
    """
    os.makedirs(output_dir, exist_ok=True)
    for i in range(count):
        img = Image.new("RGB", (config.IMG_WIDTH, config.IMG_HEIGHT), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        # Rastgele konum ve boyut
        margin = np.random.randint(20, 50)
        x0 = np.random.randint(10, 80)
        y0 = np.random.randint(10, 80)
        x1 = x0 + np.random.randint(80, 150)
        y1 = y0 + np.random.randint(80, 150)
        
        if category == "sinif_a":
            # Daire çizimi (Kırmızı tonlarında)
            draw.ellipse([x0, y0, x1, y1], fill=(220, 50, 50), outline=(150, 20, 20), width=3)
        else:
            # Dikdörtgen çizimi (Mavi tonlarında)
            draw.rectangle([x0, y0, x1, y1], fill=(40, 100, 220), outline=(20, 50, 150), width=3)
            
        file_path = os.path.join(output_dir, f"{category}_{i+1:03d}.jpg")
        img.save(file_path, quality=90)

def setup_demo_dataset_if_empty():
    """Dataset boşsa örnek veri seti oluşturur."""
    os.makedirs(config.TRAIN_DIR, exist_ok=True)
    os.makedirs(config.VAL_DIR, exist_ok=True)
    
    train_classes = [d for d in os.listdir(config.TRAIN_DIR) if os.path.isdir(os.path.join(config.TRAIN_DIR, d))]
    
    if len(train_classes) < 2:
        print(">> 'dataset/train' klasöründe veri bulunamadı.")
        print(">> Test amaçlı sentetik demo veri seti üretiliyor...")
        generate_sample_images(os.path.join(config.TRAIN_DIR, "sinif_a"), "sinif_a", count=30)
        generate_sample_images(os.path.join(config.TRAIN_DIR, "sinif_b"), "sinif_b", count=30)
        generate_sample_images(os.path.join(config.VAL_DIR, "sinif_a"), "sinif_a", count=10)
        generate_sample_images(os.path.join(config.VAL_DIR, "sinif_b"), "sinif_b", count=10)
        print(">> Demo veri seti hazırlandı: dataset/train ve dataset/validation")
    else:
        print(f">> Mevcut veri seti bulundu: {train_classes}")

if __name__ == "__main__":
    setup_demo_dataset_if_empty()
