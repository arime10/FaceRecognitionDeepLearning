import os

# Temel Dizinler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "validation")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Model ve Etiket Yolları
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.keras")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")
METRICS_PLOT_PATH = os.path.join(MODELS_DIR, "training_metrics.png")

# Görüntü ve Eğitim Parametreleri
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 32
LEARNING_RATE_WARMUP = 1e-4
LEARNING_RATE_FINETUNE = 1e-5
EPOCHS_WARMUP = 3
EPOCHS_FINETUNE = 10
DROPOUT_RATE = 0.4
THRESHOLD = 0.5

# Xception'da Grad-CAM için kullanılacak son evrişim (conv) katmanı
LAST_CONV_LAYER_NAME = "block14_sepconv2_act"

# Gerekli dizinleri otomatik oluştur
os.makedirs(MODELS_DIR, exist_ok=True)
