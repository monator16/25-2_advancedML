# config.py
import torch
from torchvision import transforms

# --- 하드웨어/시드 설정 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# --- 경로 설정 ---
# Colab 경로
DATA_DIR = '/content/drive/MyDrive/25_AML_OASIS_dataset/input'
# 로컬 경로
CVAE_MODEL_PATH = "saved_models/best_cvae.pth"
# 로컬 경로 - classifier 관련 경로
CLASSIFIER_PATH = "saved_models/best_classifier_resnet18.pth"
OUTPUT_FOLDER = "results/GEN_SAMPLES"

# --- 모델 및 데이터 설정 ---
IMAGE_SIZE = 224
IMAGE_CHANNEL = 1
LATENT_DIM = 128
CLS_DIM = 32 # z_class_raw 차원
NUM_CLASSES = 3
CLASSES = {0: 'Non Demented', 1: 'Very mild Dementia', 2: 'Mild Dementia'}
CLASS_NAMES_MAP = {v: k for k, v in CLASSES.items()}

# --- 학습 하이퍼파라미터 ---
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
BATCH_SIZE = 32

# --- Loss weights ---
BETA_KLD_WEIGHT = 2.0
BCE_WEIGHT = 1.0        # Reconstruction Loss (BCE/L1) 가중치
LAMBDA_LPIPS = 1.0      # LPIPS 가중치
W_CENTER = 10.0         # Latent Clustering Center Loss 가중치
W_SEPARATION = 5.0      # Latent Clustering Separation Loss 가중치
MARGIN = 2.0            # Separation Loss 마진
CLASSIFIER_LOSS_WEIGHT = 2.0


