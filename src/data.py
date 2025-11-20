# src/data.py
import os
import random
from glob import glob
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import warnings
from .compute_stats import load_stats as _load_stats

# 💡 설정값 임포트
from config import DATA_DIR, IMAGE_SIZE, CLASSES, NUM_CLASSES, SEED

class OASISDataset(Dataset):
  def __init__(self, root_dir=DATA_DIR, transform=None, split='train', 
                 val_ratio=0.2, test_ratio=0.1, random_seed=SEED):
        self.transform = transform
        self.data = []
        self.classes = {'Non Demented': 0, 'Very mild Dementia': 1,
                        'Mild Dementia': 2} #'Moderate Dementia': 3 제외
        self.num_classes = len(self.classes)

        # 모든 이미지 수집 및 환자별 그룹화
        subject_to_imgs = defaultdict(list)
        subject_to_label = {} #환자별 대표 레이블 기록

        for class_name, label_idx in self.classes.items():
          class_dir = os.path.join(root_dir, class_name)
          if not os.path.isdir(class_dir):
            continue
          for img_path in glob(os.path.join(class_dir, '*.jpg')):
              subj_id = os.path.basename(img_path).split('_')[1]
              subject_to_imgs[subj_id].append((img_path, label_idx))
              subject_to_label[subj_id] = label_idx

        # 레이블별로 환자 묶기
        label_to_subjects = defaultdict(list)
        for subj_id, label_idx in subject_to_label.items():
            label_to_subjects[label_idx].append(subj_id)

        train_subjects = []
        val_subjects = []
        test_subjects = []

        random.seed(random_seed)

        # 각 레이블별로 따로 분할 진행

        for label_idx, subjects in label_to_subjects.items():
            random.shuffle(subjects)
            n_subj = len(subjects)

            # 비율에 따라 나눌 인원수 계산
            n_test = int(n_subj * test_ratio)
            n_val = int(n_subj * val_ratio)

            # 희귀 레이블 경우 조정
            # 최소 환자 2명인 moderate 은 학습만 시키고, 테스트는 x
            if n_subj >= 2 and n_val == 0:
                 n_val = 1 # 환자가 2명 이상이면 최소 1명은 검증셋으로

            # 남은 인원은 모두 Train으로
            n_train = n_subj - n_val - n_test


            test_subjects.extend(subjects[:n_test])
            val_subjects.extend(subjects[n_test : n_test + n_val])
            train_subjects.extend(subjects[n_test + n_val:])


        if split == 'train':
            target_subjects = train_subjects
        elif split == 'val':
            target_subjects = val_subjects
        elif split == 'test':
            target_subjects = test_subjects
        else:
            target_subjects = train_subjects + val_subjects + test_subjects

        # 최종 데이터 리스트 생성
        for subj_id, imgs in subject_to_imgs.items():
            if subj_id in target_subjects:
                for img_path, label_idx in imgs:
                    self.data.append((subj_id, img_path, label_idx))


        # 클래스별 분포 확인용 출력 (디버깅용)
        class_counts = defaultdict(int)
        for _, _, label_idx in self.data:
            class_counts[label_idx] += 1
        # print(f"   -> 클래스 분포: {dict(class_counts)}")




  def __len__(self):
      return len(self.data)

  def __getitem__(self, idx):
      subj, img_path, label_idx = self.data[idx]
      img = Image.open(img_path).convert("L") # 그레이스케일로 이미지 열기
      anchor = self.transform(img) if self.transform else transforms.ToTensor()(img)
      cond_onehot = torch.zeros(self.num_classes, dtype=torch.float32)
      cond_onehot[label_idx] = 1.0
      return anchor, cond_onehot, label_idx, subj # 이미지, 레이블원핫, 레이블, 환자 ID

# --- Transformation ---
transform_cvae = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    # (선택) CVAE 학습 시에는 정규화는 빼고 BCE/Sigmoid를 쓰거나, 아니면 Normalize(0.5, 0.5) 후 MSE를 사용
])

# Classifier 전용 Transformation (정규화 포함)
_stats_path = Path(__file__).resolve().parents[1] / 'train_norm_stats.json'
_loaded = _load_stats(_stats_path)
if _loaded is None:
    warnings.warn(
        f"Normalization stats not found at {_stats_path}. Using default mean=0.5,std=0.5. "
        "Run src/compute_stats.compute_mean_std on the training set in Colab and save to train_norm_stats.json in repo root.")
    _mean, _std = [0.5], [0.5]
else:
    _mean, _std = _loaded

transform_classification = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_mean, std=_std),
])