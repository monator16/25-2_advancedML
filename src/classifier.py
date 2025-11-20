# classifier 학습 & 평가 

import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import timm # timm 임포트
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

# OUTPUT_DIR 설정
OUTPUT_DIR = 'classification_results'
os.makedirs(OUTPUT_DIR, exist_ok=True) # 폴더 생성
MODEL_NAME = 'resnet18'
saved_model_filename = f'best_classifier_{MODEL_NAME}.pth'
CLASSIFIER_PATH = os.path.join(OUTPUT_DIR, saved_model_filename)

# 하이퍼파라미터
CLASSIFIER_EPOCHS = 15
CLASSIFIER_LR = 1e-4
from config import (
    DEVICE, 
    NUM_CLASSES, BATCH_SIZE, DATA_DIR, CLASSES
)
from src.data import OASISDataset, transform_classification


class SimpleClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_classes=NUM_CLASSES, in_chans=1):
        super().__init__()
        # If grayscale + pretrained, load 3-channel pretrained model and adapt first conv
        if in_chans == 1:
            base = timm.create_model(model_name, pretrained=True, num_classes=1000, in_chans=3)
            # adapt first conv weights from 3->1 by averaging across RGB
            for name, mod in base.named_modules():
                if isinstance(mod, nn.Conv2d) and mod.in_channels == 3:
                    conv_name = name
                    conv_mod = mod
                    break
            else:
                # fallback: create single-channel model without pretrained conv
                self.model = timm.create_model(model_name, pretrained=False, num_classes=num_classes, in_chans=1)
                return

            w = conv_mod.weight.data
            w_gray = w.mean(dim=1, keepdim=True)
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=conv_mod.out_channels,
                kernel_size=conv_mod.kernel_size,
                stride=conv_mod.stride,
                padding=conv_mod.padding,
                dilation=conv_mod.dilation,
                groups=conv_mod.groups,
                bias=(conv_mod.bias is not None),
            )
            new_conv.weight.data = w_gray.clone()
            if conv_mod.bias is not None:
                new_conv.bias.data = conv_mod.bias.data.clone()

            parent = base
            *prefix, last = conv_name.split('.')
            for p in prefix:
                parent = getattr(parent, p)
            setattr(parent, last, new_conv)

            try:
                base.reset_classifier(num_classes)
            except Exception:
                if hasattr(base, 'fc'):
                    base.fc = nn.Linear(base.fc.in_features, num_classes)
                elif hasattr(base, 'classifier'):
                    base.classifier = nn.Linear(base.classifier.in_features, num_classes)

            self.model = base
        else:
            self.model = timm.create_model(model_name, pretrained=(in_chans == 3), num_classes=num_classes, in_chans=in_chans)

    def forward(self, x):
        return self.model(x)

def load_classifier(
    path=CLASSIFIER_PATH,
    model_name=MODEL_NAME
):
    """미리 학습된 분류기 모델을 로드하고 평가 모드로 설정합니다."""
    cls = SimpleClassifier(model_name=model_name, num_classes=NUM_CLASSES, in_chans=1, device=DEVICE).to(DEVICE)
    # always set eval and freeze params to be safe
    cls.eval()
    for p in cls.parameters():
        p.requires_grad = False

    try:
        ckpt = torch.load(path, map_location=DEVICE)
        cls.load_state_dict(ckpt)
        print(f"✅ Classifier loaded from {path}")
    except FileNotFoundError:
        print(f"⚠️ Classifier weights not found at {path}. Model initialized randomly.")
    except Exception as e:
        print(f"⚠️ Failed to load classifier weights from {path}: {e}. Using initialized model.")

    return cls

# -----------------------------------------------------------------
# 1. Classification 학습 및 검증 통합 함수
# -----------------------------------------------------------------

def train_classifier(data_dir=DATA_DIR):
    """
    분류기 모델을 훈련하고 각 에포크마다 검증 정확도를 평가하여 최적 모델을 저장합니다.
    """
    print("\n--- Starting Classifier Training & Validation ---")
    
    # 데이터 로더 준비: 💡 transform_classification 사용
    train_ds = OASISDataset(data_dir, split='train', transform=transform_classification)
    val_ds = OASISDataset(data_dir, split='val', transform=transform_classification)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 모델 초기화 
    model = SimpleClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CLASSIFIER_LR)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    
    for epoch in range(CLASSIFIER_EPOCHS):
        # 훈련 단계
        model.train()
        train_loss = 0.0
        # OASISDataset.__getitem__ 반환값: anchor, cond_onehot, label_idx, subj
        for img, _, label_idx, _ in tqdm(train_loader, desc=f"Classifier Ep {epoch+1} (Train)", leave=False):
            img, label_idx = img.to(DEVICE), label_idx.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(img), label_idx)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * img.size(0)

        # 검증 단계
        val_acc, val_loss, val_f1 = validate_classifier_epoch(model, val_loader, criterion)
        
        print(f"\n[Classifier] Ep {epoch+1} | Train Loss: {train_loss / len(train_ds):.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.2f}%")
        
        # 최적 모델 저장
        if val_acc > best_acc:
            best_acc = val_acc
            save_dir = os.path.dirname(CLASSIFIER_PATH)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), CLASSIFIER_PATH)
            print(f"🔥 Best classifier saved (Acc: {best_acc:.2f}%) at {CLASSIFIER_PATH}")
    
    print("\n--- Classifier Training Finished ---")
    return model


@torch.no_grad()
def validate_classifier_epoch(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    for img, _, label_idx, _ in data_loader:
        img, label_idx = img.to(DEVICE), label_idx.to(DEVICE)
        outputs = model(img)
        loss = criterion(outputs, label_idx)
        
        total_loss += loss.item() * img.size(0)
        
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(label_idx.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    labels_np = np.array(all_labels)
    preds_np = np.array(all_preds)
    
    accuracy = accuracy_score(labels_np, preds_np) * 100
    f1 = f1_score(labels_np, preds_np, average='weighted', zero_division=0) * 100
    
    return accuracy, avg_loss, f1

# -----------------------------------------------------------------
# 2. Classifier 성능 테스트 및 상세 메트릭 출력
# -----------------------------------------------------------------

@torch.no_grad()
def test_classifier_performance(data_dir=DATA_DIR):
    """
    저장된 분류기 모델을 로드하여 테스트 데이터셋에 대한 상세 평가 지표를 출력합니다.
    """
    # 1. 모델 로드 (load_classifier 재사용)
    classifier = load_classifier(CLASSIFIER_PATH)
    if classifier.training: # 로드 실패 시 load_classifier에서 출력하므로 여기서 재확인
        print("테스트를 진행할 분류기 모델을 로드할 수 없습니다.")
        return 0
        
    # 2. 테스트 데이터 로더 준비
    test_ds = OASISDataset(data_dir, split='test', transform=transform_classification) # 💡 transform_classification 사용
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 3. 평가 실행
    accuracy, true_labels, predictions = evaluate_classifier_for_metrics(classifier, test_loader)

    # 4. 결과 출력
    average_type = 'weighted' 
    precision = precision_score(true_labels, predictions, average=average_type, zero_division=0)
    recall = recall_score(true_labels, predictions, average=average_type, zero_division=0)
    f1 = f1_score(true_labels, predictions, average=average_type, zero_division=0)
    target_names = list(CLASSES.keys()) # 클래스 이름 사용

    print("\n--- Classifier Test Performance ---")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test Precision ({average_type}): {precision:.4f}")
    print(f"Test Recall ({average_type}): {recall:.4f}")
    print(f"Test F1-Score ({average_type}): {f1:.4f}")
    print("\n" + "-"*50)
    
    print("\nClassification Report:")
    # target_names는 CLASSES.keys()로 가져온 이름 순서에 맞춰야 함
    print(classification_report(true_labels, predictions, target_names=target_names, digits=4))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, predictions))
    
    return accuracy

@torch.no_grad()
def evaluate_classifier_for_metrics(model, data_loader): # 💡 criterion 제거 (손실 계산 불필요)
    """상세 지표를 위해 레이블과 예측값을 반환하는 평가 함수."""
    model.eval()
    all_labels = []
    all_preds = []
    
    for img, _, label_idx, _ in data_loader:
        img, label_idx = img.to(DEVICE), label_idx.to(DEVICE)
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)
        
        all_labels.extend(label_idx.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

    labels_np = np.array(all_labels)
    preds_np = np.array(all_preds)
    accuracy = accuracy_score(labels_np, preds_np) * 100
    
    return accuracy, labels_np, preds_np


if __name__ == '__main__':
    # 학습 및 평가 실행 (Colab 노트북에서 호출하거나 직접 스크립트 실행)
    print(f"DEVICE: {DEVICE}")
    print(f"CLASSIFIER_PATH: {CLASSIFIER_PATH}")
    
    # 1. 분류기 훈련 및 저장
    train_classifier() 

    # 2. 저장된 모델 로드하여 테스트 셋으로 최종 평가
    test_classifier_performance()