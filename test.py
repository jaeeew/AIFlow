import cv2
import torch
import numpy as np
import os
import csv
from torchvision import transforms
from UNet import UNet  # 학습된 U-Net 모델을 가져온다고 가정

# UNet.py
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# U-Net 모델 로드 (이미 학습된 모델)
model = UNet()  # 모델 정의
#model.load_state_dict(torch.load('unet_model.pth'))  # 학습된 가중치 파일
model.eval()  # 평가 모드

# 오염도 계산 함수 (기존 코드 그대로 사용)
def calculate_color_intensity(image_path, mask_path):
    """HSV 기준 색 진하기 계산"""
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        return 0.0

    mask_bin = mask > 127
    if not np.any(mask_bin):
        return 0.0  # 오염 없음

    masked_pixels = image[mask_bin]
    hsv = cv2.cvtColor(masked_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)

    saturation = hsv[:, 1] / 255.0  # 채도
    value = hsv[:, 2] / 255.0       # 명도

    # 색 진하기: 채도 × (1 - 명도)
    color_intensity = np.mean(saturation * (1 - value))
    return color_intensity

def calculate_area_ratio(mask_path):
    """오염 면적 비율 계산"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0.0
    return np.sum(mask > 127) / mask.size

def contamination_score(image_path, mask_path, weight_area=0.6, weight_color=0.4):
    """면적과 색 진하기 기반 오염 점수 및 등급 계산"""
    area_ratio = calculate_area_ratio(mask_path)
    color_intensity = calculate_color_intensity(image_path, mask_path)

    score = weight_area * area_ratio + weight_color * color_intensity

    # 오염도 0~3단계 분류
    if score < 0.01:
        level = 0  # 매우 깨끗함
    elif score < 0.1:
        level = 1  # 약간 오염
    elif score < 0.2:
        level = 2  # 보통 오염
    else:
        level = 3  # 심한 오염

    return level, score, area_ratio, color_intensity

# 새로운 이미지에 대해 U-Net 모델로 마스크 생성
def generate_mask(image_path):
    """U-Net 모델을 사용하여 새로운 이미지에 대해 마스크를 생성"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB로 변환
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),  # 모델에 맞는 크기로 이미지 리사이즈
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  # 배치 차원 추가

    # 예측
    with torch.no_grad():
        output = model(image)
        output = output.squeeze(0).cpu().numpy()  # 배치 차원 제거
        output = (output > 0.5).astype(np.uint8)  # 확률을 이진 마스크로 변환

    # 마스크 크기 원복
    original_image = cv2.imread(image_path)
    mask = cv2.resize(output, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    return mask

# 경로 설정
image_dir = r'C:/AIFlow_1/dataset/test/dirty'
mask_dir = r'C:/AIFlow_1/dataset/test/dirty/masks'
output_csv = 'contamination_results.csv'

# CSV 파일 초기화
results = []

# 이미지 폴더 내 모든 이미지 파일에 대해 오염도 계산
for filename in os.listdir(image_dir):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(image_dir, filename).replace("\\", "/")

        # U-Net을 사용하여 새로운 이미지에 대해 마스크 생성
        generated_mask = generate_mask(image_path)

        # 생성된 마스크를 파일로 저장
        mask_path = os.path.join(mask_dir, os.path.splitext(filename)[0] + "_mask.png").replace("\\", "/")
        cv2.imwrite(mask_path, generated_mask)

        # 오염도 계산
        level, score, area_ratio, color_intensity = contamination_score(image_path, mask_path)

        # 결과 저장
        results.append((filename, level, score, area_ratio, color_intensity))

# 결과를 CSV 파일로 저장
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'contamination_level', 'contamination_score', 'area_ratio', 'color_intensity'])
    writer.writerows(results)

print(f"[완료] {len(results)}개의 이미지에 대해 오염도를 계산하고 CSV 파일로 저장했습니다.")
