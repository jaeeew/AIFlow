import cv2
import numpy as np
import os
import csv
import glob

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

#image_path = r'C:/AIFlow_1/dataset/test/images/2.jpg'
#mask_path = r'C:/AIFlow_1/dataset/test/masks/2_mask.png'
#level, score, area_ratio, color_intensity = contamination_score(image_path, mask_path)
#print(f"[{os.path.basename(image_path)}]")
#print(f"오염도 레벨: {level} / 점수: {score:.4f}")
#print(f"→ 면적 비율: {area_ratio:.4f}, 색 진하기: {color_intensity:.4f}")

# 경로 설정
image_dir = r'C:/AIFlow_1/dataset/test/images'
mask_dir = r'C:/AIFlow_1/dataset/test/masks'
output_csv = 'contamination_results.csv'

# CSV 파일 초기화
results = []

# 이미지 폴더 내 모든 이미지 파일에 대해 오염도 계산
for filename in os.listdir(image_dir):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(image_dir, filename).replace("\\", "/")
        mask_path = os.path.join(mask_dir, os.path.splitext(filename)[0] + "_mask.png").replace("\\", "/")

        level, score, area_ratio, color_intensity = contamination_score(image_path, mask_path)

        if level is not None:
            results.append((filename, level, score, area_ratio, color_intensity))

# 결과를 CSV 파일로 저장
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'contamination_level', 'contamination_score', 'area_ratio', 'color_intensity'])
    writer.writerows(results)

print(f"[완료] {len(results)}개의 이미지에 대해 오염도를 계산하고 CSV 파일로 저장했습니다.")


