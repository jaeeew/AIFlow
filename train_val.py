import os
import shutil
import random

# 원본 경로
image_dir = 'dataset/images'
mask_dir = 'dataset/masks'

# 출력 경로
output_base = 'dataset_split'
train_img_out = os.path.join(output_base, 'train/images')
train_mask_out = os.path.join(output_base, 'train/masks')
val_img_out = os.path.join(output_base, 'val/images')
val_mask_out = os.path.join(output_base, 'val/masks')

# 비율 설정
train_ratio = 0.8

# 디렉토리 생성
for path in [train_img_out, train_mask_out, val_img_out, val_mask_out]:
    os.makedirs(path, exist_ok=True)

# 이미지 파일 리스트 추출 (.jpg, .jpeg 포함)
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith('.png')]

# 이름 정리
image_dict = {os.path.splitext(f)[0]: f for f in image_files}
mask_dict = {os.path.splitext(f)[0].replace('_mask', ''): f for f in mask_files}

# 공통 이름 추출
valid_names = list(set(image_dict.keys()) & set(mask_dict.keys()))
random.shuffle(valid_names)

# 분할
split_idx = int(len(valid_names) * train_ratio)
train_names = valid_names[:split_idx]
val_names = valid_names[split_idx:]

# 복사 함수
def copy_files(file_names, img_dict, mask_dict, img_dst, mask_dst):
    for name in file_names:
        img_path = os.path.join(image_dir, img_dict[name])
        mask_path = os.path.join(mask_dir, mask_dict[name])

        if os.path.exists(img_path) and os.path.exists(mask_path):
            shutil.copy(img_path, os.path.join(img_dst, img_dict[name]))
            shutil.copy(mask_path, os.path.join(mask_dst, mask_dict[name]))
        else:
            print(f"[주의] 누락된 파일: {name}")

# 복사 실행
copy_files(train_names, image_dict, mask_dict, train_img_out, train_mask_out)
copy_files(val_names, image_dict, mask_dict, val_img_out, val_mask_out)

print(f"[완료] 총 {len(valid_names)}개 중 Train {len(train_names)}개, Val {len(val_names)}개로 분할 완료.")
