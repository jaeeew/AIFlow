from PIL import Image
import os
import json

# 경로 설정
json_path = r"C:\AIFlow_1\labels\modified_labelstudio.json"  # Label Studio export한 JSON
image_dir = "dataset/images"
mask_dir = "dataset/masks"

# JSON 로드
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 라벨링된 이미지 ID 수집
labeled_ids = set()
for item in data:
    if item.get('annotations') and item['annotations'][0].get('result'):
        image_path = item['data']['image'].split('/')[-1]
        image_id = os.path.splitext(image_path)[0]
        labeled_ids.add(image_id)

# 이미지 디렉토리 내 모든 이미지 탐색
for file in os.listdir(image_dir):
    if not file.lower().endswith(('.jpg', '.jpeg')):
        continue

    image_id = os.path.splitext(file)[0]

    # 라벨링되지 않은 이미지에 대해 빈 마스크 생성
    if image_id not in labeled_ids:
        image_path = os.path.join(image_dir, file)
        img = Image.open(image_path)
        w, h = img.size
        mask = Image.new('L', (w, h), 0)  # 검정색 마스크
        mask.save(os.path.join(mask_dir, f"{image_id}_mask.png"))

print(f"[완료] 라벨링되지 않은 이미지에 대해 빈 마스크 생성 완료.")
