from PIL import Image, ImageDraw
import json
import os

json_path = 'C:\AIFlow_1\labels\modified_labelstudio.json'
image_dir = 'C:/AIFlow_1/dataset/images'
output_mask_dir = 'C:/AIFlow_1/dataset/masks'
os.makedirs(output_mask_dir, exist_ok=True)

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    image_name = item['data']['image'].split('/')[-1]
    image_path = os.path.join(image_dir, image_name)

    if not os.path.exists(image_path):
        print(f"[오류] 원본 이미지가 존재하지 않음: {image_path}")
        continue

    # ✅ 'annotations' 키가 있고 비어있지 않은지 확인
    annotations = item.get('annotations')
    if not annotations or not annotations[0].get('result'):
        print(f"[스킵] 라벨링 없음: {image_name}")
        continue

    results = annotations[0]['result']
    width = results[0].get('original_width')
    height = results[0].get('original_height')

    if not width or not height:
        print(f"[스킵] 이미지 크기 정보 없음: {image_name}")
        continue

    # 마스크 초기화
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for result in results:
        if result['type'] == 'polygonlabels':
            points = result['value']['points']
            polygon = [(p[0] / 100 * width, p[1] / 100 * height) for p in points]
            draw.polygon(polygon, fill=255)

    mask_name = os.path.splitext(image_name)[0] + '_mask.png'
    mask.save(os.path.join(output_mask_dir, mask_name))
    print(f"[완료] {mask_name}")

