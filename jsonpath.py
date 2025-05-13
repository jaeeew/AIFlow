import json
import re
import os

# 입력 및 출력 JSON 경로
input_json_path = r'C:\AIFlow_1\labels\exported_labelstudio.json'
output_json_path = r'C:\AIFlow_1\labels\modified_labelstudio.json'

# JSON 파일 읽기
with open(input_json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# JSON 데이터에서 이미지 경로 수정
for task in data:
    original_path = task['data']['image']
    filename = os.path.basename(original_path)

    # 정규표현식으로 파일명에서 숫자만 추출 (예: 'c2c510b6-20.jpg' 또는 'c2c510b6-20.jpeg' → '20.jpg' 또는 '20.jpeg')
    match = re.search(r'(\d+)\.(jpg|jpeg)', filename)
    if match:
        # 숫자만 추출하여 새로운 파일명으로 설정
        new_filename = match.group(0)  # '20.jpg' 또는 '20.jpeg'
        task['data']['image'] = new_filename
    else:
        print(f"⚠️ 숫자를 추출할 수 없습니다: {filename}")

# 수정된 데이터를 새로운 JSON 파일로 저장
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("✅ 이미지 파일명을 숫자 기반으로 수정 완료!")
