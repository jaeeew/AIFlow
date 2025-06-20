from flask import Flask, request, jsonify
import os
from PIL import Image

# 모델 추론 함수 import
from clf_model import predict_material
from vinyl_model import detect_vinyl
from ctm_model import predict_dirty_level

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400

    image = request.files['image']
    save_path = os.path.join(UPLOAD_FOLDER, '1.jpg')
    image.save(save_path)
    print(f'✅ 저장 완료: {save_path}')

    try:
        image = Image.open(save_path).convert("RGB")

        # 🔎 1단계: 재질 분류
        print("🔎 1단계: 종이/플라스틱 분류")
        material = predict_material(image)

        if material == 'paper':
            return jsonify({'result': 'paper'})

        print("🧴 플라스틱으로 분류되었습니다.")

        # 🔎 2단계: 비닐 감지
        print("🔎 2단계: 비닐 감지")
        vinyl_result = detect_vinyl(image)
        if vinyl_result == 'vinyl':
            return jsonify({'result': 'plastic_with_vinyl'})

        print("✅ 비닐 없음. 다음 단계 진행")

        # 🔎 3단계: 오염도 평가
        print("🔎 3단계: 오염도 평가")
        contamination_level = predict_dirty_level(image)

        contamination_messages = {
            'clean',
            'slight',
            'heavy'
        }

        return jsonify({'result': f'plastic_{contamination_level}'})

    except Exception as e:
        print(f'❌ 예측 중 오류: {e}')
        return jsonify({'error': '예측 실패', 'detail': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)