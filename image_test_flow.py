import cv2
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from clf_model import predict_material
from vinyl_model import detect_vinyl
from ctm_model import predict_dirty_level

def main():
    # ✅ 테스트할 이미지 경로
    image_path = '1.jpg'  # 이미지 경로 변경 가능
    image = Image.open(image_path).convert("RGB")

    print("1단계: 종이/플라스틱 분류")
    material = predict_material(image)

    if material == 'paper':
        print("종이입니다. 음식물이 묻지 않은 상자라면 재활용 가능합니다.")
        print("음식물(기름) 묻은 종이는 일반쓰레기에 버리세요.")
        return

    print("플라스틱으로 분류되었습니다.")

    print("2단계: 비닐 감지")
    vinyl_attached = detect_vinyl(image)

    if vinyl_attached:
        print("플라스틱 용기에 비닐이 감지되었습니다.")
        print("🚫 비닐을 제거하고 다시 이미지를 업로드 해주세요.")
        return

    print("비닐 없음.")

    print("3단계: 오염도 평가")
    contamination_level = predict_dirty_level(image)

    result_message = {
        'clean': '✅ 깨끗합니다. 재활용 가능합니다!',
        'slight': '⚠️ 살짝 오염되어 있습니다. 물로 헹군 후 재활용하세요.',
        'heavy': '🚫 심하게 오염되어 재활용 불가능합니다. 세척 후 이미지를 다시 업로드 해주세요.'
    }

    print(f"    오염도 결과: {result_message[contamination_level]}")

if __name__ == '__main__':
    main()