# 🧠 AI_MODEL - 폐기물 이미지 분류 및 세분화 모델

이 프로젝트는 폐기물 사진을 자동으로 분석하여 **재활용 가능 여부**를 판단하는 AI 모델들로 구성되어 있습니다.  
아래와 같은 3단계 과정을 통해 이미지 분석을 수행합니다:

1. 🔍 **재질 분류**: 종이 / 플라스틱
2. 🧻 **비닐 감지**: 플라스틱에 비닐이 붙어있는지 판단
3. 💧 **오염도 평가**: 오염된 정도(깨끗함 / 약간 오염 / 심하게 오염) 분석

---

## 📁 디렉토리 구조

```
AI_model/
│
├── train_classification.py       # 재질 분류 (plastic vs paper) 학습 스크립트
├── test_classification.py        # 분류 모델 테스트
├── model_classification.py       # EfficientNet 기반 분류 모델 정의
│
├── train_contamination.py        # 오염도 segmentation 학습
├── test_contamination.py         # 오염도 테스트 및 등급 분류
├── model_contamination.py        # UNet + HybridLoss (BCE + Dice + Focal)
│
├── train_vinyl.py                # 비닐 감지 segmentation 학습
├── test_vinyl.py                 # 비닐 여부 테스트 및 판단
├── model_vinyl.py                # Vinyl segmentation 모델 및 손실 함수 정의
│
├── image_test_flow.py            # 전체 파이프라인 통합 실행 스크립트
└── README.md                     # 프로젝트 설명 파일
```

---

## ✅ 주요 기능

### 1️⃣ 재질 분류 (Plastic vs Paper)
- `EfficientNet-B0` 기반 이진 분류 모델 사용
- `classification_dataset` 폴더에서 이미지 로딩 후 학습
- `sigmoid` + `BCEWithLogitsLoss` 사용

### 2️⃣ 비닐 감지
- `segmentation_models_pytorch`의 UNet 구조 사용
- HybridLoss = BCE + Dice + Focal 조합으로 세그멘테이션 학습
- 비닐 픽셀 비율이 2% 이상이면 `비닐 존재`로 판단

### 3️⃣ 오염도 평가
- 오염 영역 마스크 예측 후 픽셀 비율로 등급 나눔
- contamination ratio 기준:
  - ✅ `clean` : < 3%
  - ⚠️ `slightly dirty` : 3% ~ 20%
  - ❌ `dirty` : > 20%

---

## 🚀 통합 실행 방법

```bash
python image_test_flow.py



