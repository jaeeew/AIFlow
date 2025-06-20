# AIFLOW-Backend

## 개요
AIFLOW-Backend는 이미지 분류 및 분석을 위한 Flask 기반 REST API 서비스입니다. 세 가지 주요 기능을 하나의 파이프라인으로 연결하여 제공합니다:

1. **재질 분류**: 종이(paper) vs 플라스틱(plastic)  
2. **비닐 검출**: 플라스틱 이미지에서 비닐(foil) 유무 판별  
3. **오염도 평가**: 비닐이 없는 플라스틱 이미지에서 오염도(`clean`, `slight`, `heavy`) 평가  

---

## 주요 기능
- **3단계 파이프라인**: 분류 → 비닐 검출 → 오염도 평가  
- **사전 학습된 모델** (`checkpoints/` 폴더)  
  - `best_classification_model.pth`  
  - `best_vinyl_model.pth`  
  - `best_contamination_model.pth`  
- **간단한 REST API**: `/upload` 엔드포인트  
- **오류 처리**: 예측 중 발생하는 오류를 JSON 형태로 반환  

---

## 디렉토리 구조

```
AIFLOW-Backend/
├── clf_model.py           # 재질 분류 (paper vs plastic)
├── vinyl_model.py         # 비닐 검출 (플라스틱 대상)
├── ctm_model.py           # 오염도 평가 (플라스틱, 비닐 없음)
├── server.py              # Flask API 서버
├── uploads/               # 업로드된 이미지 임시 저장
├── requirements.txt       # 의존성 목록
└── checkpoints/           # 사전 학습된 모델 파일
    ├── best_classification_model.pth
    ├── best_vinyl_model.pth
    └── best_contamination_model.pth
```

---

## API 사용법

### POST `/upload`  
이미지를 업로드하여 세 단계 파이프라인을 실행하고 결과를 반환합니다.

#### 요청
- **Content-Type**: `multipart/form-data`
- **Form 필드**: `file` (이미지 파일)

#### 응답

- **200 OK**
  
  ```
  { "result": "paper" }
  { "result": "plastic_with_vinyl" }
  { "result": "plastic_clean" }  // 또는 "plastic_slight" / "plastic_heavy"


- **500 Internal Server Error**

```
  { "error": "에러 메시지" }
```

---

## 모델 파일 설명

### `clf_model.py`
- 재질 분류 모델 로드 및 추론  
- **입력 이미지**: 224×224 리사이즈 및 정규화  
- **예측 임계값**: `0.5`  

### `vinyl_model.py`
- U-Net 기반 세그멘테이션 모델 로드  
- Albumentations 전처리 → 시그모이드 마스크 산출  
- **마스크 비율 > 2%** → 비닐 존재로 판정  

### `ctm_model.py`
- 오염도 세그멘테이션 모델 로드  
- 시그모이드 마스크로부터 오염 픽셀 비율 계산  
- **분류 기준**  
  - `<1%`: `clean`  
  - `1%–20%`: `slight`  
  - `>20%`: `heavy`  

