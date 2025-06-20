# ♻️ 쓰담쓰담 (SSUDAM-SSUDAM)

<p align="center">
  <img width="133" alt="image" src="https://github.com/user-attachments/assets/04261baa-e47a-4638-911f-7aef41935e57" />
</p>

## 🔗 애플리케이션 구조 (이미지)
<p align="center">
   <img  alt="image" src="https://github.com/user-attachments/assets/ce5840b2-7394-47ba-ab4d-92574ff5e502"/>
</p>

## 📁 파일 구조
- main ( 프로젝트 설명 - readme)
- AI_model ( 분류 ai 모델, 오염도 측정 ai 모델) 
- Backend  ( ai 모델과 frontend와의 연결 코드파일)
- Frontend  ( 앱의 frontend 코드파일 - 카메라 및 디자인 등)

## 📌 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [팀원 소개](#팀원-소개)
3. [프레임워크 및 워크플로우](#프레임워크-및-워크플로우)
4. [오염도 기준](#오염도-기준)
5. [데이터 전처리](#데이터-전처리)
6. [모델 소개](#모델-소개)
   - [종이/플라스틱 분류](#종이플라스틱-분류)
   - [비닐 검출](#비닐-검출)
   - [오염도 평가](#오염도-평가)
7. [성과 및 결과](#성과-및-결과)
8. [향후 계획](#향후-계획)
9. [설치 및 실행 방법](#설치-및-실행-방법)
10. [데이터셋 구성](#데이터셋-구성)
11. [사용 예시 (Usage Example)](#사용-예시-usage-example)
12. [라이선스](#라이선스)



---

## 💡프로젝트 개요

**서비스 목적**  
- 배달용기 분리수거 기준이 헷갈리는 사용자에게  
  `사진 한 장으로 재질(플라스틱/종이/비닐)`과 `오염도(깨끗/약간/심함)` 정보를 제공하여 올바른 분리수거 방법을 제시합니다.  
- 친환경 생활을 유도하고, 향후 포인트 제도나 배달 플랫폼 제휴를 통해 서비스 확장을 염두에 둡니다.

**대상 고객**  
- 분리수거 기준이 헷갈리거나 익숙하지 않은 일반 사용자  
- 1인 가구, 배달·포장 문화가 발달한 도심 거주자

**프로젝트 내용**  
- AI 기반 ⟪분리수거 가이드 어플리케이션⟫ 개발  
  - **사진 기반 실시간 판별**  
  - 배달용기의 재질 및 오염도(이염도)를 이미지로 분석  
- Flutter/Dart를 활용한 모바일 UI 구현 (+Figma 디자인 시안)  
- PyTorch/Scikit-Learn 기반 학습 파이프라인 구축  
- Flask 서버(추후 FastAPI 전환 예정)로 모델 추론 API 제공  

---

## 👭 팀원 소개

| 팀원    | 역할         | 담당 분야         |
|---------|--------------|------------------|
| 최서정  | 팀 리더      | AI(비닐, 오염도), Back-end (Flask)  |
| 박혜원  | AI 엔지니어  | AI(비닐), Front-end (Flutter), 디자인(Figma) |
| 신서현  | AI 엔지니어  | AI(비닐), Front-end (Flutter), 디자인(Figma) |
| 엄서린  | AI 엔지니어  | AI(비닐,오염도), Back-end (Flask)  |
| 장수진  | AI 엔지니어  | AI(비닐), Front-end (Flutter), 디자인(Figma) |
| 황재윤  | AI 엔지니어  | AI(비닐,오염도), Back-end (Flask)  |

---

## 🛠️ 프레임워크 및 워크플로우

### 사용 기술 스택

- **🎨 디자인 (Design)**  
  - Figma: UI/UX 시안 작성  

- ** 📱 프론트엔드 (Front-end)**  
  - Flutter (Dart): 안드로이드 앱 개발  

- ** 💻 백엔드 (Back-end)**  
  - Flask (Python): 모델 추론 API 제공 (추후 FastAPI 전환 예정)  

- ** 🤖 AI 모델 (AI Models)**  
  - PyTorch: Classification 및 Segmentation 모델 학습  
  - Scikit-Learn: 추가 전처리/분류 모듈 활용  

### ✅ 워크플로우 (4단계)

1. **이미지 인식 (카메라 구동)**  
   - Flutter 앱에서 카메라 화면을 띄우고, 사용자가 배달용기 사진을 촬영하면  
     해당 이미지를 서버(Flask)로 전송  
2. **플라스틱/종이 분류**  
   - 서버 측에서 전처리 후 `EfficientNet-B0` 기반 분류기로 “플라스틱” 또는 “종이” 판별  
3. **비닐 O/X 검출**  
   - 분류된 플라스틱 이미지에 대해 비닐 포함 여부를 **Segmentation** 모델로 예측  
   - 비닐 O → 일반 쓰레기, 비닐 X → 플라스틱 재활용  
4. **오염도 평가**  
   - 분류된 재질(플라스틱/종이) 영역에서 **오염도(이염도)** Segmentation을 수행  
   - 결과값에 따라 `Clean / Slight / Heavy` 세 가지 단계로 구분  

---

## 🥡 오염도 기준

| 번호 | 지역     | 오염도 기준                                  |
|:----:|:--------:|:--------------------------------------------|
|  1   | 마포구   | 깨끗이 세척 후 이물질이 없는 정도            |
|  2   | 송파구   | 깨끗이 세척 후 이물질이 없는 정도            |
|  3   | 안양시   | 깨끗이 세척 후 **완벽하게 깨끗한** 정도      |
|  4   | 화성시   | 깨끗이 세척 후 폐기. **이염된 정도는 OK**    |

- 실제 서비스에서는 사용자가 위치(지역)를 선택하거나  
  GPS 기반으로 자동 인식하여 해당 지역의 `오염도 기준`을 적용할 예정입니다.

---

## ➡️ 데이터 전처리


데이터셋 (구글 드라이브) : https://drive.google.com/drive/folders/1YI7Riz41jzbPYZoDVGUnVra6dOcSZTpq?usp=drive_link


1. **직접 촬영 (Raw Data Collection)**  
   - **플라스틱/종이 분류 모델**  
     - 플라스틱: 244장  
     - 종이: 75장  
   - **비닐 검출 모델**: 430장  
   - **오염도 평가 모델**: 507장  

2. **라벨링 (Labeling)**  
   - LabelMe 사용  
   - **비닐 & 오염된 부분**: Object 클래스 = 1, 나머지 = 0  
   - 폴리곤(polygon) 형식으로 이미지 라벨링 후 JSON 내보내기  

3. **마스크 생성 (Mask Creation)**  
   - <u>비닐</u>  
     - 비닐 영역을 흰색(255), 나머지 배경을 검은색(0)으로 마스크  
   - <u>플라스틱/오염도</u>  
     - 오염도(segmentation) 영역을 흰색, 나머지를 검은색으로 마스크  
   - U-Net 기반 모델을 사용하여 픽셀 단위 Segmentation Mask 연산  

4. **데이터 증강 (Data Augmentation)**  
   - 좌우 반전(Flip), 상하 반전(Flip)  
   - 수평/수직 이동(Translation)  
   - 확대(Zoom in) / 축소(Zoom out)  
   - 밝기(Brightness) & 대비(Contrast) 무작위(Random) 조절  

---

## 📝 모델 소개

### 종이/플라스틱 분류

- **모델 아키텍처**: EfficientNet-B0 (pretrained on ImageNet)  
- **하이퍼파라미터**  
  - Batch size: 8  
  - Epoch: 30  
- **성능(Validation 셋 기준)**  
  - **Accuracy:** 1.0000 (100%)  
  - **Loss:** 0.0199  
- **선택 이유**  
  - 비교 실험 결과, EfficientNet-B0이 데이터셋 크기 대비  
    빠른 수렴(Convergence)과 높은 정확도를 보였음  
  - 모델 파라미터 수가 적당하여 Flutter 앱 내 서버 응답 속도 향상  

---

### 비닐 검출 (Segmentation)

- **모델 아키텍처**: EfficientNet-B0 기반 U-Net 형태  
- **하이퍼파라미터**  
  - Batch size: 4  
  - Epoch: 60  
- **성능 (Validation Loss 기준)**  
  - **Loss:** 0.2674  
  - **IoU (Intersection over Union):** 0.4199  
  - **Dice Coefficient:** 0.5699  
- **선택 이유**  
  - 충분한 학습량(430장) 대비 U-Net + EfficientNet이 전체적으로  
    균형 잡힌 학습 곡선 및 안정적인 Segmentation 결과를 보여줌  
  - IoU & Dice가 일정 수준 이상으로 수렴하여 실사용 가능 판별  

---

### 오염도 평가 (Contamination Segmentation)

- **모델 아키텍처**: EfficientNet-B0 기반 U-Net  
- **하이퍼파라미터**  
  - Batch size: 4  
  - Epoch: 60  
- **성능 (Validation Loss 기준)**  
  - **Loss:** 0.1679  
  - **IoU (Intersection over Union):** 0.7288  
  - **Dice Coefficient:** 0.8314  
- **선택 이유**  
  - 플라스틱/종이 품목별 오염 면적 계산에 적합한 **High IoU** 구현  
  - Dice Coefficient가 매우 높아, 실제 `깨끗 / 약간 / 심함` 분류에  
    사용할 수 있는 정량적 근거 확립  

---

## 📖 성과 및 결과

### 1) 종이/플라스틱 분류

| 구분       | Input 이미지 (실제)              | Ground Truth | Predicted   |
|-----------|---------------------------------|--------------|-------------|
| 사례 1    | <img width="102" alt="image" src="https://github.com/user-attachments/assets/840a29e5-4c46-4331-970e-e7326cd8e7a0" /> <img width="99" alt="image" src="https://github.com/user-attachments/assets/e8ee2c04-2752-4072-bb2b-294efc1000b4" />| paper       | paper       |
| 사례 2    | <img width="101" alt="image" src="https://github.com/user-attachments/assets/d612aef4-6e62-4bd1-bbbc-af1e0e13eb4c" /> <img width="99" alt="image" src="https://github.com/user-attachments/assets/d17f432f-91b2-452e-bf2f-bee9a9d87a03" />| plastic     | plastic     |

- **결과 요약**:  
  - Validation Accuracy: 100% (경우에 따라 데이터셋 불균형 이슈 보완 필요)  
  - 오차 없이 플라스틱/종이를 정확히 구분함  

### 2) 비닐 검출 (Segmentation)

| 구분             | Input 이미지                | Ground Truth Mask             | Predicted Mask                |
|-----------------|----------------------------|-------------------------------|-------------------------------|
| 비닐 포함 케이스 | <img width="126" alt="image" src="https://github.com/user-attachments/assets/ed050a3d-bc85-4ba5-96d0-41dfbcde8e2f" />| <img width="126" alt="image" src="https://github.com/user-attachments/assets/3567916b-ddad-4e36-a3cc-8c7f9a296786" />| <img width="126" alt="image" src="https://github.com/user-attachments/assets/460fe7e2-4901-4883-bb7f-b66090d0d434" />|
| 비닐 없음 케이스 | <img width="126" alt="image" src="https://github.com/user-attachments/assets/f11243a8-e2ec-420b-9acb-e26945f6b2c6" />| <img width="128" alt="image" src="https://github.com/user-attachments/assets/ebbd4550-8dff-4811-a3fa-ae0e868a4e2a" />| <img width="128" alt="image" src="https://github.com/user-attachments/assets/88352b2c-e88f-4e6e-a349-694a8c2e8bf4" />|

- **평가 지표**  
  - Validation IoU: 0.4199  
  - Validation Dice: 0.5699  
- **결과 요약**  
  - 비닐이 있는 경우, Mask 영역이 적절히 검출됨.  
  - 검출 실패(비닐 없음) 시, Predicted Mask가 모두 0으로 나와 정상 동작 확인.

### 3) 오염도 평가

| 구분             | Input 이미지                | Ground Truth Mask             | Predicted Mask                |
|-----------------|----------------------------|-------------------------------|-------------------------------|
| 오염(진한) 사례 | <img width="170" alt="image" src="https://github.com/user-attachments/assets/8e8f2367-7d90-42fb-a0dc-3923e42029fd" />| <img width="166" alt="image" src="https://github.com/user-attachments/assets/5c5a99da-43e8-4a1d-aace-b84e70dc12b2" />| <img width="165" alt="image" src="https://github.com/user-attachments/assets/f4112f11-3b35-4e55-a1b6-b2524293ba16" />|
| 오염(깨끗) 사례 |<img width="159" alt="image" src="https://github.com/user-attachments/assets/2217de06-b8f8-49d3-92aa-96ff918b09d1" />|<img width="162" alt="image" src="https://github.com/user-attachments/assets/e6d99621-6f5e-405d-87f9-6b95fafdad74" />| <img width="162" alt="image" src="https://github.com/user-attachments/assets/f0cda9e0-a9c3-4985-b530-0fae73ba7669" />|

- **평가 지표**  
  - Validation IoU: 0.7288  
  - Validation Dice: 0.8314  
- **결과 요약**  
  - 오염도(이염도) Segmentation 성능이 높아,  
    `깨끗 / 약간 / 심함` 구분 기준을 충분히 만족함.  

---

## 🔮 향후 계획

1. **모델 성능 개선**  
   - **데이터 다양성 확대**:  
     - 전국 주요 지자체(서울시·경기도·인천·대전 등) 쓰레기 분리수거 기준 수집  
     - 다양한 각도, 조명, 배경을 포함한 배달용기 이미지 확보  
   - **모델 업그레이드**:  
     - 최신 Segmentation(예: DeepLabV3+, Mask R-CNN) 등 아키텍처 실험  
     - 하이퍼파라미터 튜닝(학습률 스케줄러, 앙상블 등)

2. **서비스 확장**  
   - **기능 영역 확장**:  
     - 배달용기 외에 “커피컵/테이크아웃 컵”  
     - 택배 박스, 의류 포장(플라스틱 필름) 판별  
   - **포인트 제도 연동**:  
     - 분리수거 실천 시 사용자에게 인앱 포인트 지급  
     - 제휴된 배달 플랫폼(예: 배달의민족, 쿠팡이츠)과 협업

3. **정책/사회적 확장**  
   - **지자체 & 환경부 제휴**:  
     - 스마트 분리수거 기기(음성 안내/자동 분류) 개발  
     - 재활용 가능 품목 자동 분류 허브 구현  
   - **교육 및 캠페인 연계**:  
     - 초·중·고교, 지역 커뮤니티와 협업하여  
       분리수거 교육 콘텐츠 제작  

---

## 설치 및 실행 방법

아래 절차는 로컬 환경에서 **모델 서버(Flask)** 와 **Flutter 앱**을 동시에 실행하는 예시입니다.

### 1. 필수 요구사항

- Python 3.8 이상  
- PyTorch 1.12 이상  
- Flutter SDK ≥ 3.0  
- Android Studio / Xcode (에뮬레이터 또는 실제 기기 실행용)  

---

### 2. 저장소 클론

```bash
$ git clone https://github.com/<GitHub-계정>/SSUDAM-SSUDAM.git
$ cd SSUDAM-SSUDAM
