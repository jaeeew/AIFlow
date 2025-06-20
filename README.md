# ♻️ 분리수거 도우미 앱

이 앱은 사용자가 배달 음식 용기의 오염 정도를 촬영하고, AI 모델이 이를 분석하여 **분리수거 가능 여부**를 알려주는 환경 교육용 어플리케이션입니다.  
각 소재별(종이, 비닐, 플라스틱 등) 올바른 배출 방법도 함께 안내합니다.

---

## 📱 주요 기능

- **카메라 촬영 및 AI 분석 결과 전송**
- **분리수거 분류 결과에 따라 다른 결과 화면(Clean / Slightly Dirty / Dirty) 출력**
- **종이/비닐/플라스틱 각각의 분리배출 가이드 제공**
- **닉네임 설정 및 유지 기능 (SharedPreferences 사용)**

---

## 📁 프로젝트 구조

```
lib/
├── main.dart                      # 앱 진입점
├── Mainpage.dart                  # 메인화면: 버튼 4개(촬영, 가이드 등)
├── camera_screen.dart             # 카메라 실행 및 서버 통신
├── loading.dart                   # 이미지 분석 결과 로딩 화면
├── CleanResult.dart               # 결과 - 깨끗한 상태
├── SlightDirtyResult.dart         # 결과 - 약간 오염
├── DirtyResult.dart               # 결과 - 심각한 오염
├── VinylResult.dart               # 결과 - 비닐 전용 결과
├── paper1.dart                    # 종이 배출 정보
├── paper_how.dart                 # 종이 배출 가이드
├── plastic_how.dart               # 플라스틱 배출 가이드
├── vinyl_how.dart                 # 비닐 배출 가이드
├── how.dart                       # 공통 가이드 진입점
├── Nickname.dart                  # 닉네임 설정 및 저장 (SharedPreferences)
pubspec.yaml                       # 의존성 정의
```

---

## 🚀 실행 방법

```bash
# 1. 의존성 설치
flutter pub get

# 2. 앱 실행 (디바이스 연결 또는 에뮬레이터 필요)
flutter run
```

> ⚠️ iOS 디바이스에서 실행하려면 Xcode 설정 및 권한 요청 필요  
> ⚠️ Android에서는 카메라 권한 및 서버 연동 URL 확인 필요

---

## 🔧 사용된 주요 패키지

- `camera` – 카메라 제어
- `http` – 이미지 서버 전송 및 응답 수신
- `shared_preferences` – 사용자 정보 저장
- `http_parser` – 서버 요청의 MIME 타입 처리

---

## 📡 서버 연동

`camera_screen.dart` 파일에서 촬영된 이미지는 서버로 전송되며,  
서버는 오염도 분석 결과(`clean`, `slight_dirty`, `dirty`)를 반환합니다.  
이 결과에 따라 각기 다른 결과 화면으로 이동합니다.

---

## 📸 UI (사용자 흐름)

앱의 전체적인 색상은 **연한 연두색(#E8F6E9)** 계열로 구성되어 있으며, 친환경 느낌과 직관적인 UI를 강조했습니다. 아이콘과 버튼은 **둥근 형태**로 구성되어 있어 부드럽고 명확한 사용자 경험을 제공합니다.

### 🔄 1. 로딩 화면 (`Loading`)
- 앱 실행 시 등장
- 앱의 마스코트인 냄비 일러스트와 함께 "쓰담쓰담" 로고 노출
- 하단 문구: “이름을 넣고 나만의 환경 지킴이 시작하기”

### 🙍 2. 닉네임 입력 화면 (`Nickname`)
- 사용자의 ID(닉네임)를 입력할 수 있는 간단한 로그인 화면
- 닉네임은 SharedPreferences를 통해 저장되어 다음에도 유지됨

### 🏠 3. 메인 화면 (`Mainpage`)
- 상단에 닉네임과 마스코트 캐릭터가 함께 표시됨
- 두 개의 주요 버튼:
  - `📘 HOW` – 분리배출 가이드 페이지로 이동
  - `📷 CAMERA` – 카메라 촬영 화면으로 이동

### 📷 4. 카메라 촬영 화면 (`CameraScreen`)
- 사용자가 직접 배달 용기를 촬영
- 상단에 "촬영할 용기의 정면을 맞춰주세요" 안내 문구
- 촬영 후 서버에 이미지가 전송되고 분석 결과에 따라 분류 화면으로 이동

### 🧼 5. 분석 결과 화면 (`Result Screens`)
분석 결과에 따라 다음 중 하나의 화면으로 이동:

- ✅ `clean1`: “깨끗한 재활용 가능”
- ❌ `dirty1`: “매우 더러움 – 재활용 불가”
- ⚠️ `dirty2`: “약간 더러움 – 재활용 기준을 참고해주세요”
- 🛍 `vinyl1`: “비닐 있음 – 비닐을 제거하고 분리수거 해주세요”
- 📄 `paper1`: “종이 오염되지 않은 종이만 분리수거 가능”

각 결과 화면에는 해당 메시지와 함께 홈으로 돌아가는 아이콘이 포함되어 있음

### ❓ 6. 분리배출 가이드 (`How → Plastic / Paper / Vinyl`)
- 공통 페이지(`how.dart`)에서는 분리배출 방법 안내와 세 가지 버튼 제공:
  - `PLASTIC`
  - `PAPER`
  - `VINYL`

#### ▶️ 각 소재별 가이드:
- `plastic_how.dart`: 플라스틱 병류의 배출 기준 안내
- `paper_how.dart`: 종이류 (신문, 전단, 포장지)의 배출 기준
- `vinyl_how.dart`: 투명/칼라 비닐, 장바구니류 배출 기준

각 페이지는 뒤로 가기 버튼과 함께 간결한 글머리 기호 목록으로 구성되어 있으며, 직관적인 아이콘이 함께 사용됨

---

## 🧩 코드별 주요 기능 요약

| 파일명 | 주요 기능 | 비고 / 특이사항 |
|--------|-----------|------------------|
| `main.dart` | 앱 진입점 (MyApp 위젯 실행) | `MaterialApp`, 라우팅 정의 포함 |
| `Mainpage.dart` | 메인 홈 화면 UI 구성 | 닉네임 표시, HOW 버튼, CAMERA 버튼 포함 |
| `Nickname.dart` | 닉네임 입력 화면 | `SharedPreferences` 사용하여 로컬 저장 |
| `camera_screen.dart` | 카메라 촬영 + 이미지 서버 전송 | `camera` 패키지, `http.MultipartRequest` 사용 |
| `loading.dart` | 결과 대기 (로딩) 화면 | 서버 응답 대기 중 로딩 애니메이션 가능 |
| `CleanResult.dart` | “깨끗함” 결과 페이지 | 재활용 가능 메시지 + 홈 이동 버튼 |
| `SlightDirtyResult.dart` | “약간 오염됨” 결과 페이지 | 기준 참고 메시지 출력 |
| `DirtyResult.dart` | “심각히 오염됨” 결과 페이지 | 재활용 불가 안내 표시 |
| `VinylResult.dart` | 비닐 검출 시 결과 페이지 | 비닐 제거 권장 메시지 출력 |
| `paper1.dart` | 종이 관련 결과 화면 | 오염되지 않은 종이만 가능 강조 |
| `how.dart` | 가이드 진입 선택 화면 | 3가지 버튼: PLASTIC, PAPER, VINYL |
| `plastic_how.dart` | 플라스틱 배출 가이드 | 플라스틱 병류 세척/배출 기준 안내 |
| `paper_how.dart` | 종이 배출 가이드 | 전단지/신문지/포장지 기준 설명 |
| `vinyl_how.dart` | 비닐 배출 가이드 | 일반/투명 비닐 분리배출 기준 안내 |
| `pubspec.yaml` | 의존성 관리 | `camera`, `http`, `shared_preferences` 등 포함 |

---

### 🔎 보충 설명

- **카메라 구현**
  - `camera_screen.dart`에서 `CameraController`로 실시간 영상 피드 제공
  - 버튼 클릭 시 이미지 캡처 → 서버 전송 → 결과값 받아 `Navigator.push`

- **결과 분기 처리**
  - 서버로부터 `"clean"`, `"slight_dirty"`, `"dirty"`, `"vinyl"` 등 응답값을 받아 각각 다른 결과 페이지로 이동

- **닉네임 저장**
  - `Nickname.dart`에서 `SharedPreferences`에 저장한 닉네임은 `Mainpage.dart`에서 불러와 사용자 표시

- **가이드 페이지**

---

## 👤 개발자 정보

- 소속: 성신여자대학교 AI융합학부
- 프로젝트: 캡스톤디자인 (AI 기반 분리배출 도우미 앱)
- 개발 플랫폼: Flutter + Dart
- 목적: 분리배출의 효율성과 시민 참여 향상

---

## ✅ TODO (향후 개선점)

- [ ] 서버 주소 외부 설정화
- [ ] 배출 안내 애니메이션 추가
- [ ] 점수제 도입 및 사용자 참여 유도
- [ ] 결과 화면 UI 개선 및 반응형 지원

