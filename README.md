# RoadGlass LaneWear 🚧

AI 기반 **차선·정지선·횡단보도 마모/훼손도 분석 플랫폼**  
모바일 크로스플랫폼(Flutter) → FastAPI 서버 → PostgreSQL 저장 → 대시보드(Web)까지  
**촬영 → 업로드 → 분석 → 시각화 → 유지보수 의사결정** 전 과정을 자동화합니다.

---

## ✨ 주요 기능
- **차선/정지선/횡단보도 마모 분석**: YOLO Segmentation + 후처리 → WearScore(0~100)
- **프라이버시 보호**: 얼굴·번호판 자동 블러링
- **이미지 저장/조회**: 원본/오버레이 JPEG 저장 및 제공
- **지표/통계 API**:
  - `/stats/summary` 최근 경향
  - `/geo/cells` 격자 Heatmap
  - `/candidates/rank` 유지보수 우선순위
- **크로스플랫폼 앱**:
  - Flutter 기반: Android/iOS/Web 대응
  - 지도/차량 UI + API 연동
- **대시보드(Web)**:
  - 최근 분석 이미지 미리보기
  - 지도 Heatmap
  - 유지보수 후보 리스트

---

## 🏗️ 전체 아키텍처

```mermaid
flowchart LR
  A[모바일 앱(Flutter)] -->|이미지+GPS+시간| B[FastAPI 서버]
  B -->|YOLO Segmentation| C[WearScore 계산]
  C -->|INSERT| D[(PostgreSQL DB)]
  C -->|저장| E[./RoadGlass/orig & overlay]
  D --> F[웹 대시보드(React/Flutter Web)]
  E --> F

---

📦 서버 폴더 구조
backend/
 ├─ blur_server.py        # FastAPI 앱 (LaneWear API)
 ├─ requirements.txt
 └─ RoadGlass/            # 이미지 저장 루트 (기본 ./RoadGlass, env로 변경 가능)
     ├─ orig/             # 블러 반영 원본
     └─ overlay/          # 분석 오버레이

🔧 요구사항

Python 3.11+

PostgreSQL 12+ (JSONB + 집계)

CUDA GPU 권장 (CPU도 가능)

⚙️ 환경변수 예시 (.env)
DB_URL="postgresql+psycopg://postgres:postgres@<host>:5432/postgres"
RG_STORE_DIR="./RoadGlass"
PUBLIC_BASE_URL="https://api.example.com"

YOLO_LANE_MODEL="best_model.pt"
YOLO_MODEL="yolov11n-face.pt"            # 얼굴
YOLO_LP_MODEL="license-plate-v1x.pt"     # 번호판

🚀 실행
pip install -r requirements.txt
uvicorn blur_server:app --host 0.0.0.0 --port 8000


헬스체크:

curl http://localhost:8000/health


OpenAPI 문서:

http://localhost:8000/docs

🔌 API 요약
POST /lane_wear_infer

입력: file, gps_lat, gps_lon, timestamp, device_id

출력: overall.wear_score, per_class, db_id, 이미지 URL

저장: DB row + RoadGlass/orig/{id}.jpg + RoadGlass/overlay/{id}.jpg

예시:

curl -X POST http://localhost:8000/lane_wear_infer \
  -F "file=@./sample.jpg" \
  -F "gps_lat=37.5665" -F "gps_lon=126.9780" \
  -F "timestamp=2025-09-24T06:12:00Z" \
  -F "device_id=rg-unit-001"

🗄️ DB 스키마
CREATE TABLE lane_wear_results (
  id SERIAL PRIMARY KEY,
  created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
  image_name VARCHAR(255),
  model VARCHAR(255),
  width INT, height INT,
  runtime_ms DOUBLE PRECISION,
  overall JSONB,
  per_class JSONB,
  gps_lat DOUBLE PRECISION,
  gps_lon DOUBLE PRECISION,
  "timestamp" TIMESTAMPTZ,
  device_id VARCHAR(255)
);

📱 모바일 (Flutter)

공용 코드베이스: Android / iOS / Web 지원

기능:

카메라 촬영 + GPS → 서버 업로드

업로드 결과 → 오버레이 이미지 다운로드/표시

지도 위 마모도 Heatmap 시각화

사용 패키지:

http, geolocator, google_maps_flutter, flutter_map, dio

💻 프론트엔드 대시보드

Flutter Web 또는 React 기반

주요 화면:

최근 업로드 결과 리스트 + 썸네일

지도 Heatmap (API /geo/cells 활용)

유지보수 후보 우선순위 표 (/candidates/rank)

🧪 Python 클라이언트
python lane_client.py --origin http://localhost:8000 test.jpg

🗺️ 클래스 라벨 예시

stop_line — 정지선

crosswalk — 횡단보도

traffic_lane_white_{solid|dotted}

traffic_lane_yellow_{solid|dotted}

traffic_lane_blue_{solid|dotted}

🧭 WearScore 계산 요소

차선/정지선/횡단보도 픽셀 면적

Skeleton 길이 → 두께 추정

주요 컴포넌트 비율

경계부 Edge Contrast

Confidence 기반 Visibility

📜 라이선스

내부 프로젝트용 (상용 시 별도 라이선스 적용)
