# 🚧 RoadGlass LaneWear

AI 기반 **차선 · 정지선 · 횡단보도 마모/훼손도 분석 플랫폼**  
모바일 크로스플랫폼(Flutter) → FastAPI 서버 → PostgreSQL 저장 → 대시보드(Web)까지  
**촬영 → 업로드 → 분석 → 시각화 → 유지보수 의사결정** 전 과정을 자동화합니다.

---

## ✨ 주요 기능

- 🛣️ **차선/정지선/횡단보도 마모 분석**: YOLO Segmentation + 후처리 → WearScore(0~100)
- 🙈 **프라이버시 보호**: 얼굴 · 번호판 자동 블러링
- 🖼️ **이미지 저장/조회**: 원본 / 오버레이 JPEG 저장 및 제공
- 📊 **지표/통계 API**
  - `/stats/summary` 최근 경향
  - `/geo/cells` 격자 Heatmap
  - `/candidates/rank` 유지보수 우선순위
- 📱 **크로스플랫폼 앱**
  - Flutter 기반: Android / iOS / Web 대응
  - 지도/차량 UI + API 연동
- 💻 **대시보드(Web)**
  - 최근 분석 이미지 미리보기
  - 지도 Heatmap
  - 유지보수 후보 리스트

---



## 🏗️ 전체 아키텍처

```mermaid
flowchart LR
  A[📱 모바일 앱(Flutter)] -->|이미지+GPS+시간| B[⚡ FastAPI 서버]
  B -->|YOLO Segmentation| C[🧮 WearScore 계산]
  C -->|INSERT| D[(🗄️ PostgreSQL DB)]
  C -->|저장| E[📂 ./RoadGlass/orig & overlay]
  D --> F[🌐 웹 대시보드(React/Flutter Web)]
  E --> F
```

------



## 📦 프로젝트 구조(권장)

```
RoadGlass/
 ├─ backend/                # FastAPI 서버
 │   ├─ blur_server.py      # ⬇️ 이 README 맨 아래 전체 코드 그대로 저장
 │   └─ requirements.txt    # ⬇️ 이 README 중간의 블록을 그대로 저장
 │
 ├─ mobile/                 # Flutter 앱 (선택)
 │   └─ ...
 │
 ├─ web/                    # Web 대시보드 (선택)
 │   └─ ...
 │
 └─ README.md               # 바로 이 파일
```



## 🔧 요구사항

- 🐍 Python **3.11+**
- 🐘 PostgreSQL **12+** (JSONB + 집계)
- ⚡ CUDA GPU 권장 (CPU로도 동작 가능하지만 느림)



## ⚙️ 환경변수 예시 (.env)

```
DB_URL="postgresql+psycopg://postgres:postgres@<host>:5432/postgres"
RG_STORE_DIR="./RoadGlass"                       # 이미지 저장 루트
PUBLIC_BASE_URL="http://<public-host>:8000"      # 공개 URL 베이스(옵션)

YOLO_LANE_MODEL="best_model.pt"                  # 세그멘테이션 pt (차선/횡단보도/정지선)
YOLO_MODEL="yolov11n-face.pt"                    # 얼굴
YOLO_LP_MODEL="license-plate-v1x.pt"             # 번호판
FACE_CONF=0.25
PLATE_CONF=0.25
BLUR_IOU=0.50
BLUR_STRENGTH=31
PIXEL_SIZE=16
BLUR_METHOD="gaussian"                           # or "pixelate"
```



## 🚀 서버 실행

```
# 1) 가상환경 권장
python -m venv .venv && source .venv/bin/activate

# 2) 요구 패키지 설치
pip install -r backend/requirements.txt

# 3) FastAPI 실행
uvicorn backend.blur_server:app --host 0.0.0.0 --port 8000
```

✅ 헬스체크:

```
curl http://localhost:8000/health
```

📖 OpenAPI 문서:

```
http://localhost:8000/docs
```

------



## 🔌 핵심 API

### ▶️ POST `/lane_wear_infer`

- 입력(Form-Data):
   `file`(이미지), `gps_lat`, `gps_lon`, `timestamp`(ISO8601), `device_id`
- 출력(JSON):
   `overall.wear_score`, `per_class`, `db_id`, `orig_url`, `overlay_url` 등
- 저장:
  - DB: 1행
  - 이미지: `./RoadGlass/orig/{id}.jpg`, `./RoadGlass/overlay/{id}.jpg`

예시:

```
curl -X POST http://localhost:8000/lane_wear_infer \
  -F "file=@./sample.jpg" \
  -F "gps_lat=37.5665" \
  -F "gps_lon=126.9780" \
  -F "timestamp=2025-09-24T06:12:00Z" \
  -F "device_id=rg-unit-001"
```

### ▶️ GET `/lane_wear/image/{id}/{kind}`

- `kind`: `orig` | `overlay`
- JPEG 바이너리 응답

### ▶️ GET `/lane_wear/{id}` / `/lane_wear/latest` / `/lane_wear/recent`

- 저장된 결과 조회(이미지 URL 포함)

### ▶️ GET `/stats/summary`

- 최근 window 내 알람 집계, 디바이스 상태, 트렌드

### ▶️ GET `/geo/cells`

- 지도 Heatmap용 격자 집계(평균/최대/상위 90퍼센타일)

### ▶️ GET `/candidates/rank` / `/candidates/rank_for_id/{id}`

- 유지보수 우선순위 스코어 및 랭킹

------



## 🗄️ DB 스키마

```
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
```

------

## 🧪 Python 클라이언트 (초간단)

```
# lane_client.py
import requests, sys, os
origin = os.environ.get("ORIGIN", "http://localhost:8000")
img = sys.argv[1]
with open(img, "rb") as f:
    r = requests.post(
        f"{origin}/lane_wear_infer",
        files={"file": (os.path.basename(img), f, "image/jpeg")},
        data={
            "gps_lat": 37.5665,
            "gps_lon": 126.9780,
            "timestamp": "2025-09-24T06:12:00Z",
            "device_id": "rg-unit-001"
        },
        timeout=120
    )
print(r.status_code, r.reason)
print(r.json())
```

------



## 🗺️ 라벨 예시

- ⛔ `stop_line` — 정지선
- 🚶 `crosswalk` — 횡단보도
- ⚪ `traffic_lane_white_{solid|dotted}`
- 🟡 `traffic_lane_yellow_{solid|dotted}`
- 🔵 `traffic_lane_blue_{solid|dotted}`

------

## 🧭 WearScore 구성요소

- 📐 픽셀 면적
- 📏 Skeleton 길이(두께 유추)
- 🔗 가장 큰 연결성분 비율 / 연결성분 수
- ✨ 경계부 Edge Contrast
- 👁️ 평균 confidence 기반 Visibility

------

## 📦 `backend/requirements.txt`

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
python-multipart==0.0.9
pillow==10.4.0
numpy==1.26.4
opencv-python-headless==4.10.0.84
ultralytics==8.3.1
scikit-image==0.23.2
SQLAlchemy==2.0.34
psycopg==3.2.1
psycopg-binary==3.2.1
python-dotenv==1.0.1
```

> ⚠️ **scikit-image / numpy 바이너리 호환** 에러가 나면:
>  `pip install --upgrade --force-reinstall --no-cache-dir numpy==1.26.4 scikit-image==0.23.2` 로 재설치 권장.



## 🚀 서버 실행

```
# 1) 가상환경 권장
python -m venv .venv && source .venv/bin/activate

# 2) 요구 패키지 설치
pip install -r backend/requirements.txt

# 3) FastAPI 실행
uvicorn backend.blur_server:app --host 0.0.0.0 --port 8000
```

✅ 헬스체크:

```
curl http://localhost:8000/health
```

📖 OpenAPI 문서:

```
http://localhost:8000/docs
```
