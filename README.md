# RoadGlass LaneWear
AI 기반 **차선·정지선·횡단보도 마모/훼손도** 분석 서비스  
모바일(Flutter) → 백엔드(FastAPI+YOLO) → 대시보드(웹) End-to-End 파이프라인

![status-badge](https://img.shields.io/badge/status-active-blue) ![python](https://img.shields.io/badge/python-3.11%2B-blue) ![fastapi](https://img.shields.io/badge/FastAPI-0.11x-green) ![postgres](https://img.shields.io/badge/PostgreSQL-12%2B-lightgrey)

---

## ✨ 핵심 기능
- **추론 API**: `/lane_wear_infer` — 이미지 + GPS/시간/디바이스 → **WearScore** 계산, DB 저장, 오버레이 생성
- **이미지 제공**: `/lane_wear/image/{id}/{orig|overlay}` — 원본/오버레이 JPEG
- **조회/통계**: `/lane_wear/{id}`, `/lane_wear/recent`, `/stats/summary`, `/geo/cells`, `/candidates/rank`
- **프라이버시**: 업로드 직후 **얼굴·번호판 자동 블러링**
- **저장 구조**: 결과 메타데이터는 **PostgreSQL**, 이미지는 `./RoadGlass/{orig,overlay}` (운영은 S3 권장)

---

## 🏗️ 아키텍처(요약)

```mermaid
flowchart LR
  A[Flutter 앱\n(촬영+GPS)] -->|multipart form-data| B[/FastAPI/]
  B -->|YOLO Seg| C[마모 지표 계산]
  C -->|INSERT| D[(PostgreSQL)]
  C -->|Save| E[orig/overlay 이미지]
  D --> F[대시보드(웹)]
  E --> F
