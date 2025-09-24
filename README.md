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
