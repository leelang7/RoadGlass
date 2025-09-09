# FPS 모니터링 및 분석 가이드

이 프로젝트는 YOLO 모델의 실시간 추론 성능을 모니터링하고 분석하는 기능을 제공합니다.

## 🚀 주요 기능

### 1. 실시간 FPS 표시
- **우상단 FPS 패널**: 현재 FPS, Engine 상태, 최소/최대 FPS 표시
- **다중 FPS 소스**: Engine FPS, UI FPS, EMA FPS 지원
- **실시간 업데이트**: 1초마다 FPS 값 갱신

### 2. FPS 로깅
- **상세 로깅**: `run_log.jsonl` 파일에 모든 FPS 데이터 저장
- **다중 이벤트**: `on_result`, `periodic_imu` 이벤트별 FPS 기록
- **메타데이터**: 타임스탬프, IMU 데이터, 결과 개수 등 포함

### 3. FPS 분석 도구

#### `analyze_fps.py` - 상세 분석
```bash
python analyze_fps.py
```
- 추론 결과 FPS와 IMU 기반 FPS 분석
- 통계 정보 (평균, 최소, 최대, 표준편차)
- FPS 분포 분석 (30+, 60+ FPS 비율)
- 시각화 그래프 생성 (`fps_analysis.png`, `fps_histogram.png`)

#### `test_fps.py` - 실시간 모니터링
```bash
python test_fps.py [시간(초)]
```
- 실시간 FPS 모니터링 (기본 60초)
- 실시간 콘솔 출력
- 자동 분석 및 그래프 생성

## 📊 FPS 표시 정보

### 화면 표시
- **FPS**: 현재 추론 FPS (Engine 우선, UI fallback)
- **Engine**: 추론 엔진 상태 (ON/OFF)
- **EMA**: 지수이동평균 FPS (Engine 없을 때)
- **Min/Max**: 최근 10개 프레임의 최소/최대 FPS

### 로그 데이터
```json
{
  "t_us": 1757386491180868,
  "event": "on_result",
  "fps": 45.9,
  "fps_engine": 45.9,
  "fps_ui": 60.0,
  "fps_ema": 44.2,
  "fps_history": [45.1, 46.2, 45.8, 45.9, 45.7],
  "imu": { ... },
  "result_count": 3,
  "classes": ["crosswalk", "traffic_lane_white_solid"],
  "results": [ ... ]
}
```

## 🔧 설정 및 최적화

### FPS 계산 방식
1. **Engine FPS**: YOLO 엔진에서 제공하는 실제 추론 FPS
2. **UI FPS**: Flutter UI 프레임 기반 FPS (fallback)
3. **EMA FPS**: 지수이동평균으로 안정화된 FPS

### 성능 최적화 팁
- **30+ FPS**: 기본적인 실시간 처리 가능
- **60+ FPS**: 부드러운 실시간 처리
- **120+ FPS**: 고성능 실시간 처리

### 로그 파일 관리
- `run_log.jsonl`: 실시간 로그 (JSON Lines 형식)
- `run_log.json`: 전체 로그 (JSON 배열 형식)
- 로그 파일 크기가 클 경우 주기적으로 정리 권장

## 📈 분석 결과 예시

```
📊 FPS 분석 결과
==================================================

🎯 추론 결과 FPS (on_result):
   총 프레임: 1250
   평균 FPS: 45.2
   최소 FPS: 28.1
   최대 FPS: 62.3
   표준편차: 8.7
   30+ FPS: 1180/1250 (94.4%)
   60+ FPS: 450/1250 (36.0%)

📱 IMU 기반 FPS (periodic_imu):
   총 샘플: 5000
   평균 FPS: 44.8
   최소 FPS: 25.2
   최대 FPS: 65.1
   표준편차: 9.2
```

## 🛠️ 문제 해결

### FPS가 0으로 표시되는 경우
1. 모델 로딩 확인
2. 카메라 권한 확인
3. 디바이스 성능 확인

### FPS가 불안정한 경우
1. 백그라운드 앱 종료
2. 디바이스 발열 확인
3. 모델 크기 최적화 고려

### 로그 파일이 생성되지 않는 경우
1. 앱 권한 확인
2. 저장 공간 확인
3. 로깅 설정 확인 (`_logging = true`)

## 📝 추가 정보

- **지원 플랫폼**: Android, iOS
- **로그 형식**: JSON Lines (JSONL)
- **분석 도구**: Python 3.7+ (matplotlib, numpy 필요)
- **권장 모니터링 시간**: 60초 이상

이 도구를 사용하여 YOLO 모델의 성능을 최적화하고 실시간 추론 품질을 향상시키세요!

