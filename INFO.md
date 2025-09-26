# 서버주소
ec2-54-67-48-0.us-west-1.compute.amazonaws.com

# DB
seoul-ht-04.cpk0oamsu0g6.us-west-1.rds.amazonaws.com

# DB 접속 명령어
psql -h seoul-ht-04.cpk0oamsu0g6.us-west-1.rds.amazonaws.com -U postgres -d postgres

# 백엔드 서버 구동 명령어
sudo /home/ubuntu/anaconda3/bin/python -m uvicorn blur_server:app --host 0.0.0.0 --port 80 --reload

# 웹(대시보드) 구동 명령어 - 해당 경로에서 => /home/ubuntu/yolo-flutter-app/example
flutter run -d web-server --web-hostname 0.0.0.0 --web-port 8080


# 지표(문서참조)
'''-------------------------------------------
wear_score : 0(좋음) ~ 100(매우 심각) 가이드: ≥70 심각, 40–69 주의, <40 양호
thickness_px : 평균 두께. 작아질수록 마모 추정↑
main_component_ratio : 가장 큰 연결 성분 비율. 낮을수록 파손/단절↑
cc_count : 연결 성분 수. 많을수록 끊김↑
edge_contrast : 경계 대비(내부 경계 – 외곽 배경). 낮을수록 지워진/흐릿한 차선 가능↑
'''

# 대시보드 지표
활성 디바이스(24h): 지난 24시간 내 데이터가 1건이라도 들어온 디바이스 수. (오프라인 탐지)
이상 알림(심각): 각 디바이스의 최신 기록 기준 wear≥70 인 디바이스 수.
카드의 트렌드는 “최근 24h 심각 건수”를 직전 24h와 비교한 증감률.
보수 후보: 각 디바이스의 최근 3건이 모두 wear≥70 → 단발성 스파이크가 아니라 지속 악화.
정상 노드: 각 디바이스 최신 wear<40.
임계치(40/70)는 서버 파라미터로 노출되어 있어 필요 시 조정 가능.