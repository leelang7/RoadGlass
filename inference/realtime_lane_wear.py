import cv2
import numpy as np
import time
import tempfile
import requests
from datetime import datetime
from ultralytics import YOLO

# ==== 사용자 설정 ====
MODEL_PATH = r"C:\lsc\ultralytics\runs\segment\train3\weights"   # YOLOv11-seg 혹은 v8-seg 가중치
LANE_CLS = 14                   # 0-base에서 차선 클래스 인덱스
CONF_THRES = 0.35               # 감지 임계치
WEAR_THRESH = 0.55              # 마모 점수 경보 임계치 (0~1)
POST_URL = "http://<YOUR_SERVER>/lane_report"  # 서버 엔드포인트

# 얇음/갭 판단 파라미터(픽셀). 카메라/해상도에 맞춰 튜닝 필요
THIN_WIDTH_PX = 6               # 이 값보다 얇으면 마모로 간주
GAP_MIN_PX = 12                 # 이 크기 이하의 구멍/틈은 무시
K_CLosing = 5                   # mask closing 커널 크기(홀 제거 안정화)

def get_gps():
    """실기기에서는 실제 GPS (예: 안드로이드 FusedLocation) 를 연결해 주세요."""
    return {"lat": None, "lon": None, "accuracy_m": None}

def lane_wear_score(gray, lane_mask):
    """
    lane_mask: 0/255 이진 마스크
    마모 지표:
      1) thin_ratio: 거리변환 기반 두께 지도에서 THIN_WIDTH_PX 미만 비율
      2) gap_ratio : closing 후 다시 열어서 생기는 작은 갭(결손) 비율
      3) (옵션) brightness_drop: 차선 내부 평균 밝기 저하 (역정규화)
    """
    H, W = lane_mask.shape
    if lane_mask.sum() == 0:
        return 0.0, {"thin_ratio": 0.0, "gap_ratio": 0.0, "brightness_drop": 0.0}

    # 안정화: 작은 구멍 메우기
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (K_CLosing, K_CLosing))
    stable = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, k)

    # 1) 두께 지도(거리변환 x2 ≈ 현지 두께)
    dt = cv2.distanceTransform((stable > 0).astype(np.uint8), cv2.DIST_L2, 3)
    width_map = dt * 2.0
    thin_ratio = float((width_map < THIN_WIDTH_PX).sum()) / float((stable > 0).sum())

    # 2) 갭/결손 비율(열림으로 잔결 제거 후 비교)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(stable, cv2.MORPH_OPEN, k2)
    holes = cv2.bitwise_and(stable, cv2.bitwise_not(opened))
    # 너무 작은 점들은 무시
    n_labels, lab = cv2.connectedComponents((holes > 0).astype(np.uint8))
    gap_pixels = 0
    for l in range(1, n_labels):
        area = int((lab == l).sum())
        if area >= GAP_MIN_PX:  # 임계치 이상만 카운트
            gap_pixels += area
    gap_ratio = float(gap_pixels) / float((stable > 0).sum())

    # 3) 밝기 저하(선택): 차선은 보통 주변보다 밝음. 평균 밝기가 낮아지면 마모 추정
    lane_mean = float(cv2.mean(gray, mask=stable)[0])
    # 프레임 전체 평균 대비 상대치
    frame_mean = float(gray.mean())
    brightness_drop = max(0.0, (frame_mean - lane_mean) / max(frame_mean, 1e-5))  # 0~1 근사

    # 가중합 (필요시 조정)
    score = 0.6 * thin_ratio + 0.3 * gap_ratio + 0.1 * brightness_drop
    return float(score), {
        "thin_ratio": float(thin_ratio),
        "gap_ratio": float(gap_ratio),
        "brightness_drop": float(brightness_drop),
    }

def masks_from_result(result, lane_cls=LANE_CLS, conf_thres=CONF_THRES):
    """
    Ultralytics YOLO 결과에서 lane 마스크 합성(여러 인스턴스 OR).
    세그가 없으면 bbox로 대체 마스크 생성.
    """
    H, W = result.orig_shape
    lane_mask = np.zeros((H, W), dtype=np.uint8)

    # 없으면 바로 리턴
    if result.boxes is None:
        return lane_mask

    # 후보 필터
    classes = (result.boxes.cls.detach().cpu().numpy()
               if result.boxes.cls is not None else [])
    confs = (result.boxes.conf.detach().cpu().numpy()
             if result.boxes.conf is not None else [])
    boxes = (result.boxes.xyxy.detach().cpu().numpy()
             if result.boxes.xyxy is not None else [])

    # 세그 있으면 result.masks.data[n] (H',W') → result.masks.xy / .data 사용
    seg_ok = (result.masks is not None and result.masks.data is not None)

    for i in range(len(boxes)):
        if int(classes[i]) != lane_cls or confs[i] < conf_thres:
            continue

        if seg_ok:
            m = result.masks.data[i].cpu().numpy()  # float mask (h',w'), 0~1
            # 원본 크기로 리스케일
            mh, mw = m.shape
            m_up = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            lane_mask[m_up > 0.5] = 255
        else:
            # fallback: bbox -> 직사각형 마스크
            x1, y1, x2, y2 = boxes[i].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)
            lane_mask[y1:y2+1, x1:x2+1] = 255

    return lane_mask

def send_to_server(image_bgr, meta):
    # 임시 파일로 저장 후 multipart 업로드
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, image_bgr)
        files = {"image": open(tmp.name, "rb")}
        try:
            r = requests.post(POST_URL, data=meta, files=files, timeout=5)
            print("[POST]", r.status_code, r.text[:200])
        except Exception as e:
            print("[POST][ERR]", e)

def main():
    cap = cv2.VideoCapture(0)  # 웹캠; 모바일/임베디드에선 카메라 인덱스/파이프 수정
    if not cap.isOpened():
        print("❌ Camera open failed")
        return

    model = YOLO(MODEL_PATH)
    print("✅ Model loaded:", MODEL_PATH)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.time()
        res = model.predict(frame, conf=CONF_THRES, imgsz=640, verbose=False)[0]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lane_mask = masks_from_result(res, LANE_CLS, CONF_THRES)
        score, parts = lane_wear_score(gray, lane_mask)

        # 시각화
        overlay = frame.copy()
        colored = np.zeros_like(frame)
        colored[:, :, 1] = lane_mask  # green channel
        vis = cv2.addWeighted(overlay, 1.0, colored, 0.5, 0)

        # 박스도 함께
        if res.boxes is not None:
            for b, c, cf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
                if int(c) == LANE_CLS and cf >= CONF_THRES:
                    x1, y1, x2, y2 = map(int, b)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

        fps = 1.0 / max(1e-6, (time.time() - t0))
        cv2.putText(vis, f"score={score:.2f}  (thin:{parts['thin_ratio']:.2f} gap:{parts['gap_ratio']:.2f})",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0) if score < WEAR_THRESH else (0, 0, 255), 2)
        cv2.putText(vis, f"{fps:.1f} FPS", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("lane_wear", vis)

        # 임계 초과 → 캡쳐 & 전송
        if score >= WEAR_THRESH:
            gps = get_gps()
            meta = {
                "timestamp_utc": datetime.utcnow().isoformat(),
                "wear_score": f"{score:.3f}",
                "thin_ratio": f"{parts['thin_ratio']:.3f}",
                "gap_ratio": f"{parts['gap_ratio']:.3f}",
                "lat": gps["lat"], "lon": gps["lon"], "accuracy_m": gps["accuracy_m"],
            }
            print("⚠️  wear severe -> send", meta)
            send_to_server(frame, meta)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
