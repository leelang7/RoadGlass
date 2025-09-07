# lane_wear_infer.py
import os, json, time, argparse
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

# ========= 시각화 색상(BGR) 및 스타일 =========
COLOR_MAP = {
    0: (0, 0, 255),      # stop_line -> 빨강
    1: (255, 0, 0),      # traffic_lane_blue_dotted  -> 파랑
    2: (180, 0, 0),      # traffic_lane_blue_solid   -> 짙은 파랑
    3: (255, 255, 255),  # traffic_lane_white_dotted -> 흰색
    4: (200, 200, 200),  # traffic_lane_white_solid  -> 회백(흰색 구분)
    5: (0, 255, 255),    # traffic_lane_yellow_dotted-> 노랑
    6: (0, 200, 200),    # traffic_lane_yellow_solid -> 짙은 노랑
}
ALPHA_FILL = 0.35   # 반투명 오버레이 강도
CNT_THICK   = 2     # 컨투어 선 두께

# ========= 유틸 =========
def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def to_gray(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def sobel_magnitude(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)

def mask_to_skeleton(bin_mask):
    sk = skeletonize(bin_mask.astype(bool))
    return sk.astype(np.uint8)

def boundary_ring(mask, ring=2):
    er = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=1)
    bd = cv2.subtract(mask, er)
    dil = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=ring)
    outer = cv2.subtract(dil, mask)
    return bd, outer

def largest_component_ratio(mask):
    if mask.sum() == 0:
        return 0.0, 0
    lab = label(mask, connectivity=2)
    props = regionprops(lab)
    areas = np.array([p.area for p in props]) if props else np.array([])
    if len(areas) == 0:
        return 0.0, 0
    main = areas.max()
    return float(main / areas.sum()), len(areas)

# ========= 지표 계산 =========
def compute_metrics(frame_bgr, lane_mask, inst_scores=None):
    m = (lane_mask > 0).astype(np.uint8)
    if m.any():
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1)

    area = int(m.sum())
    sk = mask_to_skeleton(m) if area > 0 else m
    sk_len = int(sk.sum()) if area > 0 else 0
    thickness = float(area / sk_len) if sk_len > 0 else 0.0
    main_ratio, cc_count = largest_component_ratio(m) if area > 0 else (0.0, 0)

    gray = to_gray(frame_bgr)
    grad = sobel_magnitude(gray)
    bd, outer = boundary_ring(m, ring=2) if area > 0 else (np.zeros_like(m), np.zeros_like(m))
    bd_vals = grad[bd.astype(bool)]
    outer_vals = grad[outer.astype(bool)]
    edge_contrast = float(bd_vals.mean() - outer_vals.mean()) if (bd_vals.size and outer_vals.size) else 0.0
    vis_score = float(np.mean(inst_scores)) if inst_scores is not None and len(inst_scores) else 1.0

    metrics = dict(
        area_px=area,
        skeleton_len_px=sk_len,
        thickness_px=thickness,
        main_component_ratio=main_ratio,
        cc_count=cc_count,
        edge_contrast=edge_contrast,
        visibility=vis_score
    )

    # WearScore (0~100, 높을수록 훼손 심함)
    c_main = (1.0 - main_ratio)               # 낮을수록 좋음
    c_cc   = min(cc_count / 8.0, 1.0)         # CC 많으면 나쁨
    c_th   = 1.0 - np.tanh(thickness / 8.0)   # 얇을수록 나쁨
    c_ed   = 1.0 - np.tanh(max(edge_contrast, 0.0) / 20.0)  # 대비 낮으면 나쁨
    c_vis  = 1.0 - min(vis_score, 1.0)        # 가시성 낮으면 나쁨

    w = dict(main=0.28, cc=0.22, th=0.22, ed=0.18, vis=0.10)
    wear = 100.0 * (w['main']*c_main + w['cc']*c_cc + w['th']*c_th + w['ed']*c_ed + w['vis']*c_vis)
    metrics['wear_score'] = float(np.clip(wear, 0.0, 100.0))

    return metrics, m

# ========= 결과 파싱 (클래스별 마스크/스코어) =========
def collect_class_masks(res):
    """
    {cid: {'mask':(H,W) uint8, 'scores':[float,...]}}
    """
    if res.masks is None or res.masks.data is None or len(res.masks.data) == 0:
        return {}
    masks = res.masks.data.cpu().numpy()  # (n, H, W)
    classes = res.boxes.cls.cpu().numpy().astype(int) if res.boxes.cls is not None else np.zeros((masks.shape[0],), int)
    confs   = res.boxes.conf.cpu().numpy().tolist() if res.boxes.conf is not None else [1.0]*masks.shape[0]

    out = {}
    for i, cid in enumerate(classes):
        bin_m = (masks[i] > 0.5).astype(np.uint8)
        if cid not in out:
            out[cid] = {'mask': bin_m, 'scores': [confs[i]]}
        else:
            out[cid]['mask'] = np.maximum(out[cid]['mask'], bin_m)  # 동일 클래스 인스턴스 OR
            out[cid]['scores'].append(confs[i])
    return out

# ========= 시각화 (클래스별 + FPS/메트릭) =========
def draw_overlay_per_class(frame, class_masks, metrics, fps, model_names=None, per_class_metrics=None):
    out = frame.copy()

    # ==== 0) 기본 그리기(클래스 마스크/컨투어) ====
    for cid, data in class_masks.items():
        mask = data['mask']
        if mask is None or not mask.any(): 
            continue
        color = COLOR_MAP.get(cid, (0, 255, 0))
        color_img = np.zeros_like(out)
        color_img[mask > 0] = color
        out = cv2.addWeighted(out, 1.0, color_img, ALPHA_FILL, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, CNT_THICK)

    h, w = out.shape[:2]

    # ==== 1) 좌측 상단: FPS + 전체 지표 (고정폭 패널) ====
    LEFT_PAD_X = 10
    LEFT_PAD_Y = 10
    LEFT_W     = 360   # 메인 지표 패널 고정 폭
    LINE_H     = 22
    TOP_LINE_H = 26    # FPS 행

    # 패널 배경
    lines = 1 + 6  # FPS 1줄 + 메트릭 6줄
    panel_h = TOP_LINE_H + 6*LINE_H + 16
    cv2.rectangle(out, (LEFT_PAD_X-6, LEFT_PAD_Y-6),
                  (LEFT_PAD_X+LEFT_W, LEFT_PAD_Y+panel_h),
                  (20, 20, 20), -1)

    # FPS
    x, y = LEFT_PAD_X, LEFT_PAD_Y + TOP_LINE_H
    cv2.putText(out, f"FPS: {fps:.1f}", (x, y-4), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40,255,40), 2, cv2.LINE_AA)

    # 전체 지표
    y += 6
    for k in ["wear_score","thickness_px","main_component_ratio","cc_count","edge_contrast","visibility"]:
        v = metrics.get(k, 0)
        txt = f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}"
        cv2.putText(out, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (20,20,240) if k=="wear_score" else (240,240,240), 2, cv2.LINE_AA)
        y += LINE_H

    left_panel_right = LEFT_PAD_X + LEFT_W

    # ==== 2) 우측: 클래스 범례 (남은 폭으로 1~2열 자동 배치) ====
    # 컬럼 폭/여백
    COL_W = 400
    GAP_X = 16
    RIGHT_MARGIN = 10

    # 남은 가로폭
    remaining_w = w - left_panel_right - RIGHT_MARGIN
    # 두 열이 들어갈 수 있으면 2열, 아니면 1열
    cols = 2 if remaining_w >= COL_W*2 + GAP_X else 1

    # legend 시작 X를 좌측 패널 오른쪽에 붙이되, 항상 화면 안쪽에
    if cols == 2:
        legend_x0 = max(left_panel_right + GAP_X, w - (COL_W*2 + GAP_X + RIGHT_MARGIN))
    else:
        legend_x0 = max(left_panel_right + GAP_X, w - (COL_W + RIGHT_MARGIN))

    legend_y0 = 20  # 상단 정렬 (세로는 좌측 패널과 겹쳐도 가로로만 분리되면 시각적으로 문제 없음)

    def cid2name(cid):
        if model_names is None:
            return str(cid)
        if isinstance(model_names, dict):
            return model_names.get(cid, str(cid))
        if isinstance(model_names, (list, tuple)) and cid < len(model_names):
            return model_names[cid]
        return str(cid)

    # 클래스 정렬
    class_ids = sorted(class_masks.keys())

    # 각 항목(한 클래스)의 높이(두 줄 + 여백)
    ITEM_H = 44

    for idx, cid in enumerate(class_ids):
        name = cid2name(cid)
        color = COLOR_MAP.get(cid, (0, 255, 0))
        m = class_masks[cid]['mask']
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        count = int(num_labels - 1)
        scores = class_masks[cid]['scores']
        mean_score = float(np.mean(scores)) if scores else 0.0

        pcm = per_class_metrics.get(cid, {}) if per_class_metrics else {}
        wear_c = pcm.get("wear_score", 0.0)
        th_c   = pcm.get("thickness_px", 0.0)

        # 위치 계산 (cols 열에 분할)
        col = idx % cols
        row = idx // cols
        legend_x = legend_x0 + col*(COL_W + GAP_X)
        legend_y = legend_y0 + row*ITEM_H

        # 항목 배경 박스
        cv2.rectangle(out, (legend_x-6, legend_y-6),
                      (legend_x+COL_W-20, legend_y+34), (30,30,30), -1)
        # 색상칩
        cv2.rectangle(out, (legend_x, legend_y), (legend_x+18, legend_y+18), color, -1)

        # 1행: class / cc / conf
        cv2.putText(out, f"{cid}:{name}  cc:{count}  conf:{mean_score:.2f}",
                    (legend_x+24, legend_y+14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,255,255), 1, cv2.LINE_AA)
        # 2행: wear / thickness
        cv2.putText(out, f"wear:{wear_c:.1f}  th:{th_c:.1f}",
                    (legend_x+24, legend_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200,200,200), 1, cv2.LINE_AA)

    return out


# ========= 메인 추론 루프 =========
def infer_on_video(args):
    model = YOLO(args.model)

    # 모델이 가진 클래스 ID를 가져와 CSV 컬럼 고정(없으면 런타임에 동적)
    if isinstance(model.names, dict):
        known_class_ids = sorted(model.names.keys())
    elif isinstance(model.names, (list, tuple)):
        known_class_ids = list(range(len(model.names)))
    else:
        known_class_ids = []

    cap = cv2.VideoCapture(args.source)
    assert cap.isOpened(), f"Cannot open source: {args.source}"
    fps_cap = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        ensure_dir(Path(args.save_video))
        out_video = cv2.VideoWriter(args.save_video, fourcc, fps_cap, (W, H))

    rows = []
    frame_idx = 0
    t_prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # FPS 계산
        t_now = time.time()
        dt = t_now - t_prev
        fps_runtime = (1.0 / dt) if dt > 0 else fps_cap
        t_prev = t_now

        results = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)
        res = results[0]

        # 클래스별 마스크/점수
        class_masks = collect_class_masks(res)

        # per-class 지표 계산
        per_class_metrics = {}
        for cid, data in class_masks.items():
            mtr, _ = compute_metrics(frame, data['mask'], inst_scores=data['scores'])
            per_class_metrics[cid] = mtr

        # 전체 차선 union으로 전체 지표 계산
        lane_union = np.zeros((H, W), np.uint8)
        all_scores = []
        for data in class_masks.values():
            lane_union = np.maximum(lane_union, data['mask'])
            all_scores.extend(data['scores'])
        metrics_all, _ = compute_metrics(frame, lane_union, inst_scores=all_scores)

        # 시각화
        vis = draw_overlay_per_class(
            frame, class_masks, metrics_all, fps_runtime,
            model_names=getattr(model, 'names', None),
            per_class_metrics=per_class_metrics
        )

        if out_video is not None:
            out_video.write(vis)

        if args.show:
            cv2.imshow("lane wear", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

        # 저장용 레코드
        row = {
            "frame": frame_idx,
            "fps": round(fps_runtime, 2),
            **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
               for k, v in metrics_all.items()}
        }

        # per-class 지표를 JSON에도 구조적으로 보관
        row["per_class"] = {}
        for cid, mtr in per_class_metrics.items():
            row["per_class"][cid] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                     for k, v in mtr.items()}

        # CSV에는 클래스별 wear/thickness만 컬럼로 뽑아 저장 (고정폭)
        # known_class_ids가 없으면 현재 프레임에 등장한 클래스만 저장
        cids_for_csv = known_class_ids if known_class_ids else sorted(class_masks.keys())
        for cid in cids_for_csv:
            mtr = per_class_metrics.get(cid, {})
            row[f"wear_c{cid}"] = float(mtr.get("wear_score", 0.0))
            row[f"th_c{cid}"]   = float(mtr.get("thickness_px", 0.0))

        rows.append(row)
        frame_idx += 1

    cap.release()
    if out_video is not None:
        out_video.release()
    if args.show:
        cv2.destroyAllWindows()

    # 저장
    if rows:
        # CSV
        if args.save_csv:
            import csv
            ensure_dir(Path(args.save_csv))
            # 필드명: frame/fps + 전체 지표 + per-class wear/th
            base_fields = ["frame", "fps", "wear_score", "thickness_px",
                           "main_component_ratio", "cc_count", "edge_contrast", "visibility"]
            # 클래스 컬럼
            sample_row = rows[0]
            cls_fields = [k for k in sample_row.keys() if k.startswith("wear_c") or k.startswith("th_c")]
            fieldnames = base_fields + cls_fields
            with open(args.save_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in fieldnames})

        # JSON (per_class 포함한 전체 구조)
        if args.save_json:
            ensure_dir(Path(args.save_json))
            with open(args.save_json, "w") as f:
                json.dump(rows, f, indent=2)

    print(f"[DONE] frames={len(rows)} | csv={args.save_csv} | json={args.save_json} | video={args.save_video}")

# ========= CLI =========
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="YOLO segmentation .pt (e.g., yolo11n-seg.pt or your best.pt)")
    ap.add_argument("--source", required=True, help="video file path or camera index (e.g., 0)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou",  type=float, default=0.5)
    ap.add_argument("--save_csv", default="outputs/lane_wear_metrics.csv")
    ap.add_argument("--save_json", default="outputs/lane_wear_metrics.json")
    ap.add_argument("--save_video", default="outputs/lane_wear_overlay.mp4")
    ap.add_argument("--show", action="store_true")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # 카메라 인덱스 지원
    if args.source.isdigit():
        args.source = int(args.source)
    infer_on_video(args)

'''-------------------------------------------
wear_score : 0(좋음) ~ 100(매우 심각)
thickness_px : 평균 두께. 작아질수록 마모 추정↑
main_component_ratio : 가장 큰 연결 성분 비율. 낮을수록 파손/단절↑
cc_count : 연결 성분 수. 많을수록 끊김↑
edge_contrast : 경계 대비(내부 경계 – 외곽 배경). 낮을수록 지워진/흐릿한 차선 가능↑
'''