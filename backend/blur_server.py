# roadglass_server.py
import io
import os
import math
import pathlib
from datetime import datetime
from typing import Literal, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, Float, select, desc, text
from sqlalchemy.dialects.postgresql import JSONB
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from ultralytics import YOLO

# =========================================================
# 앱 & 저장소 설정
# =========================================================
app = FastAPI(title="RoadGlass LaneWear API")

STORE_ROOT = os.getenv("RG_STORE_DIR", "./RoadGlass")
ORIG_DIR = os.path.join(STORE_ROOT, "orig")
OVERLAY_DIR = os.path.join(STORE_ROOT, "overlay")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# =========================================================
# DB 연결 및 스키마
# =========================================================
DEFAULT_DB_URL = "postgresql+psycopg://postgres:postgres@seoul-ht-04.cpk0oamsu0g6.us-west-1.rds.amazonaws.com:5432/postgres"
DATABASE_URL = os.getenv("DB_URL", DEFAULT_DB_URL)

engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)
metadata = MetaData()

lane_wear_results = Table(
    "lane_wear_results",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP")),
    Column("image_name", String(255)),
    Column("model", String(255)),
    Column("width", Integer), Column("height", Integer),
    Column("runtime_ms", Float),
    Column("overall", JSONB),
    Column("per_class", JSONB),
    Column("gps_lat", Float), Column("gps_lon", Float),
    Column("timestamp", DateTime),
    Column("device_id", String),
)

@app.on_event("startup")
def on_startup():
    metadata.create_all(engine)
    os.makedirs(ORIG_DIR, exist_ok=True)
    os.makedirs(OVERLAY_DIR, exist_ok=True)

# =========================================================
# 모델 로딩
# =========================================================
MODEL_PATH = os.getenv("YOLO_MODEL", "yolov11n-face.pt")       # 얼굴
LANE_MODEL_PATH = os.getenv("YOLO_LANE_MODEL", "best_model.pt") # 차선/정지선/횡단보도 세그
LP_MODEL_PATH = os.getenv("YOLO_LP_MODEL", "license-plate-finetune-v1x.pt")

FACE_CONF = float(os.getenv("FACE_CONF", "0.25"))
PLATE_CONF = float(os.getenv("PLATE_CONF", "0.25"))
BLUR_IOU  = float(os.getenv("BLUR_IOU",  "0.50"))
BLUR_STRENGTH = int(os.getenv("BLUR_STRENGTH", "31"))
PIXEL_SIZE    = int(os.getenv("PIXEL_SIZE", "16"))
BLUR_METHOD   = os.getenv("BLUR_METHOD", "gaussian")  # "gaussian" | "pixelate"

_model_face: Optional[YOLO] = None
_model_lane: Optional[YOLO] = None
_model_lp: Optional[YOLO] = None

def _check_model_file(path: str, label: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} model not found: {path}")

def get_face_model() -> YOLO:
    global _model_face
    if _model_face is None:
        _check_model_file(MODEL_PATH, "Face")
        _model_face = YOLO(MODEL_PATH)
    return _model_face

def get_lane_model() -> YOLO:
    global _model_lane
    if _model_lane is None:
        _check_model_file(LANE_MODEL_PATH, "Lane")
        _model_lane = YOLO(LANE_MODEL_PATH)
    return _model_lane

def get_lp_model() -> YOLO:
    global _model_lp
    if _model_lp is None:
        _check_model_file(LP_MODEL_PATH, "License plate")
        _model_lp = YOLO(LP_MODEL_PATH)
    return _model_lp

# =========================================================
# 유틸 (IO/변환)
# =========================================================
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 20 * 1024 * 1024))
ALLOWED_MIME = {"image/jpeg", "image/png"}

def read_image_from_upload(file: UploadFile) -> Image.Image:
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=415, detail=f"Unsupported Content-Type: {file.content_type}")
    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")
    img = Image.open(io.BytesIO(raw))
    try:
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        img = img.convert("RGB")
    return img

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    arr = np.array(img, dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv2_to_jpeg_bytes(img_bgr: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()

def resize_long_edge(pil_img: Image.Image, max_edge: int) -> Image.Image:
    w, h = pil_img.size; m = max(w, h)
    if m <= max_edge: return pil_img
    s = max_edge / m
    return pil_img.resize((int(w * s), int(h * s)), Image.LANCZOS)

def _save_jpg(path: str, img_bgr: np.ndarray, quality: int = 92):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")

def _build_url(req: Optional[Request], path: str) -> str:
    if PUBLIC_BASE_URL: return f"{PUBLIC_BASE_URL}{path}"
    if req is not None:
        base = str(req.base_url).rstrip("/")
        return f"{base}{path}"
    return path

# =========================================================
# 기본 지표 계산(공통)
# =========================================================
def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)

def mask_to_skeleton(bin_mask: np.ndarray) -> np.ndarray:
    if bin_mask.dtype != np.uint8: bin_mask = bin_mask.astype(np.uint8)
    sk = skeletonize(bin_mask.astype(bool))
    return sk.astype(np.uint8)

def boundary_ring(mask: np.ndarray, ring: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    er = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=1)
    bd = cv2.subtract(mask, er)
    dil = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=ring)
    outer = cv2.subtract(dil, mask)
    return bd, outer

def largest_component_ratio(mask: np.ndarray):
    if mask.sum() == 0: return 0.0, 0
    lab = label(mask, connectivity=2)
    props = regionprops(lab)
    areas = np.array([p.area for p in props]) if props else np.array([])
    if len(areas) == 0: return 0.0, 0
    main = areas.max()
    return float(main / areas.sum()), int(len(areas))

def compute_metrics(frame_bgr: np.ndarray, lane_mask: np.ndarray, inst_scores=None):
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
    bd_vals = grad[bd.astype(bool)]; outer_vals = grad[outer.astype(bool)]
    edge_contrast = float(bd_vals.mean() - outer_vals.mean()) if (bd_vals.size and outer_vals.size) else 0.0
    vis_score = float(np.mean(inst_scores)) if inst_scores is not None and len(inst_scores) else 1.0

    # wear (공통)
    c_main = (1.0 - main_ratio)
    c_cc   = min(cc_count / 8.0, 1.0)
    c_th   = 1.0 - np.tanh(thickness / 8.0)
    c_ed   = 1.0 - np.tanh(max(edge_contrast, 0.0) / 20.0)
    c_vis  = 1.0 - min(vis_score, 1.0)
    w = dict(main=0.28, cc=0.22, th=0.22, ed=0.18, vis=0.10)
    wear = 100.0 * (w['main']*c_main + w['cc']*c_cc + w['th']*c_th + w['ed']*c_ed + w['vis']*c_vis)

    return dict(
        area_px=area,
        skeleton_len_px=sk_len,
        thickness_px=float(thickness),
        main_component_ratio=float(main_ratio),
        cc_count=int(cc_count),
        edge_contrast=float(edge_contrast),
        visibility=float(vis_score),
        wear_score=float(np.clip(wear, 0.0, 100.0)),
    ), m

# =========================================================
# 패턴 인지/특화 지표 (solid/dotted/crosswalk)
# =========================================================
def _pca_main_angle(bin_mask: np.ndarray) -> Optional[float]:
    ys, xs = np.nonzero(bin_mask)
    if len(xs) < 20: return None
    pts = np.column_stack((xs, ys)).astype(np.float32)
    pts -= pts.mean(axis=0, keepdims=True)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    main_vec = eigvecs[:, np.argmax(eigvals)]
    ang = math.atan2(main_vec[1], main_vec[0])  # [-pi, pi]
    return ang

def _project_vals(bin_mask: np.ndarray, angle: float) -> np.ndarray:
    """마스크를 angle 방향으로 투영한 1D 프로파일 (합)"""
    ys, xs = np.nonzero(bin_mask)
    if len(xs) == 0: return np.array([])
    c, s = math.cos(angle), math.sin(angle)
    t = xs * c + ys * s
    t = t - t.min()
    t = (t / (t.max() + 1e-6) * 1000.0).astype(int)  # 리샘플 1D 버킷
    prof = np.bincount(t, minlength=(t.max()+1))
    return prof.astype(np.float32)

def _periodicity_score(prof: np.ndarray) -> float:
    """간단 ACF 기반 주기성: 2~N/2 래그 중 최대/0래그"""
    if prof.size < 16: return 0.0
    prof = (prof - prof.mean()) / (prof.std() + 1e-6)
    n = len(prof)
    # FFT로 ACF 근사
    f = np.fft.rfft(prof)
    acf = np.fft.irfft(f * np.conj(f), n=n).real
    acf = acf / (acf[0] + 1e-6)
    peak = float(np.max(acf[2:n//2])) if n//2 > 2 else 0.0
    return float(np.clip(peak, 0.0, 1.0))

def _skeleton_largest_continuity(sk: np.ndarray) -> float:
    """스켈레톤에서 최대 연결 길이 / 전체 길이"""
    if sk.sum() == 0: return 0.0
    num, comp = cv2.connectedComponents(sk.astype(np.uint8), connectivity=8)
    if num <= 1: return 0.0
    counts = [(comp == i).sum() for i in range(1, num)]
    return float(max(counts) / (sk.sum() + 1e-6))

def _bbox_coverage(bin_mask: np.ndarray) -> float:
    ys, xs = np.nonzero(bin_mask)
    if len(xs) == 0: return 0.0
    x1, x2 = xs.min(), xs.max(); y1, y2 = ys.min(), ys.max()
    bbox_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    return float(bin_mask.sum() / (bbox_area + 1e-6))

def _centroids_and_angles(bin_mask: np.ndarray):
    lab = label(bin_mask, connectivity=2)
    props = regionprops(lab)
    cents = []
    angs = []
    for p in props:
        if p.area < 20: continue
        cents.append(p.centroid)  # (y,x)
        angs.append(p.orientation)  # rad, note: skimage orientation is different convention
    return cents, angs  # angs ~ [-pi/2, pi/2]

def _angle_parallelism(angles: np.ndarray) -> float:
    """평행도 점수 (1=매우 평행, 0=불규칙). 주: 각도 wrap 보정."""
    if angles.size < 2: return 0.0
    # 각도 평균을 기준으로 편차 계산 (wrap 최소화)
    mu = np.mean(angles)
    diffs = np.arctan2(np.sin(angles - mu), np.cos(angles - mu))
    std = np.std(diffs)
    score = 1.0 - float(np.clip(std / (math.pi/6), 0.0, 1.0))  # std ~ 30deg 스케일
    return score

def _spacing_cv_along_normal(centroids: np.ndarray, main_angle: float) -> float:
    """주방향에 수직(normal) 방향으로 정렬한 간격 CV."""
    if centroids.shape[0] < 3: return 1.0
    # normal angle
    ang = main_angle + math.pi/2.0
    c, s = math.cos(ang), math.sin(ang)
    proj = centroids[:,1]*c + centroids[:,0]*s  # (x,y)=(col,row)->(col*c + row*s)
    proj = np.sort(proj)
    gaps = np.diff(proj)
    if np.mean(gaps) <= 1e-6: return 1.0
    return float(np.std(gaps) / (np.mean(gaps) + 1e-6))

def _pattern_from_name(cls_name: str) -> Optional[str]:
    n = cls_name.lower()
    if "cross" in n or "walk" in n or "zebra" in n: return "crosswalk"
    if "dotted" in n or "dashed" in n or "broken" in n: return "dotted"
    if "solid" in n or "stop_line" in n or "stop" in n: return "solid"
    return None

def _heuristic_pattern(bin_mask: np.ndarray, metrics_common: Dict[str, Any], main_angle: Optional[float]) -> str:
    """클래스명이 불명확할 때 휴리스틱으로 패턴 추정"""
    cc = metrics_common.get("cc_count", 0)
    thickness = metrics_common.get("thickness_px", 0.0)
    main_ratio = metrics_common.get("main_component_ratio", 0.0)
    if cc >= 6 and main_ratio < 0.6:
        # dotted 가능성
        return "dotted"
    # crosswalk: 다수 평행 스트라이프
    cents, angs = _centroids_and_angles(bin_mask)
    if len(cents) >= 3:
        angs = np.array(angs, dtype=np.float32)
        par = _angle_parallelism(angs)
        if par > 0.6:
            return "crosswalk"
    return "solid"

def compute_pattern_metrics(bin_mask: np.ndarray, frame_bgr: np.ndarray, common: Dict[str,Any], cls_name: str) -> Dict[str, Any]:
    H, W = bin_mask.shape
    out: Dict[str, Any] = {}
    if bin_mask.sum() == 0:
        out["pattern"] = _pattern_from_name(cls_name) or "unknown"
        out["wear_score_pattern"] = common.get("wear_score", 0.0)
        out["wear_score_final"] = out["wear_score_pattern"]
        return out

    angle = _pca_main_angle(bin_mask)
    name_pat = _pattern_from_name(cls_name)
    pattern = name_pat or _heuristic_pattern(bin_mask, common, angle)
    out["pattern"] = pattern

    # 공통 보조량
    sk = mask_to_skeleton(bin_mask)
    length_cont = _skeleton_largest_continuity(sk)  # [0,1]
    out["length_continuity"] = float(length_cont)

    # 패턴별
    if pattern == "solid":
        # width_cv 근사: 거리변환 기반 두께 표준편차/평균
        dist = cv2.distanceTransform((bin_mask>0).astype(np.uint8), cv2.DIST_L2, 3)
        vals = dist[dist>0] * 2.0  # 지름≈두께
        width_cv = float(np.std(vals)/ (np.mean(vals)+1e-6)) if vals.size>10 else 0.0
        out["width_cv"] = float(np.clip(width_cv, 0.0, 3.0))

        # 패턴 점수
        # main_component_ratio 높을수록 좋음, thickness_px 높을수록 좋음, edge_contrast 높을수록 좋음,
        # length_continuity 높을수록 좋음, width_cv 낮을수록 좋음
        mcr = common.get("main_component_ratio", 0.0)
        th  = common.get("thickness_px", 0.0)
        ed  = common.get("edge_contrast", 0.0)
        c = (
            0.35*(1.0 - mcr) +
            0.25*(1.0 - np.tanh(th/8.0)) +
            0.25*(1.0 - np.tanh(max(ed,0.0)/20.0)) +
            0.15*(np.clip(width_cv/0.8, 0.0, 1.0)) +   # 폭 불균일惡
            0.10*(1.0 - length_cont)                   # 불연속惡
        )
        wear_pat = float(np.clip(100.0*c, 0.0, 100.0))
        out["wear_score_pattern"] = wear_pat

    elif pattern == "dotted":
        if angle is None: angle = 0.0
        prof = _project_vals(bin_mask, angle)
        periodicity = _periodicity_score(prof)
        out["periodicity"] = periodicity

        # 연결성분 길이 추정(주방향 투영 범위)
        lab = label(bin_mask, connectivity=2); props = regionprops(lab)
        lengths = []
        min_t, max_t = np.inf, -np.inf
        c, s = math.cos(angle), math.sin(angle)
        for p in props:
            if p.area < 20: continue
            coords = p.coords  # (row,col)
            ts = coords[:,1]*c + coords[:,0]*s
            lengths.append(float(ts.max()-ts.min()+1.0))
            min_t = min(min_t, ts.min()); max_t = max(max_t, ts.max())
        lengths = np.array(lengths, dtype=np.float32) if len(lengths) else np.array([], np.float32)
        length_cv = float(np.std(lengths)/(np.mean(lengths)+1e-6)) if lengths.size>1 else 1.0
        out["length_cv"] = float(np.clip(length_cv, 0.0, 5.0))

        duty_cycle = float(np.clip((lengths.sum() / (max(max_t-min_t,1.0))), 0.0, 1.0)) if len(lengths) else 0.0
        out["duty_cycle"] = duty_cycle

        ed = common.get("edge_contrast", 0.0)
        th = common.get("thickness_px", 0.0)
        # 기대 duty cycle이 0.4~0.6 정도라 가정(없으면 중간값 기준)
        duty_err = abs(duty_cycle - 0.5)
        c = (
            0.30 * np.clip(duty_err/0.5, 0.0, 1.0) +
            0.25 * np.clip(length_cv/1.0, 0.0, 1.0) +
            0.20 * (1.0 - np.clip(periodicity, 0.0, 1.0)) +
            0.15 * (1.0 - np.tanh(max(ed,0.0)/20.0)) +
            0.10 * (1.0 - np.tanh(th/8.0))
        )
        wear_pat = float(np.clip(100.0*c, 0.0, 100.0))
        out["wear_score_pattern"] = wear_pat

    elif pattern == "crosswalk":
        # 스트라이프(연결성분) 별 방향/간격/커버리지
        cents, angs = _centroids_and_angles(bin_mask)
        stripe_count = len(cents)
        out["stripe_count"] = int(stripe_count)
        if stripe_count >= 2:
            angs = np.array(angs, dtype=np.float32)
            parallelism = _angle_parallelism(angs)
            out["parallelism"] = float(parallelism)
        else:
            parallelism = 0.0
            out["parallelism"] = 0.0

        if angle is None:
            # 평균 각도로 대체
            if stripe_count >= 2:
                angle = float(np.mean(angs))
            else:
                angle = 0.0

        cents_np = np.array([(y,x) for (y,x) in cents], dtype=np.float32) if stripe_count>0 else np.zeros((0,2), np.float32)
        spacing_cv = _spacing_cv_along_normal(cents_np, angle) if stripe_count>=3 else 1.0
        out["spacing_cv"] = float(np.clip(spacing_cv, 0.0, 5.0))

        coverage_ratio = _bbox_coverage(bin_mask)  # bbox 내 도색 비율
        out["coverage_ratio"] = float(coverage_ratio)

        ed = common.get("edge_contrast", 0.0)
        # 기대 스트라이프 개수/간격이 없으면 통계 지표 기반
        c = (
            0.30 * np.clip(spacing_cv/0.6, 0.0, 1.0) +         # 간격 불규칙惡
            0.25 * (1.0 - np.clip(parallelism, 0.0, 1.0)) +   # 평행도 낮음惡
            0.15 * (1.0 - np.clip(coverage_ratio/0.6, 0.0, 1.0)) +  # 채움 낮음惡
            0.20 * (1.0 - np.tanh(max(ed,0.0)/20.0)) +        # 경계 흐림惡
            0.10 * (1.0 - _skeleton_largest_continuity(mask_to_skeleton(bin_mask)))
        )
        wear_pat = float(np.clip(100.0*c, 0.0, 100.0))
        out["wear_score_pattern"] = wear_pat

    else:
        # unknown → 공통 점수 사용
        out["wear_score_pattern"] = float(common.get("wear_score", 0.0))

    out["wear_score_final"] = float(max(common.get("wear_score", 0.0), out["wear_score_pattern"]))
    return out

# =========================================================
# 세그멘테이션 결과 수집
# =========================================================
def collect_class_masks(res, target_hw):
    H, W = target_hw
    out = {}
    if res is None or getattr(res, "masks", None) is None:
        return out
    # tensor masks
    if getattr(res.masks, "data", None) is not None and len(res.masks.data) > 0:
        masks = res.masks.data.detach().cpu().numpy()
        clss  = res.boxes.cls.detach().cpu().numpy().astype(int) if res.boxes.cls is not None else np.zeros((masks.shape[0],), int)
        confs = res.boxes.conf.detach().cpu().numpy().tolist() if res.boxes.conf is not None else [1.0]*masks.shape[0]
        for i, cid in enumerate(clss):
            m = (masks[i] > 0.5).astype(np.uint8)
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            if cid not in out:
                out[cid] = {'mask': m, 'scores': [confs[i]]}
            else:
                out[cid]['mask'] = np.maximum(out[cid]['mask'], m)
                out[cid]['scores'].append(confs[i])
        return out
    # polygon masks
    if getattr(res.masks, "xy", None) is not None and len(res.masks.xy) > 0:
        clss  = res.boxes.cls.detach().cpu().numpy().astype(int) if res.boxes.cls is not None else np.zeros((len(res.masks.xy),), int)
        confs = res.boxes.conf.detach().cpu().numpy().tolist() if res.boxes.conf is not None else [1.0]*len(res.masks.xy)
        for i, cid in enumerate(clss):
            poly = res.masks.xy[i]
            canvas = np.zeros((H, W), np.uint8)
            if poly is not None and len(poly) >= 3:
                pts = np.array([[int(round(x)), int(round(y))] for x, y in poly], dtype=np.int32)
                pts[:, 0] = np.clip(pts[:, 0], 0, W-1)
                pts[:, 1] = np.clip(pts[:, 1], 0, H-1)
                cv2.fillPoly(canvas, [pts], 1)
            if cid not in out:
                out[cid] = {'mask': canvas, 'scores': [confs[i]]}
            else:
                out[cid]['mask'] = np.maximum(out[cid]['mask'], canvas)
                out[cid]['scores'].append(confs[i])
        return out
    return out

# =========================================================
# 블러 유틸
# =========================================================
def _detect_boxes(model: YOLO, frame_bgr: np.ndarray, conf: float, iou: float, imgsz: int) -> np.ndarray:
    r = model.predict(frame_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    return r.boxes.xyxy.detach().cpu().numpy() if r and r.boxes is not None and len(r.boxes) > 0 else np.empty((0,4), float)

def apply_blur(img_bgr: np.ndarray, boxes_xyxy: np.ndarray,
               method: str = "gaussian", blur_strength: int = 31, pixel_size: int = 16):
    out = img_bgr.copy()
    h, w = out.shape[:2]
    for x1, y1, x2, y2 in boxes_xyxy:
        x1 = max(int(x1), 0); y1 = max(int(y1), 0)
        x2 = min(int(x2), w); y2 = min(int(y2), h)
        if x2 <= x1 or y2 <= y1: continue
        roi = out[y1:y2, x1:x2]
        if method == "pixelate":
            sh, sw = max(1, (y2-y1)//pixel_size), max(1, (x2-x1)//pixel_size)
            small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_LINEAR)
            roi_blur = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
        else:
            k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
            roi_blur = cv2.GaussianBlur(roi, (k, k), 0)
        out[y1:y2, x1:x2] = roi_blur
    return out

# =========================================================
# 헬스체크
# =========================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "face_model": os.path.basename(MODEL_PATH),
        "lane_model": os.path.basename(LANE_MODEL_PATH),
        "lp_model": os.path.basename(LP_MODEL_PATH),
        "lane_classes": getattr(get_lane_model(), "names", {}),
    }

# =========================================================
# 오버레이 렌더
# =========================================================
COLOR_MAP = {0:(0,0,255),1:(255,0,0),2:(180,0,0),3:(255,255,255),4:(200,200,200),5:(0,255,255),6:(0,200,200)}
ALPHA_FILL = 0.35; CNT_THICK = 2

def make_overlay_image(frame_bgr: np.ndarray, class_masks: Dict[int, Dict[str, Any]]) -> np.ndarray:
    out = frame_bgr.copy()
    for cid, data in class_masks.items():
        mask = data.get('mask')
        if mask is None or not np.any(mask): continue
        color = COLOR_MAP.get(int(cid), (0, 255, 0))
        color_img = np.zeros_like(out)
        color_img[mask.astype(bool)] = color
        out = cv2.addWeighted(out, 1.0, color_img, ALPHA_FILL, 0)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, CNT_THICK)
    return out

# =========================================================
# 테스트용 블러 API (얼굴/번호판)
# =========================================================
@app.post("/blur")
def blur(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.01, le=1.0),
    iou: float = Query(0.50, ge=0.05, le=0.95),
    method: Literal["gaussian", "pixelate"] = Query("gaussian"),
    blur_strength: int = Query(31, ge=3, le=199),
    pixel_size: int = Query(16, ge=2, le=128),
    max_size: int = Query(1280, ge=320, le=4096),
    jpeg_quality: int = Query(90, ge=60, le=100),
):
    pil_img = read_image_from_upload(file)
    pil_img_rs = resize_long_edge(pil_img, max_size)
    img_bgr = pil_to_cv2(pil_img_rs)

    boxes_face = _detect_boxes(get_face_model(), img_bgr, conf, iou, max_size)
    try:
        boxes_plate = _detect_boxes(get_lp_model(), img_bgr, conf, iou, max_size)
    except Exception:
        boxes_plate = np.empty((0,4), float)

    boxes = boxes_face if len(boxes_face) else np.empty((0,4), float)
    if len(boxes_plate): boxes = np.concatenate([boxes, boxes_plate], axis=0) if len(boxes) else boxes_plate

    if len(boxes):
        img_bgr = apply_blur(img_bgr, boxes, method=method, blur_strength=blur_strength, pixel_size=pixel_size)
    return Response(content=cv2_to_jpeg_bytes(img_bgr, quality=jpeg_quality), media_type="image/jpeg")

# =========================================================
# Lane wear 추론 + 저장 (패턴 지표 포함)
# =========================================================
@app.post("/lane_wear_infer")
def lane_wear_infer(
    request: Request,
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.01, le=1.0),
    iou: float  = Query(0.50, ge=0.05, le=0.95),
    max_size: int = Query(1280, ge=320, le=4096),
    gps_lat: float = Form(...),
    gps_lon: float = Form(...),
    timestamp: datetime = Form(...),
    device_id: str = Form(...),
):
    import time
    pil_img = read_image_from_upload(file)
    pil_img_rs = resize_long_edge(pil_img, max_size)
    frame = pil_to_cv2(pil_img_rs)
    H, W = frame.shape[:2]

    # (1) 얼굴/번호판 블러
    try: boxes_face = _detect_boxes(get_face_model(), frame, FACE_CONF, BLUR_IOU, max_size)
    except Exception: boxes_face = np.empty((0,4), float)
    try: boxes_plate = _detect_boxes(get_lp_model(), frame, PLATE_CONF, BLUR_IOU, max_size)
    except Exception: boxes_plate = np.empty((0,4), float)
    if len(boxes_face) or len(boxes_plate):
        all_boxes = boxes_face if len(boxes_face) else np.empty((0,4), float)
        if len(boxes_plate): all_boxes = np.concatenate([all_boxes, boxes_plate], axis=0) if len(all_boxes) else boxes_plate
        frame = apply_blur(frame, all_boxes, method=BLUR_METHOD, blur_strength=BLUR_STRENGTH, pixel_size=PIXEL_SIZE)

    # (2) 차선/표지 세그 + 지표
    model = get_lane_model()
    t0 = time.time()
    res = model.predict(frame, imgsz=max_size, conf=conf, iou=iou, verbose=False)[0]
    elapsed_ms = (time.time() - t0) * 1000.0
    names = getattr(model, "names", {})

    class_masks = collect_class_masks(res, (H, W))

    lane_union = np.zeros((H, W), np.uint8); all_scores = []
    for data in class_masks.values():
        lane_union = np.maximum(lane_union, data['mask'])
        all_scores.extend(data['scores'])
    metrics_all, _ = compute_metrics(frame, lane_union, inst_scores=all_scores)

    per_class = {}
    for cid, data in class_masks.items():
        mtr, _ = compute_metrics(frame, data['mask'], inst_scores=data['scores'])
        cls_name = str(names.get(int(cid), cid)) if isinstance(names, dict) else str(cid)
        # 패턴 특화
        m_pat = compute_pattern_metrics(data['mask'], frame, mtr, cls_name)
        merged = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k,v in mtr.items()}
        merged.update({k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k,v in m_pat.items()})
        merged["class_name"] = cls_name
        per_class[int(cid)] = merged

    # (3) DB 저장
    db_id = None; db_error = None
    try:
        with engine.begin() as conn:
            ins = conn.execute(
                lane_wear_results.insert().values(
                    image_name = getattr(file, "filename", None),
                    model      = os.path.basename(LANE_MODEL_PATH),
                    width      = W, height = H,
                    runtime_ms = round(elapsed_ms, 2),
                    overall    = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in metrics_all.items()},
                    per_class  = per_class,
                    gps_lat    = gps_lat, gps_lon = gps_lon,
                    timestamp  = timestamp, device_id = device_id,
                )
            )
            db_id = ins.inserted_primary_key[0]
    except Exception as e:
        db_error = str(e)

    # (4) 이미지 저장 (블러 반영 원본 + 오버레이)
    orig_url = overlay_url = None
    try:
        if db_id is not None:
            orig_path = os.path.join(ORIG_DIR, f"{db_id}.jpg")
            overlay_path = os.path.join(OVERLAY_DIR, f"{db_id}.jpg")
            _save_jpg(orig_path, frame, quality=92)
            overlay_img = make_overlay_image(frame, class_masks)
            _save_jpg(overlay_path, overlay_img, quality=92)
            orig_url    = _build_url(request, f"/lane_wear/image/{db_id}/orig")
            overlay_url = _build_url(request, f"/lane_wear/image/{db_id}/overlay")
    except Exception as e:
        db_error = (db_error + " | " if db_error else "") + f"save_image: {e}"

    return {
        "model": os.path.basename(LANE_MODEL_PATH),
        "image_size": {"width": W, "height": H},
        "runtime_ms": round(elapsed_ms, 2),
        "overall": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in metrics_all.items()},
        "per_class": per_class,
        "db_id": db_id,
        "db_error": db_error,
        "orig_url": orig_url,
        "overlay_url": overlay_url,
    }

# =========================================================
# 이미지 서빙
# =========================================================
@app.api_route("/lane_wear/image/{id:int}/{kind}", methods=["GET", "HEAD"])
def get_lane_wear_image_kind(id: int, kind: Literal["orig", "overlay"]):
    path = os.path.join(ORIG_DIR if kind == "orig" else OVERLAY_DIR, f"{id}.jpg")
    if not os.path.exists(path):
        raise HTTPException(404, f"image not found: {kind} {id}")
    headers = {"Cache-Control": "public, max-age=86400"}
    return FileResponse(path, media_type="image/jpeg", headers=headers)

@app.api_route("/lane_wear/image/{id:int}", methods=["GET", "HEAD"])
def get_lane_wear_image_query(id: int, type: Literal["orig", "overlay"] = Query("orig")):
    return get_lane_wear_image_kind(id, type)

# =========================================================
# 조회/리스트
# =========================================================
def _shape_row(row: dict, req: Optional[Request] = None) -> dict:
    rid = row["id"]
    d = {
        "id": rid,
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
        "image_name": row.get("image_name"),
        "model": row.get("model"),
        "image_size": {"width": row.get("width"), "height": row.get("height")},
        "runtime_ms": row.get("runtime_ms"),
        "overall": row.get("overall") or {},
        "per_class": row.get("per_class") or {},
        "gps_lat": row.get("gps_lat"), "gps_lon": row.get("gps_lon"),
        "timestamp": row.get("timestamp").isoformat() if row.get("timestamp") else None,
        "device_id": row.get("device_id"),
    }
    if os.path.exists(os.path.join(ORIG_DIR, f"{rid}.jpg")):
        d["orig_url"] = _build_url(req, f"/lane_wear/image/{rid}/orig")
    if os.path.exists(os.path.join(OVERLAY_DIR, f"{rid}.jpg")):
        d["overlay_url"] = _build_url(req, f"/lane_wear/image/{rid}/overlay")
    return d

@app.get("/lane_wear/latest")
def get_lane_wear_latest(request: Request, image_name: Optional[str] = None):
    stmt = select(lane_wear_results)
    if image_name:
        stmt = stmt.where(lane_wear_results.c.image_name == image_name)
    stmt = stmt.order_by(desc(lane_wear_results.c.created_at)).limit(1)
    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().first()
        if not row: raise HTTPException(404, "no data")
        return _shape_row(row, request)

@app.get("/lane_wear/recent")
def get_lane_wear_recent(request: Request, limit: int = 20, offset: int = 0):
    limit = max(1, min(limit, 200))
    with engine.connect() as conn:
        rows = conn.execute(
            select(lane_wear_results)
            .order_by(desc(lane_wear_results.c.created_at))
            .limit(limit).offset(offset)
        ).mappings().all()
        return [_shape_row(r, request) for r in rows]

@app.get("/lane_wear/{id:int}")
def get_lane_wear(request: Request, id: int):
    with engine.connect() as conn:
        row = conn.execute(
            select(lane_wear_results).where(lane_wear_results.c.id == id)
        ).mappings().first()
        if not row: raise HTTPException(404, "not found")
        return _shape_row(row, request)

# =========================================================
# 요약 통계
# =========================================================
@app.get("/stats/summary")
def stats_summary(
    window_h: int = 24,
    warning: float = 40.0,
    critical: float = 70.0,
) -> Dict[str, Any]:
    with engine.begin() as conn:
        # 창 통계 (NULL 방지: COALESCE)
        q_window = text(f"""
            SELECT
              COUNT(*) AS detections,
              COUNT(DISTINCT CASE WHEN device_id IS NOT NULL THEN device_id END) AS active_devices,
              COALESCE(SUM(CASE WHEN (overall->>'wear_score')::float >= :critical THEN 1 ELSE 0 END), 0) AS alerts_critical,
              COALESCE(SUM(CASE WHEN (overall->>'wear_score')::float >= :warning
                                 AND (overall->>'wear_score')::float < :critical
                           THEN 1 ELSE 0 END), 0) AS alerts_warning
            FROM lane_wear_results
            WHERE created_at >= NOW() - INTERVAL '{int(window_h)} hour'
        """)
        row = conn.execute(q_window, {"warning": warning, "critical": critical}).mappings().one()

        # 디바이스 최신 상태 요약 (NULL 방지)
        q_latest = text("""
            WITH latest AS (
              SELECT DISTINCT ON (device_id)
                device_id, created_at, (overall->>'wear_score')::float AS wear
              FROM lane_wear_results
              WHERE device_id IS NOT NULL
              ORDER BY device_id, created_at DESC
            )
            SELECT
              COALESCE(SUM(CASE WHEN wear < :warning THEN 1 ELSE 0 END), 0)                       AS ok,
              COALESCE(SUM(CASE WHEN wear >= :warning AND wear < :critical THEN 1 ELSE 0 END), 0) AS warning_cnt,
              COALESCE(SUM(CASE WHEN wear >= :critical THEN 1 ELSE 0 END), 0)                     AS critical_cnt
            FROM latest;
        """)
        dev = conn.execute(q_latest, {"warning": warning, "critical": critical}).mappings().one()

        # 트렌드 (COUNT(*)는 0을 반환하므로 그대로 OK)
        q_trend = text(f"""
            WITH cur AS (
              SELECT COUNT(*) AS c FROM lane_wear_results
              WHERE created_at >= NOW() - INTERVAL '{int(window_h)} hour'
                AND (overall->>'wear_score')::float >= :critical
            ),
            prev AS (
              SELECT COUNT(*) AS p FROM lane_wear_results
              WHERE created_at >= NOW() - INTERVAL '{int(window_h*2)} hour'
                AND created_at <  NOW() - INTERVAL '{int(window_h)} hour'
                AND (overall->>'wear_score')::float >= :critical
            )
            SELECT c, p FROM cur, prev;
        """)
        t = conn.execute(q_trend, {"critical": critical}).mappings().one()
        c = float(t["c"] or 0)
        p = float(t["p"] or 0)
        delta = (c - p) / p if p > 0 else (1.0 if c > 0 else None)

        # 유지보수 후보 count (COUNT(*)는 0)
        q_maint = text("""
            WITH ranked AS (
              SELECT device_id, created_at, (overall->>'wear_score')::float AS wear,
                     ROW_NUMBER() OVER (PARTITION BY device_id ORDER BY created_at DESC) AS rn
              FROM lane_wear_results
              WHERE device_id IS NOT NULL
            ),
            last3 AS (
              SELECT device_id,
                     SUM( (wear >= :critical)::int ) AS crit3,
                     COUNT(*) AS n
              FROM ranked
              WHERE rn <= 3
              GROUP BY device_id
            )
            SELECT COUNT(*) AS candidates
            FROM last3
            WHERE n = 3 AND crit3 = 3;
        """)
        maint = conn.execute(q_maint, {"critical": critical}).mappings().one()

    return {
        "window_h": window_h,
        "thresholds": {"warning": warning, "critical": critical},
        "detections_24h": int(row["detections"] or 0),
        "active_devices_24h": int(row["active_devices"] or 0),
        "alerts_24h": {
            "critical": int(row["alerts_critical"] or 0),
            "warning": int(row["alerts_warning"] or 0),
            "trend_vs_prev": delta
        },
        "latest_device_state": {
            "ok": int(dev["ok"] or 0),
            "warning": int(dev["warning_cnt"] or 0),
            "critical": int(dev["critical_cnt"] or 0),
        },
        "maintenance_candidates": int(maint["candidates"] or 0),
    }

# =========================================================
# 공간 집계 & 후보 랭크
# =========================================================
@app.get("/geo/cells")
def geo_cells(
    min_lat: float = Query(...), min_lon: float = Query(...),
    max_lat: float = Query(...), max_lon: float = Query(...),
    step_m: int = Query(50, ge=10, le=500),
    window_h: int = Query(24, ge=1, le=168),
    agg: str = Query("p90", pattern="^(avg|max|p90)$"),
    min_count: int = Query(1, ge=1, le=100),
) -> Dict[str, Any]:
    if min_lat > max_lat or min_lon > max_lon:
        raise HTTPException(status_code=400, detail="invalid bbox")
    lat0 = (min_lat + max_lat) / 2.0
    lat_deg_per_m = 1.0 / 111_320.0
    lon_deg_per_m = 1.0 / (111_320.0 * math.cos(math.radians(lat0)) + 1e-9)
    lat_step = step_m * lat_deg_per_m; lon_step = step_m * lon_deg_per_m

    with engine.begin() as conn:
        q = text(f"""
            WITH raw AS (
              SELECT
                FLOOR( (gps_lat - :min_lat) / :lat_step )::int AS cy,
                FLOOR( (gps_lon - :min_lon) / :lon_step )::int AS cx,
                (overall->>'wear_score')::float AS wear,
                "timestamp" AS ts
              FROM lane_wear_results
              WHERE gps_lat IS NOT NULL AND gps_lon IS NOT NULL
                AND created_at >= NOW() - INTERVAL '{int(window_h)} hour'
                AND gps_lat BETWEEN :min_lat AND :max_lat
                AND gps_lon BETWEEN :min_lon AND :max_lon
            ),
            ag AS (
              SELECT
                cy, cx, COUNT(*) AS n,
                AVG(wear) AS wear_avg,
                MAX(wear) AS wear_max,
                PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY wear) AS wear_p90,
                MAX(ts) AS last_ts
              FROM raw
              GROUP BY cy, cx
            )
            SELECT * FROM ag WHERE n >= :min_count
        """)
        rows = conn.execute(q, {"min_lat": min_lat, "max_lat": max_lat,
                                "min_lon": min_lon, "max_lon": max_lon,
                                "lat_step": lat_step, "lon_step": lon_step,
                                "min_count": min_count}).mappings().all()

    cells = []
    for r in rows:
        cy, cx = r["cy"], r["cx"]
        lat = min_lat + (cy + 0.5) * lat_step
        lon = min_lon + (cx + 0.5) * lon_step
        rep = r["wear_p90"] if agg == "p90" else (r["wear_max"] if agg == "max" else r["wear_avg"])
        cells.append({
            "cy": int(cy), "cx": int(cx),
            "lat": float(lat), "lon": float(lon),
            "count": int(r["n"]),
            "wear_avg": float(r["wear_avg"]),
            "wear_max": float(r["wear_max"]),
            "wear_p90": float(r["wear_p90"]),
            "rep": float(rep),
            "last_ts": r["last_ts"].isoformat() if r["last_ts"] else None,
        })
    return {"bbox": [min_lat, min_lon, max_lat, max_lon], "step_m": step_m, "window_h": window_h, "agg": agg, "cells": cells}

def _priority_formula_sql(window_h: int) -> str:
    return f"""
    WITH ranked AS (
      SELECT id, device_id, created_at, (overall->>'wear_score')::float AS wear,
             ROW_NUMBER() OVER (PARTITION BY device_id ORDER BY created_at DESC) AS rn
      FROM lane_wear_results
      WHERE created_at >= NOW() - INTERVAL '{int(window_h)} hour' AND device_id IS NOT NULL
    ),
    agg AS (
      SELECT device_id,
             MAX(CASE WHEN rn=1 THEN wear END)               AS w_last,
             AVG(CASE WHEN rn<=3 THEN wear END)              AS w_last3,
             AVG(CASE WHEN rn BETWEEN 4 AND 6 THEN wear END) AS w_prev3,
             SUM(CASE WHEN rn<=3 AND wear >= :critical THEN 1 ELSE 0 END) AS crit3,
             COUNT(*) AS n,
             MAX(created_at) AS last_ts
      FROM ranked GROUP BY device_id
    ),
    score AS (
      SELECT device_id, w_last, w_last3, w_prev3,
             COALESCE(w_last3 - COALESCE(w_prev3, w_last3), 0) AS trend,
             crit3, n, last_ts,
             EXTRACT(EPOCH FROM (NOW() - last_ts))/3600.0 AS hours_since,
             (
               0.50 * LEAST(GREATEST(w_last,0),100)/100.0 +
               0.20 * (COALESCE(w_last3 - COALESCE(w_prev3, w_last3),0)/100.0) +
               0.20 * (crit3/3.0) +
               0.10 * LEAST(n/10.0, 1.0)
             ) * EXP(- LEAST(EXTRACT(EPOCH FROM (NOW() - last_ts))/3600.0, 168)/72.0)
             AS priority
      FROM agg
    )
    SELECT * FROM score
    """

@app.get("/candidates/rank")
def candidates_rank(window_h: int = 168, critical: float = 70.0, limit: int = 20, offset: int = 0):
    sql = _priority_formula_sql(window_h) + " ORDER BY priority DESC LIMIT :limit OFFSET :offset"
    with engine.begin() as conn:
        rows = conn.execute(text(sql), {"critical": critical, "limit": limit, "offset": offset}).mappings().all()
    return [
        {"device_id": r["device_id"], "priority": float(r["priority"] or 0.0),
         "w_last": float(r["w_last"] or 0.0), "trend": float(r["trend"] or 0.0),
         "crit3": int(r["crit3"] or 0), "n": int(r["n"] or 0),
         "last_ts": r["last_ts"].isoformat() if r["last_ts"] else None,
         "hours_since": float(r["hours_since"] or 0.0)}
        for r in rows
    ]

@app.get("/candidates/rank_for_id/{id:int}")
def candidate_rank_for_id(id: int, window_h: int = 168, critical: float = 70.0):
    with engine.begin() as conn:
        row = conn.execute(select(lane_wear_results.c.device_id).where(lane_wear_results.c.id == id)).first()
    if not row or not row[0]: return {"rank": None, "total": 0, "row": None}
    device_id = row[0]
    sql_all = _priority_formula_sql(window_h)
    with engine.begin() as conn:
        all_rows = conn.execute(text(sql_all), {"critical": critical}).mappings().all()
    all_rows = sorted(all_rows, key=lambda r: (r["priority"] or 0.0), reverse=True)
    total = len(all_rows)
    rank = next((i+1 for i, r in enumerate(all_rows) if r["device_id"] == device_id), None)
    cur = next((r for r in all_rows if r["device_id"] == device_id), None)
    if cur is None: return {"rank": None, "total": total, "row": None}
    def _shape(r):
        return {"device_id": r["device_id"], "priority": float(r["priority"] or 0.0),
                "w_last": float(r["w_last"] or 0.0), "trend": float(r["trend"] or 0.0),
                "crit3": int(r["crit3"] or 0), "n": int(r["n"] or 0),
                "last_ts": r["last_ts"].isoformat() if r["last_ts"] else None}
    top10 = [_shape(r) for r in all_rows[:10]]
    return {"rank": rank, "total": total, "row": _shape(cur), "top10": top10}
