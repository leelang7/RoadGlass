import io
import os
from typing import Literal, List

import cv2
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from ultralytics import YOLO

# =========================
# 환경설정 (필요시 조정)
# =========================
MODEL_PATH = os.getenv("YOLO_MODEL", "yolov11n-face.pt")  # 얼굴 전용 가중치 권장
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 20 * 1024 * 1024))  # 20MB
ALLOWED_MIME = {"image/jpeg", "image/png"}  # 모바일 기본 포맷

# =========================
# 앱 & CORS
# =========================
app = FastAPI(title="Face Blur API (mobile-ready)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요시 도메인 화이트리스트로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 모델: 싱글톤 로딩
# =========================
_model: YOLO | None = None
def get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)  # CUDA 있으면 자동 사용
    return _model

# =========================
# 유틸
# =========================
def read_image_from_upload(file: UploadFile) -> Image.Image:
    # 용량/형식 검증
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=415, detail=f"Unsupported Content-Type: {file.content_type}")
    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    # EXIF 회전 보정 + RGB
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
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, buf = cv2.imencode(".jpg", img_bgr, encode_param)
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()

def resize_long_edge(pil_img: Image.Image, max_edge: int) -> Image.Image:
    w, h = pil_img.size
    m = max(w, h)
    if m <= max_edge:
        return pil_img
    scale = max_edge / m
    return pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

def pick_face_boxes(result, names) -> np.ndarray:
    """모델 classes에 'face'가 있으면 face만, 단일 클래스라면 전체."""
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 4), dtype=float)

    cls = result.boxes.cls.detach().cpu().numpy().astype(int)
    xyxy = result.boxes.xyxy.detach().cpu().numpy()

    face_ids: List[int] = []
    if isinstance(names, dict):
        face_ids = [k for k, v in names.items() if str(v).lower() == "face"]

    if face_ids:
        mask = np.isin(cls, face_ids)
        return xyxy[mask]
    return xyxy  # 단일 클래스 등: 전부 사용

def apply_blur(
    img_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    method: Literal["gaussian", "pixelate"] = "gaussian",
    blur_strength: int = 31,
    pixel_size: int = 16,
) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()

    # 커널/픽셀 파라미터 정리
    if method == "gaussian":
        k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        k = max(3, min(k, 199))
    else:
        px = max(2, min(pixel_size, 128))

    for (x1, y1, x2, y2) in boxes_xyxy:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue

        roi = out[y1:y2, x1:x2]
        if method == "gaussian":
            roi_blur = cv2.GaussianBlur(roi, (k, k), 0)
        else:
            h2, w2 = roi.shape[:2]
            small = cv2.resize(roi, (max(1, w2 // px), max(1, h2 // px)), interpolation=cv2.INTER_LINEAR)
            roi_blur = cv2.resize(small, (w2, h2), interpolation=cv2.INTER_NEAREST)

        out[y1:y2, x1:x2] = roi_blur

    return out

# =========================
# 헬스체크
# =========================
@app.get("/health")
def health():
    model = get_model()
    return {"status": "ok", "model": os.path.basename(MODEL_PATH), "classes": getattr(model, "names", {})}

# =========================
# 모바일 멀티파트 업로드 전용 엔드포인트
# =========================
@app.post("/blur")
def blur(
    file: UploadFile = File(..., description="이미지 파일 (multipart/form-data, key: file)"),
    conf: float = Query(0.25, ge=0.01, le=1.0, description="Confidence threshold"),
    iou: float = Query(0.50, ge=0.05, le=0.95, description="NMS IoU"),
    method: Literal["gaussian", "pixelate"] = Query("gaussian"),
    blur_strength: int = Query(31, ge=3, le=199),
    pixel_size: int = Query(16, ge=2, le=128),
    max_size: int = Query(1280, ge=320, le=4096, description="추론용 긴 변 최대 길이"),
    jpeg_quality: int = Query(90, ge=60, le=100, description="응답 JPEG 품질"),
):
    # 1) 입력 이미지 로드 + 회전 보정
    pil_img = read_image_from_upload(file)

    # 2) 추론 전 다운스케일(모바일 촬영 대용량 대비)
    pil_img_rs = resize_long_edge(pil_img, max_size)

    # 3) 추론
    model = get_model()
    results = model.predict(pil_img_rs, conf=conf, iou=iou, imgsz=max_size, verbose=False)
    r = results[0] if results else None
    boxes = pick_face_boxes(r, getattr(model, "names", {}))

    # 4) 블러 적용
    img_bgr = pil_to_cv2(pil_img_rs)
    if len(boxes) > 0:
        img_bgr = apply_blur(img_bgr, boxes, method, blur_strength, pixel_size)

    # 5) JPEG로 반환
    return Response(content=cv2_to_jpeg_bytes(img_bgr, quality=jpeg_quality), media_type="image/jpeg")
