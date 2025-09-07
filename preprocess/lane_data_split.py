import os, json, random, shutil
from glob import glob
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import cv2
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.ops import unary_union

# ============== 경로/설정 ==============
RAW_ROOT   = r"C:\Users\302-t\Downloads\070\raw_val"
IMAGES_DIR = os.path.join(RAW_ROOT, "images")
JSON_DIR   = os.path.join(RAW_ROOT, "labels_json")   # 없으면 RAW_ROOT/*.json도 탐색
OUT_ROOT   = os.path.join(RAW_ROOT, "lane_split_yolo")  # 결과 루트

SPLIT_TRAIN = 0.8
SEED        = 42

# 폴리라인 두께(px) → buffer 반지름은 절반
STROKE_PX   = 8

# 버퍼 모양
BUFFER_JOIN = "round"   # 'round'|'mitre'|'bevel'
BUFFER_CAP  = "round"   # 'round'|'flat'|'square'
MITRE_LIMIT = 5.0

# 폴리곤 단순화 허용오차(px) (0이면 원본 유지)
APPROX_EPS_PX = 1.0

random.seed(SEED)
# ======================================

def log(*a): print(*a, flush=True)

def ensure_dirs():
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(OUT_ROOT, sub), exist_ok=True)

def find_jsons():
    files = []
    if os.path.isdir(JSON_DIR):
        files += glob(os.path.join(JSON_DIR, "*.json"))
    files += [p for p in glob(os.path.join(RAW_ROOT, "*.json")) if p not in files]
    return sorted(list(set(files)))

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_img_size_and_scale(file_name: str, declared_size: Any) -> Tuple[int,int,float,float]:
    """실제 이미지 W,H와 JSON 좌표 → 실제 픽셀로 스케일(sx,sy)을 돌려줌."""
    W_act = H_act = None
    if file_name:
        p = os.path.join(IMAGES_DIR, file_name)
        if os.path.isfile(p):
            img = cv2.imread(p)
            if img is not None:
                H_act, W_act = img.shape[:2]

    W_json = H_json = None
    if isinstance(declared_size, (list, tuple)) and len(declared_size) == 2:
        H_json, W_json = int(declared_size[0]), int(declared_size[1])

    # 스케일 계산
    if W_act is None or H_act is None:
        raise RuntimeError("이미지를 열 수 없습니다: " + str(file_name))
    if W_json is None or H_json is None:
        # JSON에 사이즈가 없으면 JSON 좌표가 이미 실제 픽셀이라고 가정 -> 스케일 1
        return W_act, H_act, 1.0, 1.0

    sx = float(W_act) / float(W_json) if W_json else 1.0
    sy = float(H_act) / float(H_json) if H_json else 1.0
    return W_act, H_act, sx, sy

def norm_key(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(s).strip().lower()).strip("_")

def class_key_from_ann(ann: Dict[str, Any]) -> str:
    base = norm_key(ann.get("class", "unknown"))
    color = lane_type = None
    for attr in ann.get("attributes", []):
        code = str(attr.get("code","")).lower()
        val  = norm_key(attr.get("value",""))
        if code == "lane_color": color = val
        elif code == "lane_type": lane_type = val
    parts = [base]
    if color: parts.append(color)
    if lane_type: parts.append(lane_type)
    return "_".join(parts)

def collect_classes(json_files: List[str]) -> Tuple[Dict[str,int], List[str]]:
    keys = set()
    for jp in json_files:
        js = read_json(jp)
        for ann in js.get("annotations", []):
            if str(ann.get("category","")).lower() != "polyline":
                continue
            key = class_key_from_ann(ann)
            if key: keys.add(key)
    names = sorted(keys)
    return {k:i for i,k in enumerate(names)}, names

def shapely_buffer_lines(lines_xy: List[List[Tuple[float,float]]], radius_px: float):
    geoms = []
    for pts in lines_xy:
        if len(pts) < 2: 
            continue
        ls = LineString(pts)
        if ls.length == 0: 
            continue
        poly = ls.buffer(
            radius_px,
            cap_style={"round":1, "flat":2, "square":3}[BUFFER_CAP],
            join_style={"round":1, "mitre":2, "bevel":3}[BUFFER_JOIN],
            mitre_limit=MITRE_LIMIT
        )
        geoms.append(poly)
    if not geoms:
        return None
    return unary_union(geoms)

def simplify_polygon(poly: Polygon, eps_px: float):
    if eps_px and eps_px > 0:
        return poly.simplify(eps_px, preserve_topology=True)
    return poly

def polygon_to_yolo_lines(geom, W: int, H: int, cls_id: int) -> List[str]:
    def norm_coords(coords):
        out = []
        for x, y in coords:
            xn = max(0.0, min(1.0, x / float(W)))
            yn = max(0.0, min(1.0, y / float(H)))
            out += [f"{xn:.6f}", f"{yn:.6f}"]
        return out

    lines = []
    if isinstance(geom, Polygon):
        g = simplify_polygon(geom, APPROX_EPS_PX)
        coords = list(g.exterior.coords)[:-1]  # 닫힘점 제거
        if len(coords) >= 3:
            lines.append(f"{cls_id} " + " ".join(norm_coords(coords)))
    elif isinstance(geom, MultiPolygon):
        for p in geom.geoms:
            g = simplify_polygon(p, APPROX_EPS_PX)
            coords = list(g.exterior.coords)[:-1]
            if len(coords) >= 3:
                lines.append(f"{cls_id} " + " ".join(norm_coords(coords)))
    return lines

def convert_json_to_yolo_lines(jp: str, class_map: Dict[str,int]) -> Tuple[str, List[str]]:
    js = read_json(jp)
    image_info = js.get("image") or js.get("images") or {}
    file_name = image_info.get("file_name")
    if not file_name:
        return None, []
    W, H, sx, sy = get_img_size_and_scale(file_name, image_info.get("image_size"))

    # 클래스별로 폴리라인 좌표(스케일 보정 적용)를 모은다
    grouped = defaultdict(list)
    for ann in js.get("annotations", []):
        if str(ann.get("category","")).lower() != "polyline":
            continue
        key = class_key_from_ann(ann)
        if key not in class_map:
            continue
        pts = []
        for p in ann.get("data", []):
            # JSON 좌표 → 실제 픽셀로 스케일 보정
            x = float(p["x"]) * sx
            y = float(p["y"]) * sy
            # 경계 클립
            x = min(max(x, 0.0), W-1.0)
            y = min(max(y, 0.0), H-1.0)
            pts.append((x, y))
        if len(pts) >= 2:
            grouped[key].append(pts)

    # 각 클래스별로 벡터 버퍼링 → 폴리곤 → YOLO 세그 줄 생성
    lines = []
    for key, list_of_lines in grouped.items():
        geom = shapely_buffer_lines(list_of_lines, STROKE_PX/2.0)
        if geom is None:
            continue
        cls_id = class_map[key]
        lines += polygon_to_yolo_lines(geom, W, H, cls_id)

    stem, _ = os.path.splitext(os.path.basename(file_name))
    return stem, lines

def copy_image_by_stem(stem: str) -> str:
    # 지정 파일명 그대로가 아닐 수 있으므로 stem 기반 탐색
    for ext in [".jpg",".jpeg",".png",".bmp",".webp",".JPG",".PNG"]:
        p = os.path.join(IMAGES_DIR, stem+ext)
        if os.path.isfile(p):
            return p
    # 마지막 수단: 같은 스템 검색
    cands = [p for p in glob(os.path.join(IMAGES_DIR, "*"))
             if os.path.isfile(p) and os.path.splitext(os.path.basename(p))[0] == stem]
    return cands[0] if cands else None

def write_yaml(names: List[str]):
    y = os.path.join(OUT_ROOT, "data.yaml")
    with open(y, "w", encoding="utf-8") as f:
        # Corrected line
        path_str = OUT_ROOT.replace('\\', '/')
        f.write(f"path: {path_str}\n")
        
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {len(names)}\n")
        f.write("names:\n")
        for i, n in enumerate(names):
            f.write(f"  {i}: {n}\n")
    log(f"[INFO] data.yaml 생성: {y}")

def split_and_save(items: List[Tuple[str, List[str]]]):
    random.shuffle(items)
    cut = int(len(items) * SPLIT_TRAIN)
    splits = {"train": items[:cut], "val": items[cut:]}
    for split, arr in splits.items():
        imgs = lbls = 0
        for stem, lines in arr:
            img_path = copy_image_by_stem(stem)
            if img_path is None:
                continue
            out_img = os.path.join(OUT_ROOT, "images", split, os.path.basename(img_path))
            os.makedirs(os.path.dirname(out_img), exist_ok=True)
            shutil.copy2(img_path, out_img)
            imgs += 1

            if lines:
                out_lbl = os.path.join(OUT_ROOT, "labels", split, stem + ".txt")
                os.makedirs(os.path.dirname(out_lbl), exist_ok=True)
                with open(out_lbl, "w", encoding="utf-8") as f:
                    for L in lines:
                        f.write(L + "\n")
                lbls += 1
        log(f"[OUT:{split}] images={imgs}, labels={lbls}")

def main():
    ensure_dirs()
    json_files = find_jsons()
    if not json_files:
        log("[ERROR] JSON 없음"); return

    # 1) 실제 등장하는 클래스 조합 수집
    class_map, class_names = collect_classes(json_files)
    log(f"[INFO] classes={len(class_names)}")
    for i, n in enumerate(class_names): log(f"  - {i}: {n}")

    # 2) 변환 (좌표 스케일 보정 + YOLO 정규화 포함)
    items = []
    miss_img = 0
    for jp in json_files:
        try:
            stem, lines = convert_json_to_yolo_lines(jp, class_map)
        except Exception as e:
            log(f"[WARN] 변환 실패: {jp} ({e})"); continue
        if stem is None:
            miss_img += 1; continue
        items.append((stem, lines))
    log(f"[INFO] 변환 샘플={len(items)}, 이미지 정보 누락={miss_img}")

    # 3) 8:2 스플릿 + 복사/저장
    split_and_save(items)

    # 4) data.yaml 생성
    write_yaml(class_names)

    log(f"[DONE] 출력 루트: {OUT_ROOT}")

if __name__ == "__main__":
    main()
