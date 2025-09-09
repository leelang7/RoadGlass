import requests

API_URL = "http://localhost:8000/blur"   # 서버 주소
INPUT_IMAGE = "zidane.jpg"               # 업로드할 원본 이미지
OUTPUT_IMAGE = "blurred.jpg"             # 저장할 결과 파일

# 쿼리 파라미터
params = {
    "method": "gaussian",   # 'gaussian' | 'pixelate'
    "conf": 0.3,            # confidence threshold
    "blur_strength": 41,    # gaussian kernel size (홀수 권장)
    "pixel_size": 16,       # pixelation block size
    "max_size": 1280,       # 추론 시 긴 변 크기 제한
    "jpeg_quality": 90      # 결과 JPEG 품질
}

# 멀티파트 전송
with open(INPUT_IMAGE, "rb") as f:
    files = {"file": (INPUT_IMAGE, f, "image/jpeg")}
    response = requests.post(API_URL, params=params, files=files)

# 결과 확인 및 저장
if response.status_code == 200:
    with open(OUTPUT_IMAGE, "wb") as out:
        out.write(response.content)
    print(f"✅ 저장 완료: {OUTPUT_IMAGE}")
else:
    print(f"❌ 에러 {response.status_code}: {response.text}")
