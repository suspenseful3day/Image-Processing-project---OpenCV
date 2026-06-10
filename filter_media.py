import cv2
import numpy as np
import threading
import time
import os
import urllib.request
from queue import Queue, Empty
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ============================================================
# 파일 경로 헬퍼: 이미지 파일을 .py 파일과 같은 폴더에서 찾게 함
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def asset_path(filename: str) -> str:
    return os.path.join(BASE_DIR, filename)


# ============================================================
# 0. MediaPipe 0.10.x Tasks API — 모델 다운로드 및 detector 생성
# ============================================================
_MP_MODEL_PATH = os.path.join(BASE_DIR, "models", "blaze_face_short_range.tflite")

def download_mediapipe_model():
    """BlazeFace Short-Range .tflite 모델 자동 다운로드"""
    model_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    if not os.path.exists(_MP_MODEL_PATH) or os.path.getsize(_MP_MODEL_PATH) < 10_000:
        print("📥 MediaPipe BlazeFace 모델 다운로드 중...")
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_detector/blaze_face_short_range/float16/latest/"
            "blaze_face_short_range.tflite"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp, \
                 open(_MP_MODEL_PATH, "wb") as f:
                f.write(resp.read())
            print(f"✅ 모델 다운로드 완료 ({os.path.getsize(_MP_MODEL_PATH):,} bytes)")
        except Exception as e:
            print(f"❌ 모델 다운로드 실패: {e}")
            raise
    return _MP_MODEL_PATH


def create_mediapipe_detector(model_path: str, min_confidence: float = 0.5):
    """MediaPipe 0.10.x Tasks API FaceDetector 생성 (스레드마다 별도 생성 필요)"""
    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    opts = mp_vision.FaceDetectorOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,
        min_detection_confidence=min_confidence,
        min_suppression_threshold=0.3,
    )
    return mp_vision.FaceDetector.create_from_options(opts)


# ============================================================
# 해상도 설정 (960x540)
# ============================================================
CAM_W, CAM_H = 960, 540


# ============================================================
# 1. 비동기 웹캠 스트림
# ============================================================
class WebcamStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stopped = False
        self.frame   = None
        self._lock   = threading.Lock()

    def start(self):
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while not self.stopped:
            ok, frame = self.stream.read()
            if not ok:
                self.stopped = True
                break
            with self._lock:
                self.frame = frame

    def read(self):
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.stream.release()


# ============================================================
# 오버레이 이미지 로드 및 캐싱 (하트, 천사링, 스파클)
# ============================================================
HEART_IMG = cv2.imread(asset_path("heart.png"), cv2.IMREAD_COLOR)
HALO_IMG = cv2.imread(asset_path("1780416773924_image.png"), cv2.IMREAD_UNCHANGED)

if HEART_IMG is None:
    print("⚠️ 경고: 'heart.png' 이미지를 불러올 수 없습니다. 경로를 확인하세요.")
else:
    print("✅ 'heart.png' 로드 완료")

if HALO_IMG is None:
    print("⚠️ 경고: 천사링 이미지를 불러올 수 없습니다. 파일명/위치를 확인하세요.")
else:
    print("✅ 천사링 이미지 로드 완료")


_heart_cache: dict = {}
_halo_cache: dict = {}


def get_resized_heart_with_mask(target_w: int):
    """핑크색 추출 마스크 알고리즘을 적용한 하트 및 알파 마스크 반환"""
    if target_w not in _heart_cache:
        if HEART_IMG is None:
            return None, None
        
        ratio = HEART_IMG.shape[0] / HEART_IMG.shape[1]
        th = max(int(target_w * ratio), 1)
        resized_heart = cv2.resize(HEART_IMG, (target_w, th))
        
        heart_hsv = cv2.cvtColor(resized_heart, cv2.COLOR_BGR2HSV)
        lower_pink = np.array([140, 30, 50])
        upper_pink = np.array([180, 255, 255])
        
        pink_mask = cv2.inRange(heart_hsv, lower_pink, upper_pink)
        alpha_channel = cv2.GaussianBlur(pink_mask, (3, 3), 0) / 255.0
        custom_alpha_mask = cv2.merge([alpha_channel, alpha_channel, alpha_channel])
        
        _heart_cache[target_w] = (resized_heart, custom_alpha_mask)
        if len(_heart_cache) > 10:
            _heart_cache.pop(next(iter(_heart_cache)))
            
    return _heart_cache[target_w]


def make_overlay_bgr_and_alpha(img: np.ndarray, target_w: int, mode: str):

    if img is None:
        return None, None

    ratio = img.shape[0] / img.shape[1]
    target_h = max(int(target_w * ratio), 1)

    resized = cv2.resize(
        img,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA
    )

    bgr = resized[:, :, :3]

    # ---------- PNG 알파 있으면 ----------
    if resized.shape[2] == 4:

        alpha = resized[:, :, 3].astype(np.float32)

    else:
        alpha = np.ones(
            resized.shape[:2],
            dtype=np.float32
        ) * 255


    # ---------- 체크무늬 제거 ----------
    hsv = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2HSV
    )

    # 회색/흰색 체크무늬 영역 추출
    checker = cv2.inRange(
        hsv,
        np.array([0,0,140]),
        np.array([180,60,255])
    )

    # 체크무늬 부분 알파=0
    alpha[checker > 0] = 0


    # 부드러운 경계
    alpha = cv2.GaussianBlur(
        alpha,
        (9,9),
        0
    )

    alpha = alpha.astype(np.float32)/255.0

    alpha_3 = cv2.merge([
        alpha,
        alpha,
        alpha
    ])

    return bgr, alpha_3


def get_resized_halo_with_mask(target_w: int):
    if target_w not in _halo_cache:
        _halo_cache[target_w] = make_overlay_bgr_and_alpha(HALO_IMG, target_w, "halo")
        if len(_halo_cache) > 10:
            _halo_cache.pop(next(iter(_halo_cache)))
    return _halo_cache[target_w]



def overlay_image(base: np.ndarray, overlay: np.ndarray, alpha: np.ndarray, x: int, y: int, strength: float = 1.0):
    """base 위에 overlay를 알파 블렌딩. 화면 밖으로 나가도 안전하게 처리."""
    if overlay is None or alpha is None:
        return base

    h, w = overlay.shape[:2]
    H, W = base.shape[:2]

    x1, x2 = max(0, x), min(W, x + w)
    y1, y2 = max(0, y), min(H, y + h)

    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        return base

    ox1 = x1 - x
    ox2 = ox1 + (x2 - x1)
    oy1 = y1 - y
    oy2 = oy1 + (y2 - y1)

    roi = base[y1:y2, x1:x2].astype(np.float32)
    ov = overlay[oy1:oy2, ox1:ox2].astype(np.float32)
    al = alpha[oy1:oy2, ox1:ox2].astype(np.float32) * strength
    al = np.clip(al, 0.0, 1.0)

    blended = ov * al + roi * (1.0 - al)
    blended = np.clip(blended,0,255)
    base[y1:y2, x1:x2] = blended.astype(np.uint8)
    return base


# ============================================================
# 2. 필터 함수 구현부
# ============================================================
_vignette_cache: dict = {}
_scanline_cache: dict = {}

def get_vignette(h: int, w: int) -> np.ndarray:
    key = (h, w)
    if key not in _vignette_cache:
        vx = np.linspace(0, 1, w)
        vy = np.linspace(0, 1, h)
        vX, vY = np.meshgrid(vx, vy)
        vignette = 1.0 - 0.6 * ((vX - 0.5)**2 + (vY - 0.5)**2) * 3.5
        vignette = np.clip(vignette, 0.2, 1.0)[:, :, np.newaxis]
        _vignette_cache[key] = vignette.astype(np.float32)
    return _vignette_cache[key]

def get_scanline(h: int, w: int) -> np.ndarray:
    key = (h, w)
    if key not in _scanline_cache:
        mask = np.ones((h, w, 3), dtype=np.float32)
        mask[::2, :] = 0.55
        _scanline_cache[key] = mask
    return _scanline_cache[key]


def filter_retro(frame: np.ndarray, bboxes) -> np.ndarray:
    """📺 VHS 레트로/스케치 필터 - 최적화 버전"""
    h, w = frame.shape[:2]

    # ── 1. 색수차 (채널 어긋남) ──
    shift = int(w * 0.008)
    b, g, r = cv2.split(frame)
    r = np.roll(r, shift, axis=1)
    b = np.roll(b, -shift, axis=1)
    result = cv2.merge([b, g, r]).astype(np.float32)

    # ── 2. 노이즈 ──
    noise = np.random.randint(-10, 10, result.shape, dtype=np.int8).astype(np.float32)
    result = np.clip(result + noise, 0, 255)

    # ── 3. 스캔라인 ──
    result = result * get_scanline(h, w)

    # ── 4. 글리치 ──
    glitch_seed = int(time.time() * 4) % 30
    if glitch_seed < 4:
        gy = np.random.randint(0, h - 20)
        gshift = np.random.randint(-18, 18)
        result[gy:gy+8, :] = np.roll(result[gy:gy+8, :], gshift, axis=1)

    # ── 5. 비네팅 ──
    result = np.clip(result * get_vignette(h, w), 0, 255)

    # ── 6. VHS UI ──
    out = result.astype(np.uint8)
    if int(time.time() * 2) % 2 == 0:
        cv2.circle(out, (24, 24), 7, (0, 0, 200), -1)
        cv2.putText(out, "REC", (36, 29), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 220), 1, cv2.LINE_AA)
    cv2.putText(out, time.strftime("SP  %H:%M:%S"), (w - 160, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 180), 1, cv2.LINE_AA)

    return out


def filter_halo(frame: np.ndarray, bboxes) -> np.ndarray:
    """😇 천사링(이미지 or 대체도형) + 깔끔한 코딩형 기본 반짝이 필터"""
    result = frame.copy()
    if not bboxes:
        return result

    for bbox in bboxes:
        x, y, w, h = bbox
        y_off = int(np.sin(time.time() * 2.5) * 6)

        # 1) 헤일로(천사링) 크기/위치 처리
        halo_w = max(int(w * 1.1), 80)
        resized_halo, halo_alpha = get_resized_halo_with_mask(halo_w)

        if resized_halo is not None:
            halo_h = resized_halo.shape[0]
            halo_x = x + w // 2 - halo_w // 2
            halo_y = y - int(halo_h * 0.7) + y_off
            result = overlay_image(result, resized_halo, halo_alpha, halo_x, halo_y, strength=1.0)
        else:
            # 천사링 에셋도 없을 때 대체 타원 그리기
            halo_h = max(int(w * 0.30), 20)
            halo_y = y - halo_h + y_off
            cv2.ellipse(result, (x + w // 2, halo_y + halo_h // 2),
                        (w // 2, max(w // 9, 8)), 0, 0, 360,
                        (255, 220, 100), 4, cv2.LINE_AA)

        # 2) ✨ 대안 스파클: 이미지를 안 쓰고 여러 개의 예쁜 십자 반짝이를 코드로 생성
        flicker1 = abs(np.sin(time.time() * 4.0))
        flicker2 = abs(np.cos(time.time() * 3.5))
        
        # 반짝이 색상 정의 (B, G, R)
        yellow = (120, 240, 255)
        cyan = (255, 230, 150)
        
        # 얼굴 주변 4곳에 타이밍이 다르게 반짝이는 십자가 배치
        if flicker1 > 0.3:
            # 반짝이 1 (좌측 상단 - 노란색)
            cx1, cy1 = x - 20, y - 30 + y_off
            cv2.line(result, (cx1 - 8, cy1), (cx1 + 8, cy1), yellow, 2, cv2.LINE_AA)
            cv2.line(result, (cx1, cy1 - 8), (cx1, cy1 + 8), yellow, 2, cv2.LINE_AA)
            
            # 반짝이 2 (우측 하단 - 하늘색)
            cx2, cy2 = x + w + 15, y + h // 2
            cv2.line(result, (cx2 - 6, cy2), (cx2 + 6, cy2), cyan, 2, cv2.LINE_AA)
            cv2.line(result, (cx2, cy2 - 6), (cx2, cy2 + 6), cyan, 2, cv2.LINE_AA)

        if flicker2 > 0.4:
            # 반짝이 3 (우측 상단 - 노란색)
            cx3, cy3 = x + w + 20, y - 25 + y_off
            cv2.line(result, (cx3 - 10, cy3), (cx3 + 10, cy3), yellow, 2, cv2.LINE_AA)
            cv2.line(result, (cx3, cy3 - 10), (cx3, cy3 + 10), yellow, 2, cv2.LINE_AA)
            
            # 반짝이 4 (좌측 중앙 - 하늘색)
            cx4, cy4 = x - 25, y + h // 3
            cv2.line(result, (cx4 - 7, cy4), (cx4 + 7, cy4), cyan, 2, cv2.LINE_AA)
            cv2.line(result, (cx4, cy4 - 7), (cx4, cy4 + 7), cyan, 2, cv2.LINE_AA)

    return result


def filter_heart(frame: np.ndarray, bboxes) -> np.ndarray:
    """💖 하트 필터"""
    result = frame.copy()
    if not bboxes:
        return result

    for bbox in bboxes:
        x, y, w, h = bbox

        nw = int(w * 1.3)
        resized_heart, custom_alpha_mask = get_resized_heart_with_mask(nw)
        y_off = int(np.sin(time.time() * 3.5) * 8)

        if resized_heart is not None:
            nh = resized_heart.shape[0]
            nx = x - int((nw - w) / 2)
            ny = y - int(nh * 0.85) + y_off
            
            x1, x2 = max(0, nx), min(result.shape[1], nx + nw)
            y1, y2 = max(0, ny), min(result.shape[0], ny + nh)
            
            if (x2 - x1) > 0 and (y2 - y1) > 0:
                roi = result[y1:y2, x1:x2].astype(np.float32)
                hx1 = x1 - nx
                hx2 = hx1 + (x2 - x1)
                hy1 = y1 - ny
                hy2 = hy1 + (y2 - y1)
                
                cropped_heart_bgr = resized_heart[hy1:hy2, hx1:hx2].astype(np.float32)
                cropped_alpha_mask = custom_alpha_mask[hy1:hy2, hx1:hx2]
                
                composite = (cropped_heart_bgr * cropped_alpha_mask) + (roi * (1.0 - cropped_alpha_mask))
                result[y1:y2, x1:x2] = composite.astype('uint8')
        else:
            cx, cy = x + w // 2, y - 35 + y_off
            r = max(w // 8, 15)
            cv2.circle(result, (cx - r // 2, cy), r // 2, (0, 0, 220), -1)
            cv2.circle(result, (cx + r // 2, cy), r // 2, (0, 0, 220), -1)
            pts = np.array([[cx - r, cy], [cx + r, cy], [cx, cy + r + 4]], np.int32)
            cv2.fillPoly(result, [pts], (0, 0, 220))
        
    return result

def filter_mosaic(frame: np.ndarray, bboxes) -> np.ndarray:
    result = frame.copy()
    if not bboxes:
        return result

    for bbox in bboxes:
        x, y, w, h = bbox

        cx, cy = x + w // 2, y + h // 2
        shrink = 0.85
        bw, bh = int(w * shrink), int(h * shrink)
        pad = int(max(w, h) * 0.4)

        rx1 = max(0, cx - bw // 2 - pad)
        ry1 = max(0, cy - bh // 2 - pad)
        rx2 = min(frame.shape[1], cx + bw // 2 + pad)
        ry2 = min(frame.shape[0], cy + bh // 2 + pad)
        rw, rh = rx2 - rx1, ry2 - ry1
        if rw <= 0 or rh <= 0:
            continue

        roi = result[ry1:ry2, rx1:rx2].copy()

        # ── 핵심 최적화: 다운스케일 → 블러 1번 → 업스케일 ──
        scale = 0.15  # 15% 크기로 줄여서 블러
        small = cv2.resize(roi, (max(1, int(rw * scale)), max(1, int(rh * scale))),
                           interpolation=cv2.INTER_LINEAR)
        small = cv2.GaussianBlur(small, (15, 15), 0)  # 작은 크기에서 1번만
        blurred = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_LINEAR)

        # ── 타원 마스크 (변경 없음) ──
        mask = np.zeros((rh, rw), dtype=np.uint8)
        ellipse_cx = cx - rx1
        ellipse_cy = cy - ry1
        cv2.ellipse(mask, (ellipse_cx, ellipse_cy), (bw // 2, bh // 2), 0, 0, 360, 255, -1)

        feather = max(int(min(bw, bh) * 0.55), 31)
        if feather % 2 == 0:
            feather += 1
        mask = cv2.GaussianBlur(mask, (feather, feather), 0)

        mask_f   = mask.astype(np.float32) / 255.0
        mask_3ch = cv2.merge([mask_f, mask_f, mask_f])

        roi_f     = roi.astype(np.float32)
        blurred_f = blurred.astype(np.float32)
        blended   = (blurred_f * mask_3ch + roi_f * (1.0 - mask_3ch)).astype(np.uint8)
        result[ry1:ry2, rx1:rx2] = blended

    return result


_mask_cache = {}  # 함수 밖에 선언

def get_cached_mask(rw, rh, ellipse_cx, ellipse_cy, bw, bh, feather):
    # 크기 기준으로 캐싱 (매 프레임 재계산 방지)
    key = (rw, rh, ellipse_cx, ellipse_cy, bw, bh, feather)
    if key not in _mask_cache:
        mask = np.zeros((rh, rw), dtype=np.uint8)
        cv2.ellipse(mask, (ellipse_cx, ellipse_cy), (bw // 2, bh // 2), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (feather, feather), 0)
        _mask_cache[key] = mask.astype(np.float32) / 255.0
        
        # 캐시 사이즈 제한
        if len(_mask_cache) > 20:
            del _mask_cache[next(iter(_mask_cache))]
    
    return _mask_cache[key]

# ============================================================
# 필터 메타 정보 및 스레드 워커 클래스
# ============================================================
FILTER_INFO = [
    {"name": "Retro",   "color": (200, 180, 60)},  # Aquarium에서 Retro로 이름 변경
    {"name": "Halo",    "color": (255, 220, 80)},
    {"name": "Heart",   "color": (150, 100, 255)},
    {"name": "Mosaic",  "color": (100, 200, 255)},
]
FILTER_FUNCS = [filter_retro, filter_halo, filter_heart, filter_mosaic]


class FilterWorker(threading.Thread):
    def __init__(self, filter_id: int):
        super().__init__(daemon=True)
        self.filter_id    = filter_id
        self.input_queue  = Queue(maxsize=1)
        self.output_queue = Queue(maxsize=1)

    def run(self):
        while True:
            try:
                frame, bboxes = self.input_queue.get(timeout=1.0)
                processed   = FILTER_FUNCS[self.filter_id](frame, bboxes)
                self._put(processed)
            except Empty:
                continue
            except Exception as e:
                print(f"[Worker {self.filter_id}] 오류: {e}")

    def _put(self, frame):
        if self.output_queue.full():
            try:
                self.output_queue.get_nowait()
            except Empty:
                pass
        self.output_queue.put(frame)


# ============================================================
# UI 렌더링 헬퍼 함수
# ============================================================
def draw_label(img, text, pos, color, font_scale=0.6, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    pad  = 4
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), (20, 20, 20), -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_face_box(img, bbox, color=(0, 255, 180), thickness=2):
    if bbox is None:
        return
    x, y, w, h = bbox
    L = min(w, h) // 5
    for (c, p1, p2) in [
        ((x, y),     (x+L, y),     (x, y+L)),
        ((x+w, y),   (x+w-L, y),   (x+w, y+L)),
        ((x, y+h),   (x+L, y+h),   (x, y+h-L)),
        ((x+w, y+h), (x+w-L, y+h), (x+w, y+h-L)),
    ]:
        cv2.line(img, c, p1, color, thickness, cv2.LINE_AA)
        cv2.line(img, c, p2, color, thickness, cv2.LINE_AA)


# ============================================================
# 비동기 얼굴 검출 워커 스레드 (MediaPipe 0.10.x Tasks API)
# ============================================================
class FaceDetectorWorker(threading.Thread):
    def __init__(self, model_path, w, h):
        super().__init__(daemon=True)
        self.model_path     = model_path
        self.W, self.H      = w, h
        self.input_queue    = Queue(maxsize=1)
        self.bboxes         = []
        self._result_lock   = threading.Lock()

    def run(self):
        # MediaPipe는 스레드 안전하지 않으므로 워커 스레드 내부에서 생성
        detector = create_mediapipe_detector(self.model_path, min_confidence=0.5)
        while True:
            try:
                frame = self.input_queue.get(timeout=1.0)
                # Tasks API는 mp.Image(BGR 그대로 전달 가능, format 지정)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )
                result = detector.detect(mp_image)
                bboxes = []
                for det in result.detections:
                    box = det.bounding_box   # mediapipe.tasks BoundingBox (절대 픽셀)
                    fx, fy = box.origin_x, box.origin_y
                    fw, fh = box.width,    box.height
                    px, py = int(fw * 0.15), int(fh * 0.15)
                    bx = max(0, fx - px)
                    by = max(0, fy - py)
                    bboxes.append((
                        bx, by,
                        min(self.W - bx, fw + px * 2),
                        min(self.H - by, fh + py * 2),
                    ))
                with self._result_lock:
                    self.bboxes = bboxes
            except Empty:
                continue
            except Exception as e:
                print(f"[FaceDetector] 오류: {e}")

    def submit(self, frame):
        if self.input_queue.empty():
            self.input_queue.put(frame)

    def get_bbox(self):
        with self._result_lock:
            return self.bboxes[0] if self.bboxes else None

    def get_bboxes(self):
        with self._result_lock:
            return list(self.bboxes)


# ============================================================
# 5. 메인 루프
# ============================================================
def main():
    print("=" * 55)
    print(" 🎭 AI 실시간 4분할 필터 포토부스 [MediaPipe 버전]")
    print("=" * 55)
    print(" [Space] : 화면 순차 캡처 (1→2→3→4)")
    print(" [R] : 캡처 초기화")
    print(" [S] : 4분할 화면 저장")
    print(" [Q/ESC] : 종료")
    print("=" * 55)

    model_path = download_mediapipe_model()

    webcam = WebcamStream(src=0).start()
    time.sleep(0.8)

    init_frame = webcam.read()
    if init_frame is None:
        print("❌ 카메라를 열 수 없습니다.")
        return

    H, W   = init_frame.shape[:2]
    HALF_W = W // 2
    HALF_H = H // 2

    face_worker = FaceDetectorWorker(model_path, W, H)
    face_worker.start()
    print(f"✅ MediaPipe FaceDetection 로드 완료 (입력 해상도: {W}x{H})")

    workers = [FilterWorker(i) for i in range(4)]
    for wk in workers:
        wk.start()

    freeze_step    = 0
    frozen_frames  = [None] * 4
    latest_results = [None] * 4
    prev_time       = time.time()

    while True:
        frame = webcam.read()
        if frame is None:
            continue

        frame = cv2.flip(frame, 1)

        # ── 얼굴 검출 워커에 원본 프레임 즉시 전송 (중복 가우스 블러 제거) ──
        face_worker.submit(frame)
        bboxes = face_worker.get_bboxes()

        # ── 각 필터 워커에 프레임 분배 및 결과 획득 ──
        for i, wk in enumerate(workers):
            if i >= freeze_step and wk.input_queue.empty():
                wk.input_queue.put((frame.copy(), bboxes))
            try:
                latest_results[i] = wk.output_queue.get_nowait()
            except Empty:
                if latest_results[i] is None:
                    latest_results[i] = frame.copy()

        # ── 4분할 그리드 화면 구성 ──
        grid_cells = []
        for i in range(4):
            src  = frozen_frames[i] if i < freeze_step else latest_results[i]
            if src is None:
                src = frame

            cell = cv2.resize(src, (HALF_W, HALF_H))

            is_frozen = i < freeze_step
            label = f"[{i+1}] {FILTER_INFO[i]['name']} {'FREEZE' if is_frozen else 'LIVE'}"
            color = (80, 80, 255) if is_frozen else FILTER_INFO[i]["color"]
            draw_label(cell, label, (8, 24), color)

            if is_frozen:
                cv2.rectangle(cell, (0, 0), (HALF_W-1, HALF_H-1), (80, 80, 255), 3)

            grid_cells.append(cell)

        top    = np.hstack((grid_cells[0], grid_cells[1]))
        bottom = np.hstack((grid_cells[2], grid_cells[3]))
        canvas = np.vstack((top, bottom))

        # ── 하단 UI 정보 출력 (FPS 및 가독성 개선) ──
        curr_time = time.time()
        dt = curr_time - prev_time
        fps = 1.0 / dt if dt > 0 else 0.0
        prev_time = curr_time
        draw_label(canvas, f"FPS:{fps:.0f}", (8, H - 10), (0, 255, 200))

        cap_text = f"CAPTURED: {freeze_step}/4"
        if freeze_step == 4:
            cap_text += " COMPLETE! [S]save [R]reset"
        draw_label(canvas, cap_text, (W // 2 - 140, H - 10), (255, 220, 80))

        cv2.imshow("AI 4-Split Filter Booth", canvas)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if freeze_step < 4 and latest_results[freeze_step] is not None:
                frozen_frames[freeze_step] = latest_results[freeze_step].copy()
                freeze_step += 1
                print(f" 📸 [{freeze_step}/4] 캡처 - {FILTER_INFO[freeze_step-1]['name']}")
                if freeze_step == 4:
                    print(" ✨ 완료! [S]저장 [R]리셋")
        elif key in (ord('r'), ord('R')):
            freeze_step, frozen_frames = 0, [None] * 4
            print(" 🔄 리셋")
        elif key in (ord('s'), ord('S')):
            fn = f"output_{int(time.time())}.png"
            cv2.imwrite(fn, canvas)
            print(f" 💾 저장 완료: {fn}")
        elif key in (ord('q'), ord('Q'), 27):
            print(" 👋 포토부스를 종료합니다.")
            break

    webcam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()