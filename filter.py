import cv2
import numpy as np
import threading
import time
import os
import urllib.request
from queue import Queue, Empty

# ============================================================
# 0. YuNet 모델 자동 다운로드
# ============================================================
def download_yunet_model():
    model_dir = "models"
    model_path = os.path.join(model_dir, "face_detection_yunet_2023mar.onnx")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 100_000:
        if os.path.exists(model_path):
            os.remove(model_path)
        print("📥 YuNet 모델 다운로드 중...")
        url = (
            "https://raw.githubusercontent.com/opencv/opencv_zoo/"
            "9871e4a1796191ef64b63897103138b72da9bf74/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req) as resp, open(model_path, "wb") as f:
                f.write(resp.read())
            size = os.path.getsize(model_path)
            print(f"✅ 다운로드 완료 ({size:,} bytes)")
            if size < 100_000:
                raise ValueError("파일이 너무 작습니다.")
        except Exception as e:
            print(f"❌ 다운로드 실패: {e}")
    return model_path


# ============================================================
# ✅ 최적화 1: 해상도 축소 (1280x720 → 960x540)
#    필터 4개 동시 처리 기준으로 픽셀 수 44% 감소
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
        # ✅ 최적화 2: 버퍼 크기 1로 고정 → 항상 최신 프레임만 유지
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
# ✅ 최적화 3: overlay_transparent - 벡터화 연산으로 교체
#    기존 Python for 루프(채널별) → numpy 브로드캐스팅 1번
# ============================================================
def overlay_transparent(background, overlay, x, y):
    bg_h, bg_w = background.shape[:2]
    if overlay.ndim == 2 or overlay.shape[2] == 3:
        return background  # 알파 없으면 스킵

    h, w = overlay.shape[:2]
    x1, x2 = max(0, x), min(bg_w, x + w)
    y1, y2 = max(0, y), min(bg_h, y + h)
    ox1 = max(0, -x);  ox2 = ox1 + (x2 - x1)
    oy1 = max(0, -y);  oy2 = oy1 + (y2 - y1)

    if x1 >= x2 or y1 >= y2:
        return background

    alpha = overlay[oy1:oy2, ox1:ox2, 3:4].astype(np.float32) / 255.0  # (h,w,1)
    fg    = overlay[oy1:oy2, ox1:ox2, :3].astype(np.float32)
    bg    = background[y1:y2, x1:x2].astype(np.float32)

    blended = (alpha * fg + (1.0 - alpha) * bg).astype(np.uint8)
    background[y1:y2, x1:x2] = blended
    return background


# ============================================================
# 에셋 로드 (흰 배경 자동 제거)
# ============================================================
def load_asset_remove_bg(path, bg_thresh=240):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.shape[2] == 3:
        img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_bgra[:, :, 3][gray > bg_thresh] = 0
        return img_bgra
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    img[:, :, 3][(gray > bg_thresh) & (img[:, :, 3] > 200)] = 0
    return img


HEART_IMG = load_asset_remove_bg("heart.png")
if HEART_IMG is None:
    print("⚠️ 'heart.png' 없음 → 기본 하트 도형으로 대체")

# ✅ 하트 이미지 리사이즈 캐시 (매 프레임 resize 방지)
_heart_cache: dict = {}

def get_resized_heart(target_w: int):
    """target_w 기준으로 캐시된 하트 이미지 반환"""
    if target_w not in _heart_cache:
        if HEART_IMG is None:
            return None
        ratio = HEART_IMG.shape[0] / HEART_IMG.shape[1]
        th = max(int(target_w * ratio), 1)
        _heart_cache[target_w] = cv2.resize(HEART_IMG, (target_w, th))
        # 캐시 최대 10개 유지
        if len(_heart_cache) > 10:
            _heart_cache.pop(next(iter(_heart_cache)))
    return _heart_cache[target_w]


# ============================================================
# ✅ 최적화 4: 필터별 연산량 대폭 감소
# ============================================================

def filter_sketch(frame: np.ndarray, bbox) -> np.ndarray:
    """스케치: GaussianBlur 커널 축소 (21→11), 얼굴 ROI만 Laplacian"""
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (11, 11), 0)   # ✅ 21→11
    sketch = cv2.divide(gray, blur, scale=256.0)
    result = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    if bbox:
        x, y, w, h = bbox
        roi      = result[y:y+h, x:x+w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges    = cv2.Laplacian(roi_gray, cv2.CV_8U, ksize=3)
        _, mask  = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)
        mask3    = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        result[y:y+h, x:x+w] = cv2.subtract(roi, mask3 // 2)
    return result


def filter_cartoon(frame: np.ndarray, bbox) -> np.ndarray:
    """
    ✅ 최적화: stylization 제거 + bilateralFilter 8회→2회
       전체 프레임 대신 다운스케일 후 처리
    """
    # 절반 크기로 줄여서 처리 후 원본 크기로 복원
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 2, h // 2))

    # bilateralFilter 2회 (기존 8회)
    cartoon = cv2.bilateralFilter(small, 7, 50, 50)
    cartoon = cv2.bilateralFilter(cartoon, 7, 50, 50)

    result = cv2.resize(cartoon, (w, h))

    if bbox:
        x, y, bw, bh = bbox
        roi      = frame[y:y+bh, x:x+bw]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_blur = cv2.medianBlur(roi_gray, 5)
        edges    = cv2.adaptiveThreshold(
            roi_blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
            blockSize=9, C=9
        )
        edge3 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # 얼굴 ROI만 bilateralFilter 2회
        face_small = cv2.resize(roi, (bw // 2, bh // 2))
        face_small = cv2.bilateralFilter(face_small, 7, 50, 50)
        face_small = cv2.bilateralFilter(face_small, 7, 50, 50)
        face_cartoon = cv2.resize(face_small, (bw, bh))

        result[y:y+bh, x:x+bw] = cv2.bitwise_and(face_cartoon, edge3)
    return result


def filter_heart(frame: np.ndarray, bbox) -> np.ndarray:
    """하트: 캐시된 리사이즈 이미지 사용"""
    result = frame.copy()
    if not bbox:
        return result
    x, y, w, h = bbox

    hw = max(int(w * 0.7), 40)
    heart = get_resized_heart(hw)

    y_off = int(np.sin(time.time() * 3.5) * 8)

    if heart is not None:
        hx = x + w // 2 - hw // 2
        hy = y - heart.shape[0] - 5 + y_off
        result = overlay_transparent(result, heart, hx, hy)
    else:
        # Fallback 하트 도형
        cx, cy = x + w // 2, y - 35 + y_off
        r = max(w // 8, 15)
        cv2.circle(result, (cx - r // 2, cy), r // 2, (0, 0, 220), -1)
        cv2.circle(result, (cx + r // 2, cy), r // 2, (0, 0, 220), -1)
        pts = np.array([[cx - r, cy], [cx + r, cy], [cx, cy + r + 4]], np.int32)
        cv2.fillPoly(result, [pts], (0, 0, 220))
    return result


def filter_mosaic(frame: np.ndarray, bbox) -> np.ndarray:
    """모자이크: 마스크 블러 제거하고 단순 픽셀화로 경량화"""
    result = frame.copy()
    if not bbox:
        return result
    x, y, w, h = bbox
    roi = result[y:y+h, x:x+w]
    if roi.size == 0:
        return result

    # 픽셀화 (블록 크기: 얼굴 너비 기준)
    bk = max(w // 10, 8)
    sw, sh = max(w // bk, 2), max(h // bk, 2)
    small     = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h),  interpolation=cv2.INTER_NEAREST)
    result[y:y+h, x:x+w] = pixelated
    return result


FILTER_INFO = [
    {"name": "Sketch",  "color": (220, 220, 220)},
    {"name": "Cartoon", "color": (80,  200, 120)},
    {"name": "Heart",   "color": (150, 100, 255)},
    {"name": "⬛ Mosaic",  "color": (100, 200, 255)},
]
FILTER_FUNCS = [filter_sketch, filter_cartoon, filter_heart, filter_mosaic]


# ============================================================
# 3. 필터 워커
# ============================================================
class FilterWorker(threading.Thread):
    def __init__(self, filter_id: int):
        super().__init__(daemon=True)
        self.filter_id    = filter_id
        self.input_queue  = Queue(maxsize=1)
        self.output_queue = Queue(maxsize=1)

    def run(self):
        while True:
            try:
                frame, bbox = self.input_queue.get(timeout=1.0)
                processed   = FILTER_FUNCS[self.filter_id](frame, bbox)
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
# 4. UI 렌더링 헬퍼
# ============================================================
def draw_label(img, text, pos, color, font_scale=0.6, thickness=1):
    """✅ 최적화: img.copy() 제거 → putText 직접 + 간단한 배경 사각형"""
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
# ✅ 최적화 5: 얼굴 검출 전용 스레드 (매 프레임 검출 → 비동기)
#    메인 루프에서 detector.detect() 호출 제거 → FPS 향상
# ============================================================
class FaceDetectorWorker(threading.Thread):
    def __init__(self, model_path, w, h):
        super().__init__(daemon=True)
        self._detector = cv2.FaceDetectorYN.create(
            model=model_path, config="",
            input_size=(w, h),
            score_threshold=0.55, nms_threshold=0.3,
        )
        self.W, self.H      = w, h
        self.input_queue    = Queue(maxsize=1)
        self.bbox           = None          # 최신 검출 결과 (공유)
        self._result_lock   = threading.Lock()

    def run(self):
        while True:
            try:
                frame = self.input_queue.get(timeout=1.0)
                _, faces = self._detector.detect(frame)
                bbox = None
                if faces is not None and len(faces) > 0:
                    fx, fy, fw, fh = map(int, faces[0][:4])
                    px, py = int(fw * 0.15), int(fh * 0.15)
                    bx = max(0, fx - px)
                    by = max(0, fy - py)
                    bbox = (bx, by,
                            min(self.W - bx, fw + px * 2),
                            min(self.H - by, fh + py * 2))
                with self._result_lock:
                    self.bbox = bbox
            except Empty:
                continue

    def submit(self, frame):
        if self.input_queue.empty():
            self.input_queue.put(frame)

    def get_bbox(self):
        with self._result_lock:
            return self.bbox


# ============================================================
# 5. 메인 루프
# ============================================================
def main():
    print("=" * 55)
    print("  🎭 AI 실시간 4분할 필터 포토부스  [최적화 버전]")
    print("=" * 55)
    print("  [Space] : 화면 순차 캡처 (1→2→3→4)")
    print("  [R]     : 캡처 초기화")
    print("  [S]     : 4분할 화면 저장")
    print("  [Q/ESC] : 종료")
    print("=" * 55)

    yunet_path = download_yunet_model()

    webcam = WebcamStream(src=0).start()
    time.sleep(0.8)

    init_frame = webcam.read()
    if init_frame is None:
        print("❌ 카메라를 열 수 없습니다.")
        return

    H, W   = init_frame.shape[:2]
    HALF_W = W // 2
    HALF_H = H // 2

    # 얼굴 검출 전용 워커 (비동기)
    face_worker = FaceDetectorWorker(yunet_path, W, H)
    face_worker.start()
    print(f"✅ YuNet 로드 완료 (입력 해상도: {W}x{H})")

    # 필터 워커 4개
    workers = [FilterWorker(i) for i in range(4)]
    for wk in workers:
        wk.start()

    freeze_step    = 0
    frozen_frames  = [None] * 4
    latest_results = [None] * 4
    prev_time      = time.time()

    while True:
        frame = webcam.read()
        if frame is None:
            continue

        frame = cv2.flip(frame, 1)

        # ── 얼굴 검출: 비동기 워커에 제출 + 이전 결과 사용 ──
        face_worker.submit(frame)
        bbox = face_worker.get_bbox()

        # ── 워커에 프레임 분배 + 결과 수집 ──────────────────
        for i, wk in enumerate(workers):
            if i >= freeze_step and wk.input_queue.empty():
                wk.input_queue.put((frame.copy(), bbox))
            try:
                latest_results[i] = wk.output_queue.get_nowait()
            except Empty:
                if latest_results[i] is None:
                    latest_results[i] = frame.copy()

        # ── 4분할 그리드 구성 ────────────────────────────────
        grid_cells = []
        for i in range(4):
            src  = frozen_frames[i] if i < freeze_step else latest_results[i]
            if src is None:
                src = frame

            cell = cv2.resize(src, (HALF_W, HALF_H))

            if i >= freeze_step and bbox and i < 2:
                sb = (bbox[0]//2, bbox[1]//2, bbox[2]//2, bbox[3]//2)
                draw_face_box(cell, sb)

            is_frozen = i < freeze_step
            label = f"[{i+1}] {FILTER_INFO[i]['name']}  {'FREEZE' if is_frozen else 'LIVE'}"
            color = (80, 80, 255) if is_frozen else FILTER_INFO[i]["color"]
            draw_label(cell, label, (8, 24), color)

            if is_frozen:
                cv2.rectangle(cell, (0, 0), (HALF_W-1, HALF_H-1), (80, 80, 255), 3)

            grid_cells.append(cell)

        top    = np.hstack((grid_cells[0], grid_cells[1]))
        bottom = np.hstack((grid_cells[2], grid_cells[3]))
        canvas = np.vstack((top, bottom))

        # ── FPS / 상태 표시 ──────────────────────────────────
        curr_time = time.time()
        fps       = 1.0 / max(curr_time - prev_time, 1e-5)
        prev_time = curr_time
        draw_label(canvas, f"FPS:{fps:.0f}", (8, H - 10), (0, 255, 200))

        cap_text = f"CAPTURED: {freeze_step}/4"
        if freeze_step == 4:
            cap_text += "  COMPLETE! [S]save [R]reset"
        draw_label(canvas, cap_text, (W // 2 - 140, H - 10), (255, 220, 80))

        cv2.imshow("AI 4-Split Filter Booth", canvas)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if freeze_step < 4 and latest_results[freeze_step] is not None:
                frozen_frames[freeze_step] = latest_results[freeze_step].copy()
                freeze_step += 1
                print(f"  📸 [{freeze_step}/4] 캡처 - {FILTER_INFO[freeze_step-1]['name']}")
                if freeze_step == 4:
                    print("  ✨ 완료! [S]저장 [R]리셋")
        elif key in (ord('r'), ord('R')):
            freeze_step, frozen_frames = 0, [None] * 4
            print("  🔄 리셋")
        elif key in (ord('s'), ord('S')):
            fn = f"output_{int(time.time())}.png"
            cv2.imwrite(fn, canvas)
            print(f"  💾 저장: {fn}")
        elif key in (ord('q'), ord('Q'), 27):
            print("  👋 종료")
            break

    webcam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()