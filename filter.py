import cv2
import numpy as np
import threading
import time
import os
import urllib.request
from queue import Queue, Empty

# ============================================================
# 0. YuNet 모델 자동 다운로드 (이전 구조 유지)
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
# ✅ 해상도 축소 유지 (960x540)
# ============================================================
CAM_W, CAM_H = 960, 540


# ============================================================
# 1. 비동기 웹캠 스트림 (구조 유지)
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
# 하트 이미지 로드 및 캐싱 (구 버전의 핵심 알고리즘 반영)
# ============================================================
# 격자무늬 제거를 위해 알파 채널을 무시하고 3채널(BGR) 이미지로만 읽습니다.
HEART_IMG = cv2.imread("heart.png", cv2.IMREAD_COLOR)

if HEART_IMG is None:
    print("⚠️ 경고: 'heart.png' 이미지를 불러올 수 없습니다. 경로를 확인하세요.")
else:
    print("✅ 'heart.png' 로드 완료 (구 버전 방식 마스크 처리 준비)")

# 리사이즈 이미지 및 구 버전 마스크용 캐시
_heart_cache: dict = {}

def get_resized_heart_with_mask(target_w: int):
    """구 버전의 핑크색 추출 마스크 알고리즘을 적용한 하트 및 알파 마스크 반환"""
    if target_w not in _heart_cache:
        if HEART_IMG is None:
            return None, None
        
        # 구 버전의 비율 맞춤형 리사이즈 로직
        ratio = HEART_IMG.shape[0] / HEART_IMG.shape[1]
        th = max(int(target_w * ratio), 1)
        resized_heart = cv2.resize(HEART_IMG, (target_w, th))
        
        # --- 🌟 [구 버전 핵심 알고리즘] 핑크색만 추출하여 격자무늬 완벽 제거 ---
        heart_hsv = cv2.cvtColor(resized_heart, cv2.COLOR_BGR2HSV)
        
        # 진한 핑크부터 연한 핑크까지 가두는 범위
        lower_pink = np.array([140, 30, 50])
        upper_pink = np.array([180, 255, 255])
        
        # 핑크만 255(흰색), 격자무늬 배경은 0(검은색) 마스크 생성
        pink_mask = cv2.inRange(heart_hsv, lower_pink, upper_pink)
        
        # 테두리 가다듬고 float32 정규화 (0.0 ~ 1.0)
        alpha_channel = cv2.GaussianBlur(pink_mask, (3, 3), 0) / 255.0
        custom_alpha_mask = cv2.merge([alpha_channel, alpha_channel, alpha_channel])
        # ---------------------------------------------------------------------
        
        _heart_cache[target_w] = (resized_heart, custom_alpha_mask)
        if len(_heart_cache) > 10:
            _heart_cache.pop(next(iter(_heart_cache)))
            
    return _heart_cache[target_w]


# ============================================================
# 2. 필터 함수 (구조 유지, 하트만 완벽 수정)
# ============================================================

def filter_sketch(frame: np.ndarray, bbox) -> np.ndarray:
    """스케치 필터 (이전 구조 유지)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    sketch = cv2.divide(gray, blur, scale=256.0)
    result = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    if bbox:
        x, y, w, h = bbox
        roi = result[y:y+h, x:x+w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Laplacian(roi_gray, cv2.CV_8U, ksize=3)
        _, mask = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)
        mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        result[y:y+h, x:x+w] = cv2.subtract(roi, mask3 // 2)
    return result


def filter_cartoon(frame: np.ndarray, bbox) -> np.ndarray:
    """만화 필터 (이전 구조 유지)"""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 2, h // 2))
    cartoon = cv2.bilateralFilter(small, 7, 50, 50)
    cartoon = cv2.bilateralFilter(cartoon, 7, 50, 50)
    result = cv2.resize(cartoon, (w, h))

    if bbox:
        x, y, bw, bh = bbox
        roi = frame[y:y+bh, x:x+bw]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_blur = cv2.medianBlur(roi_gray, 5)
        edges = cv2.adaptiveThreshold(
            roi_blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
            blockSize=9, C=9
        )
        edge3 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        face_small = cv2.resize(roi, (bw // 2, bh // 2))
        face_small = cv2.bilateralFilter(face_small, 7, 50, 50)
        face_small = cv2.bilateralFilter(face_small, 7, 50, 50)
        face_cartoon = cv2.resize(face_small, (bw, bh))
        result[y:y+bh, x:x+bw] = cv2.bitwise_and(face_cartoon, edge3)
    return result


def filter_heart(frame: np.ndarray, bbox) -> np.ndarray:
    """🎯 하트 필터: 구 버전의 핑크색 추출 마스크 공식을 현 구조에 완벽 이식"""
    result = frame.copy()
    if not bbox:
        return result
    x, y, w, h = bbox

    # 구 버전 비율 적용: 얼굴 가로폭의 1.3배 크기로 설정
    nw = int(w * 1.3)
    resized_heart, custom_alpha_mask = get_resized_heart_with_mask(nw)

    # 둥실둥실 움직이는 효과용 오프셋
    y_off = int(np.sin(time.time() * 3.5) * 8)

    if resized_heart is not None:
        nh = resized_heart.shape[0]
        
        # 구 버전 위치 선정 로직: 얼굴 가로 중앙 정렬 및 y축으로 85% 위로 안착
        nx = x - int((nw - w) / 2)
        ny = y - int(nh * 0.85) + y_off
        
        # 화면 경계를 벗어나는 상황에 대한 예외 처리 (슬라이싱 경계 계산)
        x1, x2 = max(0, nx), min(result.shape[1], nx + nw)
        y1, y2 = max(0, ny), min(result.shape[0], ny + nh)
        
        if (x2 - x1) > 0 and (y2 - y1) > 0:
            # 원본 화면의 ROI 추출
            roi = result[y1:y2, x1:x2].astype(np.float32)
            
            # 잘려나간 화면 경계 좌표에 맞게 하트 이미지와 마스크도 동기화하여 잘라냄
            # (구 버전의 [0:y2-y1, 0:x2-x1] 조절 방식을 유연하게 처리)
            hx1 = x1 - nx
            hx2 = hx1 + (x2 - x1)
            hy1 = y1 - ny
            hy2 = hy1 + (y2 - y1)
            
            cropped_heart_bgr = resized_heart[hy1:hy2, hx1:hx2].astype(np.float32)
            cropped_alpha_mask = custom_alpha_mask[hy1:hy2, hx1:hx2]
            
            # ✨ 구 버전 핵심 알파 블렌딩 합성 수식 그대로 적용
            composite = (cropped_heart_bgr * cropped_alpha_mask) + (roi * (1.0 - cropped_alpha_mask))
            
            # 원본 프레임 영역에 uint8 타입으로 안전하게 덮어쓰기
            result[y1:y2, x1:x2] = composite.astype('uint8')
    else:
        # Fallback 기본 하트 도형 그리기 로직
        cx, cy = x + w // 2, y - 35 + y_off
        r = max(w // 8, 15)
        cv2.circle(result, (cx - r // 2, cy), r // 2, (0, 0, 220), -1)
        cv2.circle(result, (cx + r // 2, cy), r // 2, (0, 0, 220), -1)
        pts = np.array([[cx - r, cy], [cx + r, cy], [cx, cy + r + 4]], np.int32)
        cv2.fillPoly(result, [pts], (0, 0, 220))
        
    return result


def filter_mosaic(frame: np.ndarray, bbox) -> np.ndarray:
    """모자이크 필터 (이전 구조 유지)"""
    result = frame.copy()
    if not bbox:
        return result
    x, y, w, h = bbox
    roi = result[y:y+h, x:x+w]
    if roi.size == 0:
        return result
    bk = max(w // 10, 8)
    sw, sh = max(w // bk, 2), max(h // bk, 2)
    small     = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h),  interpolation=cv2.INTER_NEAREST)
    result[y:y+h, x:x+w] = pixelated
    return result


# ============================================================
# 필터 메타 정보 및 스레드 워커 클래스 (구조 유지)
# ============================================================
FILTER_INFO = [
    {"name": "Sketch",  "color": (220, 220, 220)},
    {"name": "Cartoon", "color": (80,  200, 120)},
    {"name": "Heart",   "color": (150, 100, 255)},
    {"name": "⬛ Mosaic",  "color": (100, 200, 255)},
]
FILTER_FUNCS = [filter_sketch, filter_cartoon, filter_heart, filter_mosaic]


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
# UI 렌더링 헬퍼 함수 (구조 유지)
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
# 비동기 얼굴 검출 워커 스레드 (구조 유지)
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
        self.bbox           = None
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
# 5. 메인 루프 (구조 유지)
# ============================================================
def main():
    print("=" * 55)
    print(" 🎭 AI 실시간 4분할 필터 포토부스 [하트 격자 무늬 완벽 해결 버전]")
    print("=" * 55)
    print(" [Space] : 화면 순차 캡처 (1→2→3→4)")
    print(" [R] : 캡처 초기화")
    print(" [S] : 4분할 화면 저장")
    print(" [Q/ESC] : 종료")
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

    # 얼굴 검출 워커 시작
    face_worker = FaceDetectorWorker(yunet_path, W, H)
    face_worker.start()
    print(f"✅ YuNet 로드 완료 (입력 해상도: {W}x{H})")

    # 필터 처리 워커 4개 시작
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

        # ── 얼굴 검출 워커에 현재 프레임 전송 ──
        face_worker.submit(frame)
        bbox = face_worker.get_bbox()

        # ── 각 필터 워커에 프레임 분배 및 결과 획득 ──
        for i, wk in enumerate(workers):
            if i >= freeze_step and wk.input_queue.empty():
                wk.input_queue.put((frame.copy(), bbox))
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

            # 1, 2번 화면(Sketch, Cartoon)만 얼굴 초록색 가이드 박스 표시
            if i >= freeze_step and bbox and i < 2:
                sb = (bbox[0]//2, bbox[1]//2, bbox[2]//2, bbox[3]//2)
                draw_face_box(cell, sb)

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

        # ── 하단 UI 정보 출력 (FPS 및 캡처 개수) ──
        curr_time = time.time()
        fps       = 1.0 / max(curr_time - prev_time, 1e-5)
        prev_time = curr_time
        draw_label(canvas, f"FPS:{fps:.0f}", (8, H - 10), (0, 255, 200))

        cap_text = f"CAPTURED: {freeze_step}/4"
        if freeze_step == 4:
            cap_text += " COMPLETE! [S]save [R]reset"
        draw_label(canvas, cap_text, (W // 2 - 140, H - 10), (255, 220, 80))

        cv2.imshow("AI 4-Split Filter Booth", canvas)

        key = cv2.waitKey(1) & 0xFF

        # 키 제어 이벤트 처리
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