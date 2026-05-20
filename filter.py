import cv2
import time
import numpy as np

# Haar Cascade 얼굴 검출기 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 웹캠 시작
cap = cv2.VideoCapture(0)

# FPS 계산을 위한 변수 초기화
prev_time = 0

# 필터 상태 변수
is_heart = False
is_mosaic = False

window_name = "Face Filters Sandbox"
cv2.namedWindow(window_name)

# 하트 이미지 로드
heart_img = cv2.imread("heart.png", cv2.IMREAD_UNCHANGED)

if heart_img is None:
    print("경고: 'heart.png' 이미지를 불러올 수 없습니다. 경로를 확인하세요.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("웹캠을 열 수 없습니다.")
        break

    # FPS 계산
    current_time = time.time()
    fps = 0 if prev_time == 0 else 1 / (current_time - prev_time)
    prev_time = current_time

    # 얼굴 인식을 위한 흑백 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 위치 검출
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # 검출된 얼굴 처리
    for (x, y, w, h) in faces:
        if is_heart and heart_img is not None:
            # 1. 크기 조절 (얼굴 가로폭의 1.3배)
            nw = int(w * 1.3)
            nh = int(heart_img.shape[0] * (nw / heart_img.shape[1]))
            resized_heart = cv2.resize(heart_img, (nw, nh))
            
            # --- [해결 핵심] 핑크색만 추출하여 격자무늬 완벽 제거 ---
            # 색상 구별이 명확한 HSV 색공간으로 변환
            heart_hsv = cv2.cvtColor(resized_heart[:, :, :3], cv2.COLOR_BGR2HSV)
            
            # heart.png의 진한 핑크부터 연한 핑크까지 다 잡는 범위 설정
            lower_pink1 = np.array([140, 30, 50])
            upper_pink1 = np.array([180, 255, 255])
            
            # HSV에서 핑크색 영역만 255(흰색=살릴 곳), 격자는 0(검은색=지울 곳)으로 추출
            pink_mask = cv2.inRange(heart_hsv, lower_pink1, upper_pink1)
            
            # 마스크 테두리를 부드럽게 가다듬고 0.0 ~ 1.0 비율로 변환
            alpha_channel = cv2.GaussianBlur(pink_mask, (3, 3), 0) / 255.0
            custom_alpha_mask = cv2.merge([alpha_channel, alpha_channel, alpha_channel])
            
            heart_bgr_raw = resized_heart[:, :, :3]
            # -----------------------------------------------------

            # 2. 하트 위치 조정 (y축으로 85% 올려서 머리 위에 안착)
            nx = x - int((nw - w) / 2)
            ny = y - int(nh * 0.85)
            
            # 화면 경계를 벗어나는 경우 예외 처리
            x1, x2 = max(0, nx), min(frame.shape[1], nx + nw)
            y1, y2 = max(0, ny), min(frame.shape[0], ny + nh)
            
            if (x2 - x1) > 0 and (y2 - y1) > 0:
                roi = frame[y1:y2, x1:x2]
                
                # 잘려나간 화면 경계에 맞게 마스크와 이미지 동기화 슬라이싱
                cropped_alpha_mask = custom_alpha_mask[0:y2-y1, 0:x2-x1]
                cropped_heart_bgr = heart_bgr_raw[0:y2-y1, 0:x2-x1]
                
                # 3. 알파 블렌딩 합성
                composite = (cropped_heart_bgr * cropped_alpha_mask) + (roi * (1.0 - cropped_alpha_mask))
                
                # 5. 원본 프레임에 최종 적용
                frame[y1:y2, x1:x2] = composite.astype('uint8')
                
        elif is_mosaic:
            # 모자이크 처리 (유지)
            roi = frame[y:y+h, x:x+w]
            mosaic_w, mosaic_h = max(1, w // 15), max(1, h // 15)
            small = cv2.resize(roi, (mosaic_w, mosaic_h), interpolation=cv2.INTER_LINEAR)
            mosaic_roi = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            frame[y:y+h, x:x+w] = mosaic_roi

    # FPS 출력
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 화면에 출력
    cv2.imshow(window_name, frame)

    # 키 입력 처리 (1ms 대기)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('h'):
        is_heart = not is_heart
        is_mosaic = False
    elif key == ord('m'):
        is_mosaic = not is_mosaic
        is_heart = False

cap.release()
cv2.destroyAllWindows()