import cv2
import numpy as np

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 창 닫힘 이벤트를 위한 창 이름 설정
window_name = "3-Split Photo Booth"
cv2.namedWindow(window_name)

while True:
    ret, frame = cap.read()
    if not ret:
        print("웹캠을 열 수 없습니다.")
        break

    # 1. 프레임 크기 조절 (3개를 가로로 붙이면 화면이 너무 커지므로 각각 축소)
    # 가로 400, 세로 300으로 조절 (필요에 따라 변경 가능)
    frame = cv2.resize(frame, (400, 300))

    # --- [1번 화면: 원본] ---
    frame1 = frame.copy()

    # --- [2번 화면: 흑백 필터] ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 병합(hconcat)을 하려면 모든 이미지의 채널 수(색상 구조)가 같아야 함
    # 흑백(1채널)을 다시 컬러 구조(3채널)로 변환해줌 (시각적으로는 흑백 유지)
    frame2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # --- [3번 화면: 임시 블러(흐림) 필터] ---
    # 나중에 여기에 다른 효과를 추가하면 됨
    frame3 = cv2.GaussianBlur(frame, (25, 25), 0)

    # 2. 3개의 화면을 가로로 이어 붙이기 (Horizontal Concatenation)
    combined_frame = cv2.hconcat([frame1, frame2, frame3])

    # 3. 각 화면 위에 텍스트로 이름 표시 (구분을 위해)
    # 400픽셀마다 새로운 화면이 시작되므로 x좌표를 10, 410, 810으로 설정
    cv2.putText(combined_frame, "1. Original", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(combined_frame, "2. Grayscale", (410, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(combined_frame, "3. Blur (Temp)", (810, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 결과 화면 출력
    cv2.imshow(window_name, combined_frame)

    # 키보드 'q'를 누르거나 마우스로 창을 닫으면 종료
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()