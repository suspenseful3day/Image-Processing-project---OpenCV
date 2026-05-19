import cv2
import time

# XML 파일 경로 확인
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
print("Cascade Path:", cascade_path)

# 얼굴 검출기 로드
face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml'
)
# 로드 확인
if face_cascade.empty():
    print("얼굴 검출기 XML 파일을 불러오지 못했습니다.")
    exit()

# 웹캠 열기
cap = cv2.VideoCapture(0)
prev_time = 0

# 상태 저장 변수 (False: 원본 화면, True: 흑백 필터)
# 토글(반복)을 위해 참/거짓(boolean) 값으로 변경했습니다.
is_grayscale = False

# 창 닫힘 이벤트를 감지하기 위해 창 이름을 미리 지정하여 생성합니다.
window_name = "Face Detection & Filters"
cv2.namedWindow(window_name)

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

    # -----------------------------------------------------------
    # [참고] 초록색 사각형 그리는 코드는 삭제했지만, 
    # 나중에 2번, 3번 효과에서 얼굴 위치에 무언가(예: 안경, 고양이 귀)를 
    # 합성하려면 얼굴 좌표(x, y, w, h)가 필요하므로 검출 기능 자체는 살려두었습니다.
    # -----------------------------------------------------------
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # === [필터 적용 로직] ===
    if is_grayscale:
        # 흑백 모드 켜짐
        display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        # 원본 모드
        display_frame = frame.copy()

    # === [화면 그리기] ===
    # (초록색 사각형 제거됨)

    # FPS 텍스트 그리기
    cv2.putText(display_frame, f"FPS: {int(fps)}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 결과 화면 출력
    cv2.imshow(window_name, display_frame)

    # === [이벤트 처리] ===
    key = cv2.waitKey(1) & 0xFF
    
    # 1. 키보드 입력 ('q' 종료, '1' 토글)
    if key == ord('q'):      
        break
    elif key == ord('1'):    
        # 1번을 누를 때마다 상태가 반전됩니다. (True -> False -> True)
        is_grayscale = not is_grayscale
        print(f">> 필터 1번: {'흑백 모드' if is_grayscale else '원본 모드'} 켜짐")

    # 2. 마우스로 창의 'X' 닫기 버튼을 눌렀을 때 처리
    # 창의 속성(가시성)을 확인하여 1보다 작으면(창이 닫혔으면) 루프를 탈출합니다.
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()