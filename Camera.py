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

while True:
    ret, frame = cap.read()
    if not ret:
        print("웹캠을 열 수 없습니다.")
        break

    current_time = time.time()
    fps = 0 if prev_time == 0 else 1 / (current_time - prev_time)
    prev_time = current_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()