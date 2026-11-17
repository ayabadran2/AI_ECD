import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')


video_path = "people-detection.mp4" 
cap = cv2.VideoCapture(video_path)


if not cap.isOpened():
    print("Error: Could not open video. Check the file name and path.")
    exit()

CROWD_THRESHOLD = 3 #temp num

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video finished or error reading frame.")
        break

    results = model(frame, verbose=False, conf=0.5)
    person_count = 0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            if int(box.cls[0]) == 0: # 0 is 'person'
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display count
    color = (0, 0, 255) if person_count >= CROWD_THRESHOLD else (0, 255, 0)
    cv2.putText(frame, f'People: {person_count}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('Crowd Intelligence System', frame)

   
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()