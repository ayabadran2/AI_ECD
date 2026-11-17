import cv2
import os
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


print("Downloading pre-trained Fire/Smoke model...")
model_path = hf_hub_download(repo_id="rabahdev/fire-smoke-yolov8n", filename="best.pt")
print(f"Model loaded from: {model_path}")


model = YOLO(model_path)


#video_path = "D:\A.Pin\AI_ECD\dataset_fire_smoke\fire_data\test" 
video_path = r"D:\A.Pin\AI_ECD\test1.mp4"
cap = cv2.VideoCapture(video_path)

ALERTS_FOLDER = "fire_alerts"
os.makedirs(ALERTS_FOLDER, exist_ok=True)

alert_triggered = False # Spam prevention flag

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

  
    results = model(frame, verbose=False, conf=0.4)
    fire_or_smoke_detected = False

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls] # Will be 'smoke' or 'fire'
            
            fire_or_smoke_detected = True
            
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'{class_name.upper()} {conf:.2f}', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    
    if fire_or_smoke_detected and not alert_triggered:
        alert_triggered = True
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"fire_alert_{timestamp}.jpg"
        filepath = os.path.join(ALERTS_FOLDER, filename)
        
        cv2.imwrite(filepath, frame)
        print(f"EMERGENCY! Fire/Smoke detected. Snapshot saved: {filepath}")
        
        # TODO: Add backend API call 
        # send_alert_to_backend(event_type=class_name, image_path=filepath)

    elif not fire_or_smoke_detected:
        alert_triggered = False

    # Display vid
    status_text = "FIRE/SMOKE DETECTED!" if alert_triggered else "Monitoring..."
    status_color = (0, 0, 255) if alert_triggered else (0, 255, 0)
    
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.imshow('Fire & Smoke Detection System', frame)

    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()