import argparse
import datetime
import math
import os
import time
from collections import defaultdict, deque
 
import cv2
import numpy as np
from ultralytics import YOLO
 
KP_CONF = 0.4          # minimum keypoint confidence to trust a joint
LYING_ANGLE = 60.0     # torso angle from vertical (degrees) that counts as lying
LYING_SECONDS = 1.0    # the person must stay down this long
DROP_WINDOW = 1.0      # the fast drop must happen within this many seconds
DROP_RATIO = 0.30      # hip descent as a fraction of the person's height
DROP_MEMORY = 3.0      # a drop counts if it happened within this many seconds
COOLDOWN = 10.0        # seconds between alerts for the same person
 
 
class Person:
    def __init__(self):
        self.hist = deque()        # (time, hip_y, box_height)
        self.lying_since = None
        self.last_drop = -1e9
        self.last_alert = -1e9
 
 
def mid(kp, conf, a, b):
    """Midpoint of two joints, using only the ones detected with enough confidence."""
    pts = [kp[i] for i in (a, b) if conf[i] >= KP_CONF]
    return np.mean(pts, axis=0) if pts else None
 
 
def analyse(kp, conf, box):
    """Return (lying, hip_y, box_height). lying is None when the pose is unreliable."""
    x1, y1, x2, y2 = box
    box_h = max(float(y2 - y1), 1.0)
    shoulders = mid(kp, conf, 5, 6)
    hips = mid(kp, conf, 11, 12)
    hip_y = float(hips[1]) if hips is not None else None
    if shoulders is None or hips is None:
        aspect = (x2 - x1) / box_h
        return (True if aspect > 1.5 else None), hip_y, box_h
    dx, dy = hips[0] - shoulders[0], hips[1] - shoulders[1]
    angle = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))   # 0 = upright, 90 = horizontal
    return angle > LYING_ANGLE, hip_y, box_h
 
 
def update(p, t, hip_y, box_h, lying):
    """Update one person's state. Returns True when a fall alert should fire."""
    if hip_y is not None:
        p.hist.append((t, hip_y, box_h))
    while p.hist and t - p.hist[0][0] > DROP_WINDOW:
        p.hist.popleft()
    if len(p.hist) >= 2:
        descent = p.hist[-1][1] - min(h[1] for h in p.hist)       # image y grows downward
        ref_height = max(h[2] for h in p.hist)                    # person height while still upright
        if descent / ref_height >= DROP_RATIO:
            p.last_drop = t
    if lying is None:                    # pose unreliable this frame: keep the state, don't reset it
        return False
    if lying:
        if p.lying_since is None:
            p.lying_since = t
    else:
        p.lying_since = None
    return (
        p.lying_since is not None
        and t - p.lying_since >= LYING_SECONDS
        and t - p.last_drop <= DROP_MEMORY
        and t - p.last_alert >= COOLDOWN
    )
 
 
def send_alert_to_backend(event):
    # Replace with a POST to your FastAPI backend, e.g. requests.post(URL, json=event)
    print("ALERT:", event)
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="06.mp4", help="video file path, or 0 for the webcam")
    ap.add_argument("--weights", default="yolov8n-pose.pt")
    ap.add_argument("--camera-id", default="cam1")
    ap.add_argument("--alerts", default="fall_alerts")
    ap.add_argument("--show", action="store_true", default=True, help="show a live window (needs a display)")
    args = ap.parse_args()
 
    os.makedirs(args.alerts, exist_ok=True)
    model = YOLO(args.weights)
    is_webcam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_webcam else args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    people = defaultdict(Person)
    frame_idx = 0
    t0 = time.monotonic()
 
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        t = (time.monotonic() - t0) if is_webcam else frame_idx / fps   # video time, not processing time
        frame_idx += 1
 
        r = model.track(frame, persist=True, conf=0.3, imgsz=640, verbose=False)[0]
        if r.boxes is not None and r.keypoints is not None and len(r.boxes) > 0:
            ids = r.boxes.id.int().tolist() if r.boxes.id is not None else list(range(len(r.boxes)))
            boxes = r.boxes.xyxy.cpu().numpy()
            kps = r.keypoints.xy.cpu().numpy()
            confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(kps.shape[:2])
            for pid, box, kp, cf in zip(ids, boxes, kps, confs):
                person = people[pid]
                lying, hip_y, box_h = analyse(kp, cf, box)
                fired = update(person, t, hip_y, box_h, lying)
                alerted = fired or (t - person.last_alert < COOLDOWN)
                x1, y1, x2, y2 = map(int, box)
                color = (0, 0, 255) if alerted else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"id{pid} {'FALL' if alerted else 'ok'}", (x1, max(y1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if fired:
                    person.last_alert = t
                    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    path = os.path.join(args.alerts, f"{args.camera_id}_{stamp}_id{pid}.jpg")
                    cv2.imwrite(path, frame)
                    send_alert_to_backend({
                        "event_type": "fall",
                        "camera_id": args.camera_id,
                        "video_time_s": round(t, 2),
                        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                        "severity": "high",
                        "snapshot": path,
                    })
 
        if args.show:
            cv2.imshow("Fall detection", frame)
            if cv2.waitKey(1 if is_webcam else max(1, int(1000 / fps))) & 0xFF == ord("q"):
                break
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()