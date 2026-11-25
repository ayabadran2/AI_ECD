"""
Image version of the fall-detection pose check, with an optional sequence mode.

MODE 1 - single images (default)
  A single image has no time dimension, so the "fast hip drop" and "lying for N seconds"
  checks cannot run. The per-frame part of the algorithm does run: is the torso lying
  down, judged from pose keypoints. Reported as LYING = "possible fall, needs review".
  For every person it prints TWO verdicts so you can compare:
    - original : the exact logic from your video script
    - hardened : same logic + ignores joints outside the frame, and does not trust the
                 aspect-ratio fallback when the box touches the image border

MODE 2 - sequence (--sequence)
  Treats a folder of images as consecutive video frames (sorted by file name) and runs
  the FULL original algorithm: tracking, hip drop, lying timer, cooldown.
  Use --fps to say how many of these frames make up one second of video.
  Add --hardened to apply the edge checks + stale-state reset.

Usage:
  python fall_image_test.py --source photo.jpg
  python fall_image_test.py --source frames_folder
  python fall_image_test.py --source frames_folder --sequence --fps 10
  python fall_image_test.py --source frames_folder --sequence --fps 10 --hardened
"""
import argparse
import math
import os
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

KP_CONF = 0.4           # minimum keypoint confidence to trust a joint
LYING_ANGLE = 60.0      # torso angle from vertical (degrees) that counts as lying
ASPECT_LYING = 1.5      # fallback: box width / height above this counts as lying
EDGE_MARGIN = 5         # pixels: joints/boxes this close to the border are treated as cut off

# sequence mode only (same values as the video script)
LYING_SECONDS = 1.0
DROP_WINDOW = 1.0
DROP_RATIO = 0.30
DROP_MEMORY = 3.0
COOLDOWN = 10.0
UNKNOWN_RESET = 2.0     # hardened only: seconds of unreliable pose before lying state is cleared

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
JOINTS = {5: "L-shoulder", 6: "R-shoulder", 11: "L-hip", 12: "R-hip"}


class Person:
    def __init__(self):
        self.hist = deque()        # (time, hip_y, box_height)
        self.lying_since = None
        self.unknown_since = None
        self.last_drop = -1e9
        self.last_alert = -1e9


def mid(kp, conf, a, b):
    """Midpoint of two joints, using only the ones detected with enough confidence."""
    pts = [kp[i] for i in (a, b) if conf[i] >= KP_CONF]
    return np.mean(pts, axis=0) if pts else None


def analyse(kp, conf, box, frame_h, frame_w, hardened):
    """
    Return (lying, reason, hip_y, box_h).
    lying is True / False, or None when the pose is unreliable.
    """
    x1, y1, x2, y2 = box
    box_w = float(x2 - x1)
    box_h = max(float(y2 - y1), 1.0)

    if hardened:
        # a joint outside (or at the edge of) the frame is not a real detection
        inside = ((kp[:, 0] >= 0) & (kp[:, 0] < frame_w) &
                  (kp[:, 1] >= 0) & (kp[:, 1] < frame_h - EDGE_MARGIN))
        conf = np.where(inside, conf, 0.0)

    shoulders = mid(kp, conf, 5, 6)
    hips = mid(kp, conf, 11, 12)
    hip_y = float(hips[1]) if hips is not None else None

    if shoulders is None or hips is None:
        missing = []
        if shoulders is None:
            missing.append("shoulders")
        if hips is None:
            missing.append("hips")
        aspect = box_w / box_h
        touches_border = (x1 <= EDGE_MARGIN or y1 <= EDGE_MARGIN or
                          x2 >= frame_w - EDGE_MARGIN or y2 >= frame_h - EDGE_MARGIN)
        reason = (f"missing {'+'.join(missing)} -> aspect fallback: "
                  f"width/height={aspect:.2f} (limit {ASPECT_LYING}), "
                  f"box touches border={touches_border}")
        if aspect > ASPECT_LYING and not (hardened and touches_border):
            return True, reason, hip_y, box_h
        return None, reason, hip_y, box_h

    dx, dy = hips[0] - shoulders[0], hips[1] - shoulders[1]
    angle = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))  # 0 = upright, 90 = horizontal
    reason = f"torso angle from vertical = {angle:.1f} deg (limit {LYING_ANGLE})"
    return angle > LYING_ANGLE, reason, hip_y, box_h


def update(p, t, hip_y, box_h, lying, hardened):
    """Same temporal logic as the video script. Returns True when an alert should fire."""
    if hip_y is not None:
        p.hist.append((t, hip_y, box_h))
    while p.hist and t - p.hist[0][0] > DROP_WINDOW:
        p.hist.popleft()
    if len(p.hist) >= 2:
        descent = p.hist[-1][1] - min(h[1] for h in p.hist)   # image y grows downward
        ref_height = max(h[2] for h in p.hist)
        if descent / ref_height >= DROP_RATIO:
            p.last_drop = t
    if lying is None:
        if hardened:
            if p.unknown_since is None:
                p.unknown_since = t
            elif t - p.unknown_since > UNKNOWN_RESET:
                p.lying_since = None          # stale: stop trusting the old lying state
        return False
    p.unknown_since = None
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


def verdict_text(lying):
    if lying is None:
        return "UNKNOWN"
    return "LYING" if lying else "upright"


def extract(r):
    """Pull boxes / keypoints / confidences out of an ultralytics result (or None)."""
    if r.boxes is None or r.keypoints is None or len(r.boxes) == 0:
        return None
    boxes = r.boxes.xyxy.cpu().numpy()
    kps = r.keypoints.xy.cpu().numpy()
    confs = (r.keypoints.conf.cpu().numpy()
             if r.keypoints.conf is not None else np.ones(kps.shape[:2]))
    return boxes, kps, confs


def draw_joints(frame, kp, cf):
    h, w = frame.shape[:2]
    for j in JOINTS:
        x, y = int(kp[j][0]), int(kp[j][1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame, (x, y), 5, (0, 255, 255) if cf[j] >= KP_CONF else (128, 128, 128), -1)


def save(frame, out_dir, path, suffix):
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, f"{stem}_{suffix}.jpg")
    cv2.imwrite(out_path, frame)
    return out_path


# --------------------------------------------------------------------------- mode 1
def process_image(path, model, out_dir, show):
    frame = cv2.imread(path)
    if frame is None:
        print(f"[!] Could not read image: {path}")
        return
    h, w = frame.shape[:2]
    print("=" * 70)
    print(f"{path}  ({w}x{h})")

    data = extract(model.predict(frame, conf=0.3, imgsz=640, verbose=False)[0])
    if data is None:
        print("  No person detected")
        return

    for i, (box, kp, cf) in enumerate(zip(*data)):
        orig, orig_reason, _, _ = analyse(kp, cf, box, h, w, hardened=False)
        hard, hard_reason, _, _ = analyse(kp, cf, box, h, w, hardened=True)

        print(f"\n  person {i}  box=({int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])})")
        print("  key joints (confidence / position):")
        for j, name in JOINTS.items():
            x, y = kp[j]
            outside = not (0 <= x < w and 0 <= y < h)
            print(f"    {name:11s} conf={cf[j]:.2f}  at ({x:.0f},{y:.0f})"
                  f"{' OUTSIDE FRAME' if outside else ''}")
        print(f"  original logic: {verdict_text(orig):8s} | {orig_reason}")
        print(f"  hardened logic: {verdict_text(hard):8s} | {hard_reason}")
        if orig is True:
            print("  -> would be flagged as a POSSIBLE FALL (needs review)")

        color = (0, 0, 255) if hard else ((0, 165, 255) if hard is None else (0, 255, 0))
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"p{i} orig:{verdict_text(orig)} hard:{verdict_text(hard)}",
                    (x1, max(y1 - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        draw_joints(frame, kp, cf)

    print(f"\n  saved: {save(frame, out_dir, path, 'result')}")
    if show:
        cv2.imshow("Fall image test (any key = next, q = quit)", frame)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key == ord("q"):
            raise SystemExit


# --------------------------------------------------------------------------- mode 2
def process_sequence(images, model, out_dir, fps, hardened):
    people = defaultdict(Person)
    any_alert = False
    print(f"Sequence mode: {len(images)} frames at {fps} fps "
          f"({'hardened' if hardened else 'original'} logic)")

    for idx, path in enumerate(images):
        frame = cv2.imread(path)
        if frame is None:
            print(f"[!] Could not read image: {path}")
            continue
        h, w = frame.shape[:2]
        t = idx / fps
        print("=" * 70)
        print(f"frame {idx}  t={t:.2f}s  {os.path.basename(path)}")

        r = model.track(frame, persist=True, conf=0.3, imgsz=640, verbose=False)[0]
        data = extract(r)
        if data is None:
            print("  No person detected")
            save(frame, out_dir, path, "seq")
            continue
        boxes, kps, confs = data
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else list(range(len(boxes)))

        for pid, box, kp, cf in zip(ids, boxes, kps, confs):
            person = people[pid]
            lying, reason, hip_y, box_h = analyse(kp, cf, box, h, w, hardened)
            fired = update(person, t, hip_y, box_h, lying, hardened)

            lying_for = (t - person.lying_since) if person.lying_since is not None else 0.0
            drop_recent = (t - person.last_drop) <= DROP_MEMORY
            hip_txt = f"{hip_y:.0f}" if hip_y is not None else "none"
            print(f"  id{pid}: {verdict_text(lying):8s} | {reason}")
            print(f"        hip_y={hip_txt}  lying_for={lying_for:.1f}s  "
                  f"recent_hip_drop={drop_recent}")

            alerted = fired or (t - person.last_alert < COOLDOWN)
            if fired:
                person.last_alert = t
                any_alert = True
                print("        *** FALL ALERT FIRED ***")

            color = (0, 0, 255) if alerted else (0, 255, 0)
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"id{pid} {'FALL' if alerted else 'ok'}",
                        (x1, max(y1 - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            draw_joints(frame, kp, cf)

        save(frame, out_dir, path, "seq")

    print("=" * 70)
    print("RESULT:", "at least one FALL alert fired" if any_alert else "no fall alert fired")


def collect_images(source):
    if os.path.isdir(source):
        names = sorted(n for n in os.listdir(source) if n.lower().endswith(IMAGE_EXTS))
        return [os.path.join(source, n) for n in names]
    return [source]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="image file or a folder of images")
    ap.add_argument("--weights", default="yolov8n-pose.pt")
    ap.add_argument("--out", default="fall_image_results", help="folder for annotated images")
    ap.add_argument("--show", action="store_true", help="open a window for each result (single-image mode)")
    ap.add_argument("--sequence", action="store_true",
                    help="treat the folder as consecutive video frames (sorted by file name)")
    ap.add_argument("--fps", type=float, default=10.0,
                    help="sequence mode: how many frames make up one second")
    ap.add_argument("--hardened", action="store_true",
                    help="sequence mode: use the hardened logic")
    args = ap.parse_args()

    images = collect_images(args.source)
    if not images:
        print("No images found in", args.source)
        return

    model = YOLO(args.weights)
    if args.sequence:
        process_sequence(images, model, args.out, args.fps, args.hardened)
    else:
        for path in images:
            process_image(path, model, args.out, args.show)
        print("\nLegend: red = LYING, green = upright, orange = UNKNOWN (not enough joints). "
              "Yellow dots = trusted joints, grey = low confidence.")


if __name__ == "__main__":
    main()
    