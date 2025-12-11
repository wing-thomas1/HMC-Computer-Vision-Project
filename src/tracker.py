import cv2
import numpy as np
from kalman_tracker import make_kf 

class Track:
    def __init__(self, kf, track_id, radius):
        self.kf = kf
        self.id = track_id
        self.radius = radius
        self.misses = 0 


def detect_white_circles(frame, min_area=50):
    """
    Returns a list of (cx, cy, r_est).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # radius estimate from area
        r = int(np.sqrt(area / np.pi))
        detections.append((cx, cy, r))

    return detections


def make_kf_from_detection(cx, cy):
    """
    state: [x, y, vx, vy]
    """
    kf = make_kf()
    kf.x[0, 0] = cx
    kf.x[1, 0] = cy
    kf.x[2, 0] = 0.0
    kf.x[3, 0] = 0.0
    return kf


def assign_detections_to_tracks(tracks, detections, max_dist=60):
    """
    tracks: list of Track objects
    detections: list of (cx, cy, r)
    returns:
        assigned: dict {track_index : detection_index}
        unassigned_tracks: set of track indices
        used_detections: set of detection indices
    """
    assigned = {}
    used_dets = set()
    unassigned_tracks = set(range(len(tracks)))

    max_dist2 = max_dist ** 2

    for ti, tr in enumerate(tracks):
        px = float(tr.kf.x[0])
        py = float(tr.kf.x[1])

        best_di = None
        best_d2 = float("inf")

        for di, (cx, cy, r) in enumerate(detections):
            if di in used_dets:
                continue

            dx = cx - px
            dy = cy - py
            d2 = dx * dx + dy * dy

            if d2 < best_d2 and d2 < max_dist2:
                best_d2 = d2
                best_di = di

        if best_di is not None:
            assigned[ti] = best_di
            used_dets.add(best_di)
            unassigned_tracks.discard(ti)

    return assigned, unassigned_tracks, used_dets


def track_white_circles(video_path, save_path="synthetic_vid_tracked.mp4"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video.")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("Empty video?")
        return

    height, width = first_frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    tracks = []
    next_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_white_circles(frame)

        # 1. predict step
        for tr in tracks:
            tr.kf.predict()

        # 2. assign detections to tracks
        assigned, unassigned_tracks, used_dets = assign_detections_to_tracks(tracks, detections)

        # 3. update matched tracks
        for ti, di in assigned.items():
            cx, cy, r = detections[di]
            z = np.array([[cx], [cy]], dtype=np.float32)
            tracks[ti].kf.update(z)
            tracks[ti].radius = r 
            tracks[ti].misses = 0

        # 4. increment misses for tracks without matches
        for ti in unassigned_tracks:
            tracks[ti].misses += 1

        # 5. drop tracks that have been missing too long
        tracks = [tr for tr in tracks if tr.misses < 60] 

        # 6. create new tracks for unused detections
        for di, (cx, cy, r) in enumerate(detections):
            if di not in used_dets:
                kf = make_kf_from_detection(cx, cy)
                tracks.append(Track(kf, next_id, r))
                next_id += 1

        # 7. draw detections + tracks for visualization
        draw = frame.copy()

        # detections in blue
        for (cx, cy, r) in detections:
            cv2.circle(draw, (cx, cy), r, (255, 0, 0), 2)

        # tracks predicted state in red with green IDs
        for tr in tracks:
            x_est = int(tr.kf.x[0])
            y_est = int(tr.kf.x[1])
            r_est = int(tr.radius)
            cv2.circle(draw, (x_est, y_est), r_est, (0, 0, 255), 2)
            cv2.putText(draw, f"ID {tr.id}", (x_est + 10, y_est),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Synthetic Tracking", cv2.resize(draw, None, fx=0.7, fy=0.7))
        writer.write(draw)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved tracking video to {save_path}")


if __name__ == "__main__":
    track_white_circles("synthetic_vid.mp4", save_path="synthetic_vid_tracked.mp4")
