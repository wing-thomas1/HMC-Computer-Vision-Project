import cv2
import numpy as np
import random

width, height = 800, 600
fps = 30
duration_sec = 8
frames = duration_sec * fps

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter("synthetic_vid.mp4", fourcc, fps, (width, height))

circles = [
    # moves left to right, disappears later
    {
        "x": 100, "y": 300, 
        "vx": 3, "vy": 0,
        "radius": 25,
        "id": 0,
        "disappear": [(140, 170)],  # disappears after crossing
        "stop": []
    },
    # moves right to left, crosses first circle
    {
        "x": 700, "y": 300,
        "vx": -3, "vy": 0,
        "radius": 25,
        "id": 1,
        "disappear": [(180, 210)],  # also disappears later
        "grow": True  
    },
    # jittery, moving around the middle
    {
        "x": 400, "y": 150,
        "vx": 1, "vy": 2,
        "radius": 20,
        "id": 2,
        "jitter": True,
        "disappear": []
    },
]

for frame_i in range(frames):
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    for c in circles:
        # Handle "stop" intervals if you add any later
        if any(start < frame_i < end for start, end in c.get("stop", [])):
            # no motion update
            pass
        else:
            # Normal motion update
            c["x"] += c["vx"]
            c["y"] += c["vy"]

        # Bounce off walls
        if c["x"] < c["radius"]:
            c["x"] = c["radius"]
            c["vx"] *= -1
        if c["x"] > width - c["radius"]:
            c["x"] = width - c["radius"]
            c["vx"] *= -1

        if c["y"] < c["radius"]:
            c["y"] = c["radius"]
            c["vy"] *= -1
        if c["y"] > height - c["radius"]:
            c["y"] = height - c["radius"]
            c["vy"] *= -1

        # Radius change behavior
        if c.get("grow", False):
            # oscillate radius a bit around 25
            c["radius"] = 20 + int(5 * np.sin(frame_i / 10))

        # Jitter behavior
        jitter_x = jitter_y = 0
        if c.get("jitter", False):
            # small jitter so it doesn't look crazy
            jitter_x = np.random.normal(0, 1.5)
            jitter_y = np.random.normal(0, 1.5)

        # Disappearance behavior
        if any(start < frame_i < end for start, end in c.get("disappear", [])):
            continue 

        draw_x = int(c["x"] + jitter_x)
        draw_y = int(c["y"] + jitter_y)

        cv2.circle(
            frame,
            (draw_x, draw_y),
            c["radius"],
            (255, 255, 255),
            -1,
        )

    writer.write(frame)

writer.release()
print("Saved video")
