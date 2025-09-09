import os
import time
import cv2 as cv
import numpy as np
from imutils.video import VideoStream

# ---- Settings ----
data_path   = 'data'          # root folder where images will be saved
frame_size  = (320, 240)      # camera resolution (w, h)
show_help   = True            # show on-screen help overlay

# Classes and key bindings
# You can press: 2..9, 0 or 't' for 10, and a/j/q/k for faces
class_names = ["2","3","4","5","6","7","8","9","10","A","J","Q","K"]

# Map keyboard keys (ord) -> class folder name
kMappings = {
    ord('2'): "2",
    ord('3'): "3",
    ord('4'): "4",
    ord('5'): "5",
    ord('6'): "6",
    ord('7'): "7",
    ord('8'): "8",
    ord('9'): "9",
    ord('0'): "10",   # 0 key stands for "10"
    ord('t'): "10",   # t also stands for "Ten"
    ord('T'): "10",
    ord('a'): "A",
    ord('A'): "A",
    ord('j'): "J",
    ord('J'): "J",
    ord('q'): "Q",
    ord('Q'): "Q",
    ord('k'): "K",
    ord('K'): "K",
}

# Ensure class folders exist
for name in class_names:
    os.makedirs(os.path.join(data_path, name), exist_ok=True)

# Start camera
vs = VideoStream(src=0, usePiCamera=False, resolution=frame_size).start()
time.sleep(1.0)

print("[INFO] Ready. Press keys to save:")
print("  2-9  -> classes 2..9")
print("  0 or t/T -> class 10")
print("  a/A  -> A,  j/J -> J,  q/Q -> Q,  k/K -> K")
print("  q or Esc to quit\n")

# Optional: keep simple per-class counters in memory (for display only)
saved_counts = {name: len(os.listdir(os.path.join(data_path, name))) for name in class_names}

def draw_overlay(img):
    if not show_help:
        return img
    overlay = img.copy()
    h, w = img.shape[:2]

    lines = [
        "Save keys: 2-9 | 0/t=10 | a=A j=J q=Q k=K",
        "Quit: q or Esc",
        "Saved: " + " ".join(f"{k}:{saved_counts[k]}" for k in class_names)
    ]

    # Adjusted bar height based on number of lines
    line_height = 22
    top = 5
    bottom = top + line_height * len(lines) + 10  # add padding
    cv.rectangle(overlay, (5, top), (w - 5, bottom), (0, 0, 0), -1)

    # blend with frame
    img = cv.addWeighted(overlay, 0.65, img, 0.35, 0)

    # redraw crisp white text
    y = top + 20
    for line in lines:
        cv.putText(img, line, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.55,
                   (255, 255, 255), 1, cv.LINE_AA)
        y += line_height

    return img

try:
    while True:
        frame = vs.read()
        if frame is None:
            continue

        # show help/info overlay
        frame_disp = draw_overlay(frame.copy())
        cv.imshow("Card Capture", frame_disp)

        k = cv.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:   # 'q' or ESC
            break

        if k in kMappings:
            label = kMappings[k]
            # build path and filename
            out_dir = os.path.join(data_path, label)
            ts = int(time.time_ns())
            filename = f"{label}_{ts}.png"
            out_path = os.path.join(out_dir, filename)

            # save frame as-is (full frame). If you prefer ROI later, crop here.
            ok = cv.imwrite(out_path, frame)
            if ok:
                saved_counts[label] += 1
                print(f"[INFO] saved: {out_path}")
            else:
                print(f"[WARN] failed to save: {out_path}")

finally:
    try:
        vs.stop()
    except Exception:
        pass
    cv.destroyAllWindows()
