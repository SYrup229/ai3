import os
import time
import cv2 as cv
import numpy as np
from imutils.video import VideoStream

# ---- Settings ----
data_path   = 'data'          # root folder where images will be saved
frame_size  = (320, 240)      # camera resolution (w, h)
show_help   = True            # show on-screen help overlay
capture_interval = 0.5        # seconds between auto-captures

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

print("[INFO] Ready.")
print("  Press a class key to START auto-capture (1 img/sec):")
print("  2-9, 0 or t/T=10, a/A=A, j/J=J, q/Q=Q, k/K=K")
print("  Press q or Esc once to STOP capturing (app stays open).")
print("  Press q or Esc AGAIN to EXIT.\n")

# Per-class counters (for display only)
def count_files(d):
    try:
        return len([f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))])
    except FileNotFoundError:
        return 0

saved_counts = {name: count_files(os.path.join(data_path, name)) for name in class_names}

# Capture state
capturing = False
current_label = None
last_capture_time = 0.0
quit_armed = False  # True after first q/ESC press; next q/ESC exits

def draw_overlay(img):
    if not show_help:
        return img
    overlay = img.copy()
    h, w = img.shape[:2]

    status = "CAPTURING" if capturing else ("PRESS CLASS KEY" if not quit_armed else "PAUSED (q/Esc to EXIT)")
    label_txt = current_label if current_label is not None else "-"
    lines = [
        f"Status: {status} | Label: {label_txt}",
        "Keys: 2-9 | 0/t=10 | a=A j=J q=Q k=K | q/Esc: stop, then exit",
        "Saved: " + " ".join(f"{k}:{saved_counts[k]}" for k in class_names)
    ]

    # Dynamic bar
    line_height = 22
    top = 5
    bottom = top + line_height * len(lines) + 12
    cv.rectangle(overlay, (5, top), (w - 5, bottom), (0, 0, 0), -1)

    img = cv.addWeighted(overlay, 0.65, img, 0.35, 0)

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

        # Auto-capture logic
        now = time.time()
        if capturing and current_label is not None:
            if (now - last_capture_time) >= capture_interval:
                out_dir = os.path.join(data_path, current_label)
                ts = int(time.time_ns())
                filename = f"{current_label}_{ts}.png"
                out_path = os.path.join(out_dir, filename)

                ok = cv.imwrite(out_path, frame)
                if ok:
                    saved_counts[current_label] += 1
                    print(f"[INFO] saved: {out_path}")
                else:
                    print(f"[WARN] failed to save: {out_path}")

                last_capture_time = now  # reset timer

        # Show overlay
        frame_disp = draw_overlay(frame.copy())
        cv.imshow("Card Capture", frame_disp)

        k = cv.waitKey(1) & 0xFF

        # Handle quit toggles
        if  k == 27:  # q or Esc
            if capturing:
                # First press: stop capturing, arm quit
                capturing = False
                quit_armed = True
                print("[INFO] Capture stopped. Press Esc again to exit, or press a class key to resume with that label.")
            else:
                if quit_armed:
                    # Second press: exit
                    print("[INFO] Exiting.")
                    break
                else:
                    # Not capturing and not armed: arm quit
                    quit_armed = True
                    print("[INFO] Press q/Esc again to exit, or press a class key to start capturing.")
            continue

        # Handle class keys
        if k in kMappings:
            current_label = kMappings[k]
            # Start (or switch) capturing; immediate first shot
            capturing = True
            quit_armed = False
            last_capture_time = 0.0  # force immediate save on next loop
            print(f"[INFO] Capturing {current_label} at {capture_interval:.1f}s intervals. Press q/Esc to pause.")

finally:
    try:
        vs.stop()
    except Exception:
        pass
    cv.destroyAllWindows()
