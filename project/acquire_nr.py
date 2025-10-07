import os
import time
import cv2 as cv
import numpy as np

# ---------------- Settings ----------------
data_path        = 'data'           # root folder where images will be saved
device_index     = 0                # camera index (0 = default)
preferred_size   = (1920, 1080)     # requested capture size (w, h)
capture_interval = 0.1              # seconds between auto-captures (1/s)
session_target   = 25               # take exactly 25 photos per run (per label)
show_help        = True             # overlay text on preview

# Preview (display only; saved images remain full-res)
fit_to_window    = True             # toggle with 'F'
max_display_w    = 1280
max_display_h    = 720

# Classes + keys (same as your original)
class_names = ["2","3","4","5","6","7","8","9","10","A","J","Q","K","ND","EMPTY"]
kMappings = {
    ord('2'): "2",  ord('3'): "3",  ord('4'): "4",  ord('5'): "5",
    ord('6'): "6",  ord('7'): "7",  ord('8'): "8",  ord('9'): "9",
    ord('0'): "10", ord('t'): "10", ord('T'): "10",
    ord('a'): "A",  ord('A'): "A",
    ord('j'): "J",  ord('J'): "J",
    ord('q'): "Q",  ord('Q'): "Q",   # Queen (Esc is the only quit)
    ord('k'): "K",  ord('K'): "K",
    ord('n'): "ND", ord('N'): "ND", ord('e'): "EMPTY", ord('E'): "EMPTY",
}

# -------------- Utilities -----------------
def count_files(d):
    try:
        return len([f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))])
    except FileNotFoundError:
        return 0

def aspect_ratio_close(w, h, target=(16, 9), tol=0.02):
    ar = w / float(h)
    tar = target[0] / float(target[1])
    return abs(ar - tar) <= tol

def ensure_class_dirs():
    for name in class_names:
        os.makedirs(os.path.join(data_path, name), exist_ok=True)

def make_display_frame(frame_bgr):
    """Scale frame to fit max_display_w/h (for preview only)."""
    if not fit_to_window:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    scale = min(max_display_w / w, max_display_h / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv.resize(frame_bgr, (new_w, new_h), interpolation=cv.INTER_AREA)
    return frame_bgr

def try_set_mode(vs, w, h, fps=None):
    """Try to set width/height[/fps] and confirm."""
    vs.set(cv.CAP_PROP_FRAME_WIDTH, w)
    vs.set(cv.CAP_PROP_FRAME_HEIGHT, h)
    if fps is not None:
        vs.set(cv.CAP_PROP_FPS, fps)
    time.sleep(0.2)
    aw = int(vs.get(cv.CAP_PROP_FRAME_WIDTH))
    ah = int(vs.get(cv.CAP_PROP_FRAME_HEIGHT))
    afps = vs.get(cv.CAP_PROP_FPS)
    return (aw, ah, afps)

def open_camera(index, preferred_size):
    """
    Tries multiple backends and MJPG to get a sane mode.
    Falls back to common 16:9 modes if needed.
    """
    backends = []
    # Prefer platform-appropriate backends
    # On Windows: MSMF then DSHOW; on Linux: V4L2; otherwise try default
    backends.extend([cv.CAP_MSMF, cv.CAP_DSHOW, cv.CAP_ANY])
    try:
        import platform
        if platform.system() == "Linux":
            backends = [cv.CAP_V4L2, cv.CAP_ANY]
    except Exception:
        pass

    for be in backends:
        vs = cv.VideoCapture(index, be)
        if not vs.isOpened():
            continue

        # MJPG often unlocks higher resolutions on USB cams
        vs.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))

        # Ask for preferred mode first
        aw, ah, afps = try_set_mode(vs, preferred_size[0], preferred_size[1], fps=30)

        # If camera gave us zero dims, try again with another backend
        if aw == 0 or ah == 0:
            vs.release()
            continue

        # If aspect is not ~16:9, try a fallback list
        if not aspect_ratio_close(aw, ah, (16, 9), tol=0.03):
            fallback_169 = [(1920,1080), (1600,900), (1280,720), (960,540)]
            chosen = None
            for (fw, fh) in fallback_169:
                taw, tah, tafps = try_set_mode(vs, fw, fh, fps=30)
                if taw == fw and tah == fh:
                    aw, ah, afps = taw, tah, tafps
                    chosen = (fw, fh)
                    print(f"[INFO] Using 16:9 fallback: {fw}x{fh}")
                    break
            if chosen is None:
                print(f"[WARN] Camera aspect {aw}x{ah} not 16:9; continuing anyway.")

        print(f"[INFO] Camera OK on backend={be}: {aw}x{ah} @ {afps:.0f}fps")
        return vs, (aw, ah), be

    raise RuntimeError("Failed to open camera on any backend. "
                       "Try a different device_index or unplug/plug the camera.")

# -------------- Overlay -------------------
def draw_overlay(img, capturing, current_label, session_taken):
    if not show_help:
        return img
    overlay = img.copy()
    h, w = img.shape[:2]

    status = "CAPTURING" if capturing else "IDLE"
    label_txt = current_label if current_label is not None else "-"

    # IMPORTANT: no "remaining" line here (as requested)
    lines = [
        f"Status: {status} | Label: {label_txt} | Session: {session_taken}/{session_target}",
        f"Interval: {capture_interval:.1f}s | Target per run: {session_target}",
        "Saved total: " + " ".join(f"{k}:{saved_counts[k]}" for k in class_names),
        "Keys: 2-9 | 0/t=10 | a=A j=J q=Q k=K n=ND | Esc=Exit | F=FitPreview",
    ]

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

# --------------- Main ---------------------
if __name__ == "__main__":
    ensure_class_dirs()
    saved_counts = {name: count_files(os.path.join(data_path, name)) for name in class_names}

    # Open camera with sensible defaults + fallbacks
    vs, actual_size, used_backend = open_camera(device_index, preferred_size)
    aw, ah = actual_size
    print(f"[INFO] Requested: {preferred_size[0]}x{preferred_size[1]}  Actual: {aw}x{ah}")

    # Window setup (NORMAL so we can shrink if needed; start somewhere visible)
    cv.namedWindow("Card Capture", cv.WINDOW_NORMAL)
    cv.moveWindow("Card Capture", 20, 20)
    # Pre-shrink the window; actual shown image will be scaled by make_display_frame
    cv.resizeWindow("Card Capture", max_display_w, max_display_h)

    print("[INFO] Ready.")
    print("  Press a class key to START auto-capture (25 shots @ 1/s):")
    print("  2-9, 0 or t/T=10, a/A=A, j/J=J, q/Q=Q, k/K=K, n/N=ND, e for empty")
    print("  F = toggle fit-to-window preview,  Esc = exit\n")

    # Capture state
    capturing = False
    current_label = None
    last_capture_time = 0.0
    session_taken_for_label = 0  # how many taken in the current 25-shot run

    try:
        while True:
            ret, frame = vs.read()
            if not ret or frame is None:
                continue

            now = time.time()

            # Auto-capture logic (save full-res 'frame', show scaled preview)
            if capturing and current_label is not None:
                if session_taken_for_label >= session_target:
                    capturing = False
                    print(f"[DONE] '{current_label}' session complete ({session_target} images).")
                else:
                    if (now - last_capture_time) >= capture_interval:
                        out_dir = os.path.join(data_path, current_label)
                        ts = int(time.time_ns())
                        filename = f"{current_label}_{ts}.png"
                        out_path = os.path.join(out_dir, filename)

                        ok = cv.imwrite(out_path, frame)  # save full-res
                        if ok:
                            saved_counts[current_label] += 1
                            session_taken_for_label += 1
                            print(f"[INFO] saved: {out_path}  "
                                  f"(session {session_taken_for_label}/{session_target})")
                        else:
                            print(f"[WARN] failed to save: {out_path}")

                        last_capture_time = now

                        if session_taken_for_label >= session_target:
                            capturing = False
                            print(f"[DONE] '{current_label}' session complete ({session_target} images).")

            # Build overlay on a copy of full-res, then scale for display
            overlay_src = draw_overlay(frame.copy(), capturing, current_label, session_taken_for_label)
            frame_disp  = make_display_frame(overlay_src)
            cv.imshow("Card Capture", frame_disp)

            k = cv.waitKey(1) & 0xFF

            # Exit
            if k == 27:  # Esc
                print("[INFO] Exiting.")
                break

            # Toggle fit preview
            if k in (ord('f'), ord('F')):
                fit_to_window = not fit_to_window
                print(f"[INFO] Fit-to-window preview: {'ON' if fit_to_window else 'OFF'}")
                continue

            # Start a new 25-shot run for the chosen label
            if k in kMappings:
                current_label = kMappings[k]
                capturing = True
                session_taken_for_label = 0
                last_capture_time = 0.0  # forces immediate first save on next loop
                print(f"[INFO] Capturing '{current_label}' — {session_target} images @ {capture_interval:.1f}s.")
    finally:
        try:
            vs.release()
        except Exception:
            pass
        cv.destroyAllWindows()
