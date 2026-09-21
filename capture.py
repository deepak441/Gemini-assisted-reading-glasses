"""
capture.py

Connects to an MJPEG stream (e.g. an ESP32-CAM), watches for the on-screen
slide to change, waits for the camera to settle on the new slide, then
hands that frame off to test_ai.py's pipeline: OCR -> Gemini -> study notes.

"""

import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests

import test_ai as ai  # reuses extract_text, generate_notes, save_notes_to_guide, etc.

# ---------------------------------------------------------------------------
# Stream config
# ---------------------------------------------------------------------------
STREAM_URL = "http://192.168.1.127:81/stream"
CONNECT_TIMEOUT = 10
CHUNK_SIZE = 1024
MAX_BUFFER_SIZE = 5_000_000
RECONNECT_DELAY = 3

LOG_FILE = Path.home() / "Desktop" / "AI_glasses" / "readit.txt"  # lightweight raw-text log, for debugging

# ---------------------------------------------------------------------------
# Change-detection config
# ---------------------------------------------------------------------------
DIFF_SIZE = (160, 90)       # downscale frames to this size before diffing (cheap + noise-tolerant)
CHANGE_THRESHOLD = 12       # mean pixel diff (0-255) vs. last committed slide to flag "something changed"
STABILITY_THRESHOLD = 5     # mean pixel diff vs. previous frame below this counts as "not moving"
STABILITY_FRAMES = 3        # consecutive stable frames required before we commit a new slide
MIN_SECONDS_BETWEEN_SLIDES = 2  # cooldown so one transition can't trigger multiple commits


def to_small_gray(frame):
    small = cv2.resize(frame, DIFF_SIZE)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def mean_diff(a, b) -> float:
    return float(np.mean(cv2.absdiff(a, b)))


def log_raw_text(text: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"--- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write(text.strip())
        f.write("\n\n")


def next_slide_number() -> int:
    existing = (
        list(ai.SLIDES_FOLDER.glob("*.png"))
        + list(ai.SLIDES_FOLDER.glob("*.jpg"))
        + list(ai.SLIDES_FOLDER.glob("*.jpeg"))
    )
    return len(existing) + 1


def commit_slide(frame, ocr, model) -> None:
    """A new, settled slide was detected: OCR it, and if it has text, send it to Gemini."""
    ai.SLIDES_FOLDER.mkdir(parents=True, exist_ok=True)

    try:
        result = ocr.predict(frame)
    except Exception as e:
        print(f"  OCR failed on this slide, skipping: {e}")
        return

    detected_text = ai.extract_text(result)
    if not detected_text:
        print("  No text detected in new slide — not saving.")
        return

    slide_number = next_slide_number()
    img_name = f"slide_{slide_number:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    img_path = ai.SLIDES_FOLDER / img_name
    cv2.imwrite(str(img_path), frame)

    print(f"\n--- NEW SLIDE DETECTED (#{slide_number}) ---")
    print(detected_text)
    log_raw_text(detected_text)

    print("Sending to Gemini...")
    try:
        notes = ai.generate_notes(model, detected_text)
    except RuntimeError as e:
        print(f"  Giving up on this slide for now: {e}")
        print(f"  Frame was saved to {img_path} — rerun test_ai.py later to retry it.")
        return

    print("--- GENERATED STUDY NOTES ---")
    print(notes)
    print("------------------------------\n")

    ai.save_notes_to_guide(notes=notes, slide_number=slide_number, img_name=img_name)
    ai.mark_processed(img_name)
    print(f"Saved notes to: {ai.OUTPUT_FILE}")


def worker_loop(work_queue: "queue.Queue", ocr, model) -> None:
    """Runs on a background thread. Pulls one settled slide at a time off the
    queue and processes it (OCR -> Gemini -> save). This is the only place
    commit_slide ever gets called from, so ocr/model/the filesystem writes
    inside it are never touched concurrently.
    """
    while True:
        frame = work_queue.get()
        try:
            commit_slide(frame, ocr, model)
        except Exception as e:
            # Catch-all so one bad slide can't silently kill the worker thread
            # and leave the queue filling up with no consumer.
            print(f"  Worker hit an unexpected error processing a slide: {e}")
        finally:
            work_queue.task_done()


def open_stream():
    return requests.get(STREAM_URL, stream=True, timeout=CONNECT_TIMEOUT)


def run():
    from paddleocr import PaddleOCR

    print("Loading OCR model...")
    ocr = PaddleOCR(use_textline_orientation=True, lang="en")
    print("Configuring Gemini...")
    model = ai.get_model()

    # Small bounded queue: if the worker falls behind, new candidates get
    # dropped (see worker_loop docstring / module docstring) rather than
    # queueing up unboundedly or blocking the main thread.
    work_queue: "queue.Queue" = queue.Queue(maxsize=2)
    worker = threading.Thread(
        target=worker_loop, args=(work_queue, ocr, model), daemon=True
    )
    worker.start()

    # State for change detection
    reference_small = None   # downscaled grayscale of the last *committed* slide
    prev_small = None        # downscaled grayscale of the previous frame (for stability check)
    state = "IDLE"            # IDLE -> waiting for a change; STABILIZING -> waiting for it to settle
    stable_count = 0
    candidate_full = None
    last_commit_time = 0.0

    print(f"Logging raw detections to: {LOG_FILE}")
    print(f"Saving slide images to: {ai.SLIDES_FOLDER}")
    print(f"Writing study notes to: {ai.OUTPUT_FILE}")

    while True:
        try:
            print(f"\nConnecting to {STREAM_URL} ...")
            stream = open_stream()
            print("Connected. Watching for slide changes...")
            bytes_buffer = b""

            for chunk in stream.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                bytes_buffer += chunk

                if len(bytes_buffer) > MAX_BUFFER_SIZE:
                    print("Buffer exceeded max size without a valid frame; resetting.")
                    bytes_buffer = b""
                    continue

                start = bytes_buffer.find(b"\xff\xd8")
                end = bytes_buffer.find(b"\xff\xd9")
                if start == -1 or end == -1 or end < start:
                    continue

                jpg_bytes = bytes_buffer[start:end + 2]
                bytes_buffer = bytes_buffer[end + 2:]

                frame = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                small = to_small_gray(frame)

                # First frame ever: nothing to compare against yet. Treat it as the start
                # of a "new slide" so whatever the camera is already pointed at gets captured.
                if reference_small is None:
                    reference_small = small
                    prev_small = small
                    state = "STABILIZING"
                    candidate_full = frame
                    stable_count = 0
                    continue

                if state == "IDLE":
                    if mean_diff(small, reference_small) > CHANGE_THRESHOLD:
                        state = "STABILIZING"
                        candidate_full = frame
                        stable_count = 0
                    prev_small = small

                elif state == "STABILIZING":
                    if mean_diff(small, prev_small) < STABILITY_THRESHOLD:
                        stable_count += 1
                    else:
                        stable_count = 0
                    candidate_full = frame
                    prev_small = small

                    if stable_count >= STABILITY_FRAMES:
                        now = time.time()
                        if now - last_commit_time >= MIN_SECONDS_BETWEEN_SLIDES:
                            try:
                                work_queue.put_nowait(candidate_full)
                            except queue.Full:
                                print("  Worker busy with a previous slide — skipping this one.")
                            last_commit_time = now
                        reference_small = small
                        state = "IDLE"

        except KeyboardInterrupt:
            if not work_queue.empty():
                print("\nStopping — waiting for the worker to finish the current slide...")
                work_queue.join()
            print("Stopped by user.")
            return
        except requests.exceptions.RequestException as e:
            print(f"Stream connection error: {e}")
            print(f"Retrying in {RECONNECT_DELAY}s...")
            time.sleep(RECONNECT_DELAY)
        except Exception as e:
            print(f"Unexpected error: {e}")
            print(f"Retrying in {RECONNECT_DELAY}s...")
            time.sleep(RECONNECT_DELAY)


if __name__ == "__main__":
    run()
