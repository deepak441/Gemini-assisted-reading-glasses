"""
test_ai.py

Reads slide screenshots from a folder, OCRs each one, sends the extracted
text to Gemini to generate study-note bullet points, and appends the
result to a markdown study guide.

This script's job, and only job, is: turn a folder of slide images into a
study guide. It does not touch the live camera stream — see capture.py
for that.
"""

import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import google.generativeai as genai
from paddleocr import PaddleOCR

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.5-flash"

DESKTOP = Path.home() / "Desktop"
SLIDES_FOLDER = DESKTOP / "test_slides"
OUTPUT_FILE = DESKTOP / "test_study_guide.md"
PROCESSED_LOG = DESKTOP / ".test_study_guide_processed.txt"  # tracks slides already summarized

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds, doubles on each retry


PROMPT_TEMPLATE = """
You are the note-taking brain for AI reading glasses.

The user is watching a fast-moving slide or presentation. OCR extracted this text:

\"\"\"{detected_text}\"\"\"

Create detailed study notes for the user.

Rules:
- Output only bullet points.
- Use 5 to 7 bullet points.
- Each bullet should be useful for later studying.
- Add important dates, numbers, rankings, locations, capacities, and measurements when relevant.
- Add helpful background information that completes the idea, but keep it technical and concise.
- Be clean and concise without compromising clarity and information.
- Do not use filler or padding words.
- Correct obvious OCR mistakes, spelling, capitalization, and place names.
- Keep the notes natural, like a strong student wrote them during class.
- Do not say "Here is a summary."
- Do not mention OCR, screenshot, Gemini, AI, or the slide.
- Do not use markdown bold text.
- Do not use labels like "Location:", "Purpose:", or "Historical significance:".
- Do not invent facts. If you are not confident about a number or date, leave it out.
- Use approximate wording like "about" or "around" when exact values may vary.
- Start every bullet with "- ".
"""


def get_model():
    """Configure Gemini and return a ready-to-use model instance.

    Used by both this file's run() and by capture.py (via `import test_ai as ai`
    then `ai.get_model()`) so Gemini only ever gets configured in one place.
    """
    if not API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Run: export GEMINI_API_KEY='your-key-here'"
        )
    genai.configure(api_key=API_KEY)
    return genai.GenerativeModel(MODEL_NAME)


def load_processed() -> set:
    if not PROCESSED_LOG.exists():
        return set()
    return set(PROCESSED_LOG.read_text(encoding="utf-8").splitlines())


def mark_processed(img_name: str) -> None:
    with open(PROCESSED_LOG, "a", encoding="utf-8") as f:
        f.write(img_name + "\n")


def extract_text(ocr_result) -> str:
    """
    Pull plain text out of a PaddleOCR .predict() result.

    Current PaddleOCR returns a list of dict-like objects with 'rec_texts'.
    Older versions returned deeply nested lists instead — handled here too
    so this doesn't silently return nothing on a different paddleocr build.
    """
    if not ocr_result:
        return ""

    lines = []
    for page in ocr_result:
        rec_texts = page.get("rec_texts") if hasattr(page, "get") else None
        if rec_texts:
            lines.extend(rec_texts)
            continue

        if isinstance(page, list):
            for item in page:
                if isinstance(item, list) and len(item) > 1:
                    try:
                        lines.append(item[1][0])
                    except (IndexError, TypeError):
                        continue

    return "\n".join(t for t in lines if t).strip()


def save_notes_to_guide(notes: str, slide_number: int, img_name: str) -> None:
    file_exists = OUTPUT_FILE.exists()
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("# Study Guide\n\n")
            f.write(f"## Notes taken on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Slide {slide_number}: {img_name}\n\n")
        f.write(notes.strip())
        f.write("\n\n---\n\n")


def generate_notes(model, detected_text: str) -> str:
    """Call Gemini with retries. Raises RuntimeError if it never succeeds."""
    prompt = PROMPT_TEMPLATE.format(detected_text=detected_text)
    delay = RETRY_DELAY

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = model.generate_content(prompt)
        except Exception as e:
            print(f"  Gemini call failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Gemini call failed after {MAX_RETRIES} attempts") from e
            time.sleep(delay)
            delay *= 2
            continue

        # response.text raises if the response was blocked / has no text parts
        try:
            text = response.text
        except Exception as e:
            print(f"  Gemini returned no usable text (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES:
                raise RuntimeError("Gemini response had no usable text") from e
            time.sleep(delay)
            delay *= 2
            continue

        if not text or not text.strip():
            print(f"  Gemini returned empty text (attempt {attempt}/{MAX_RETRIES})")
            if attempt == MAX_RETRIES:
                raise RuntimeError("Gemini returned empty text")
            time.sleep(delay)
            delay *= 2
            continue

        return text.strip()

    raise RuntimeError("Gemini call did not succeed")  # unreachable, safety net


def run():
    model = get_model()
    ocr = PaddleOCR(use_textline_orientation=True, lang="en")

    if not SLIDES_FOLDER.exists():
        raise FileNotFoundError(
            f"Slides folder not found: {SLIDES_FOLDER}. "
            "Create it and add some slides in .png, .jpg, or .jpeg format."
        )

    img_paths = sorted(
        list(SLIDES_FOLDER.glob("*.png"))
        + list(SLIDES_FOLDER.glob("*.jpg"))
        + list(SLIDES_FOLDER.glob("*.jpeg"))
    )
    if not img_paths:
        raise FileNotFoundError(
            f"No image files found in {SLIDES_FOLDER}. "
            "Please add some slides in .png, .jpg, or .jpeg format."
        )

    processed = load_processed()
    skipped = [p for p in img_paths if p.name in processed]
    todo = [p for p in img_paths if p.name not in processed]

    if skipped:
        print(f"Skipping {len(skipped)} already-processed slide(s): {[p.name for p in skipped]}")
    if not todo:
        print("Nothing new to process. Delete the .test_study_guide_processed.txt "
              "log if you want to reprocess slides.")
        return

    for slide_number, img_path in enumerate(todo, start=1):
        print(f"\nProcessing slide {slide_number}/{len(todo)}: {img_path.name}")
        frame = cv2.imread(str(img_path))

        if frame is None:
            print(f"  Warning: could not read image {img_path}. Skipping.")
            continue

        print("  Running OCR...")
        try:
            result = ocr.predict(frame)
        except Exception as e:
            print(f"  OCR failed on this slide, skipping: {e}")
            continue

        detected_text = extract_text(result)
        if not detected_text:
            print("  No text found on this slide. Skipping.")
            mark_processed(img_path.name)  # don't retry a blank slide forever
            continue

        print("  --- RAW OCR TEXT ---")
        print("  " + detected_text.replace("\n", "\n  "))

        print("  Sending to Gemini...")
        try:
            notes = generate_notes(model, detected_text)
        except RuntimeError as e:
            print(f"  Giving up on this slide: {e}")
            print("  (not marked as processed — it will be retried next run)")
            continue

        print("  --- GENERATED STUDY NOTES ---")
        print("  " + notes.replace("\n", "\n  "))

        save_notes_to_guide(notes=notes, slide_number=slide_number, img_name=img_path.name)
        mark_processed(img_path.name)
        print(f"  Saved notes for slide {slide_number} to: {OUTPUT_FILE}")

    print(f"\nDone. Study guide created / updated here: {OUTPUT_FILE}")


if __name__ == "__main__":
    run()