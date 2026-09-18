"""
Captures webcam photos to add to the training dataset.
Run once per class, following on-screen prompts.
Press SPACE to capture a photo, ESC to finish that class.
"""

import cv2
import os
import sys

CLASS_NAME = sys.argv[1] if len(sys.argv) > 1 else None
if CLASS_NAME not in ["mask", "no_mask", "improper_mask"]:
    print("Usage: python src/capture_training_images.py <mask|no_mask|improper_mask>")
    sys.exit(1)

out_dir = f"data/raw/{CLASS_NAME}"
os.makedirs(out_dir, exist_ok=True)

existing = len(os.listdir(out_dir))
count = 0

cap = cv2.VideoCapture(0)
print(f"Capturing for class: {CLASS_NAME}")
print("Press SPACE to capture, ESC to finish.")
print("Move your head slightly (angle, distance) between captures for variety.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    cv2.putText(display, f"Class: {CLASS_NAME} | Captured: {count} | SPACE=capture ESC=finish",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Capture Training Images", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        # save a good-sized crop - centered square from the frame
        h, w = frame.shape[:2]
        size = min(h, w)
        y0 = (h - size) // 2
        x0 = (w - size) // 2
        crop = frame[y0:y0+size, x0:x0+size]
        crop = cv2.resize(crop, (300, 300))  # nice clear size, not tiny

        filename = f"{CLASS_NAME}_self_{existing + count}.png"
        cv2.imwrite(os.path.join(out_dir, filename), crop)
        count += 1
        print(f"Captured {count} images so far")

cap.release()
cv2.destroyAllWindows()
print(f"\nDone. Added {count} new images to {out_dir}")