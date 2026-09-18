"""
Randomly deletes excess images from the 'mask' class to reduce severe
class imbalance, instead of oversampling minority classes (which needs
too much RAM). Keeps only KEEP_COUNT random images from mask/.
"""

import os
import random

KEEP_COUNT = 650  # how many 'mask' images to keep
FOLDER = "data/raw/mask"

files = os.listdir(FOLDER)
print(f"Current count in '{FOLDER}': {len(files)}")

if len(files) <= KEEP_COUNT:
    print("Already at or below target count. Nothing to delete.")
else:
    random.shuffle(files)
    to_delete = files[KEEP_COUNT:]

    for fname in to_delete:
        os.remove(os.path.join(FOLDER, fname))

    print(f"Deleted {len(to_delete)} images.")
    print(f"Remaining: {len(os.listdir(FOLDER))}")