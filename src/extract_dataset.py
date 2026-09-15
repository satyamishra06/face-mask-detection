"""
Reads PASCAL VOC style annotations from the andrewmvd Face Mask Detection dataset,
crops each labeled face out of its source image, and saves it into
data/raw/<class_name>/ so it matches our project's folder structure.
"""

import os
import cv2
import xml.etree.ElementTree as ET

# ---- CONFIG: update this path to where you extracted the dataset ----
SOURCE_DIR = r"C:\Users\HP\Downloads\archieve"
IMAGES_DIR = os.path.join(SOURCE_DIR, "images")
ANNOTATIONS_DIR = os.path.join(SOURCE_DIR, "annotations")

OUTPUT_DIR = "data/raw"

# maps dataset's original label names -> our project's folder names
LABEL_MAP = {
    "with_mask": "mask",
    "without_mask": "no_mask",
    "mask_weared_incorrect": "improper_mask",
}


def ensure_output_folders():
    for folder in LABEL_MAP.values():
        os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)


def process_annotation(xml_path, counters):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    img_path = os.path.join(IMAGES_DIR, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"⚠️  Could not read image: {img_path}")
        return

    for obj in root.findall("object"):
        label = obj.find("name").text
        if label not in LABEL_MAP:
            continue

        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        face_crop = img[ymin:ymax, xmin:xmax]
        if face_crop.size == 0:
            continue

        out_class = LABEL_MAP[label]
        counters[out_class] = counters.get(out_class, 0) + 1
        out_name = f"{out_class}_{counters[out_class]}.png"
        out_path = os.path.join(OUTPUT_DIR, out_class, out_name)

        cv2.imwrite(out_path, face_crop)


def main():
    ensure_output_folders()
    counters = {}

    xml_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith(".xml")]
    print(f"Found {len(xml_files)} annotation files. Processing...")

    for i, xml_file in enumerate(xml_files, 1):
        process_annotation(os.path.join(ANNOTATIONS_DIR, xml_file), counters)
        if i % 100 == 0:
            print(f"  processed {i}/{len(xml_files)}")

    print("\nDone! Face counts per class:")
    for cls, count in counters.items():
        print(f"  {cls}: {count}")


if __name__ == "__main__":
    main()