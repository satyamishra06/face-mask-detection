# Dataset Notes

**Source:** andrewmvd Face Mask Detection dataset (Kaggle) — 853 images, PASCAL VOC annotations
**Extraction:** Cropped individual faces from bounding boxes via `src/extract_dataset.py`

## Class distribution (raw, before balancing)
| Class | Count |
|---|---|
| mask | 3232 |
| no_mask | 717 |
| improper_mask | 123 |

**Observation:** Significant class imbalance — `mask` has ~26x more samples than `improper_mask`.
This matches a known limitation reported in prior work (YOLOv5 face mask studies found poor
performance specifically on the "mask worn incorrectly" class due to this same imbalance).

**Plan:** Apply class weighting during training + heavier data augmentation on the
`improper_mask` class to partially compensate. Will report per-class F1-score separately
in final evaluation, since overall accuracy alone would be misleading with this imbalance.