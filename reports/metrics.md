## Final Model Performance

**Critical fix identified:** Earlier iterations used simple /255 pixel normalization,
which does not match MobileNetV2's expected input format (ImageNet-trained models
require the specific preprocessing scheme used during pretraining). Switching to
`tensorflow.keras.applications.mobilenet_v2.preprocess_input()` resolved this
mismatch and was the single highest-impact fix in this project.

**Final Test Set Results (235 images):**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| mask | 0.89 | 0.85 | 0.87 | 98 |
| no_mask | 0.88 | 0.88 | 0.88 | 113 |
| improper_mask | 0.52 | 0.62 | 0.57 | 24 |
| **Overall Accuracy** | | | **0.84** | 235 |

Predicted class distribution ([93, 113, 29]) closely matches actual distribution
([98, 113, 24]), confirming genuine multi-class learning rather than majority-class
bias observed in earlier iterations.

**Remaining limitation:** improper_mask class shows lower precision (0.52) than
the other two classes, consistent with it having the fewest training examples
(160 images) and being the most visually subtle category to distinguish. This
aligns with our "Existing Gaps" analysis (Section 4) regarding limited
classification of improperly worn masks.