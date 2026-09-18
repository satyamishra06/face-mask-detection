## Model Performance & Limitations

**Final approach:** MobileNetV2 (frozen base) + Focal Loss + oversampling (minority
classes duplicated to 700 samples each in training).

**Result:** Model achieves 79% overall accuracy but this is driven entirely by the
majority `mask` class (recall 1.00). `no_mask` and `improper_mask` recall = 0%.

**Root cause identified:** Diagnostic analysis of raw cropped face images showed
average crop sizes of only 26-48 pixels (mask: 33x36, no_mask: 26x29,
improper_mask: 43x48), sourced from the dataset's group/crowd photographs.
These crops are upscaled 5-8x to the required 224x224 input size, which
destroys the fine-grained visual detail (mask edges, nose/mouth visibility)
needed to distinguish between classes. Combined with severe class imbalance
(86 improper_mask vs 2262 mask training images), the model could not learn
discriminative features for the minority classes regardless of the class
imbalance technique applied (class weighting, oversampling, and Focal Loss
were all tested and produced identical results, confirming the bottleneck
is data resolution/quantity, not training methodology).

**This validates two of our own "Existing Gaps"** (Section 4 of synopsis):
"Real-World Conditions" (camera quality/distance affecting performance) and
"Limited Classification" (improperly worn masks being harder to detect).

**Recommended future fix:** Source a dataset with close-up, higher-resolution
face images per class (e.g., MaskedFace-Net's individual portrait images
rather than cropped-from-crowd images) to provide sufficient pixel detail
for the model to learn from.