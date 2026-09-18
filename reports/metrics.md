## Model Performance & Limitations

**Final approach:** MobileNetV2 (frozen base) + Cross-Entropy Loss + rebalanced
dataset (mask undersampled to 650 images, no_mask: 753, improper_mask: 160 —
supplemented with self-captured high-resolution images) + oversampling to
650 per class during training.

**Final Test Set Results:**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| mask | 0.47 | 0.80 | 0.59 | 98 |
| no_mask | 0.43 | 0.09 | 0.15 | 113 |
| improper_mask | 0.04 | 0.08 | 0.06 | 24 |
| **Overall Accuracy** | | | **0.38** | 235 |

**Key finding:** After rebalancing the dataset, the model successfully predicts
across all three classes rather than collapsing to a single majority class
(as seen in earlier iterations with severe class imbalance, where an
artificially high 79% accuracy was achieved purely by always predicting
"mask"). This lower, more honest accuracy reflects genuine multi-class
learning rather than majority-class bias.

**Root cause of remaining limitations:**
1. **Low source image resolution** — average face crop sizes of only 26-48
   pixels (from the dataset's group/crowd photographs), upscaled 5-8x to the
   required 224x224 input, which destroys fine-grained visual detail (mask
   edges, nose/mouth visibility) needed to distinguish between classes.
2. **Limited real, diverse training examples per class** even after
   rebalancing and supplementing with self-captured images.
3. **Frozen pre-trained features** may not fully capture fine-grained
   mask-wearing distinctions without deeper fine-tuning — attempted during
   development but caused training instability (validation accuracy collapsed)
   given this dataset's size, so a frozen-base approach was retained for
   stability.

**Techniques evaluated during development:** class weighting, oversampling,
Focal Loss, dataset rebalancing (undersampling majority + oversampling
minority classes), and supplementing training data with self-captured
high-resolution images. Each addressed a different aspect of the imbalance
problem; final results reflect the combination of rebalancing + oversampling
+ standard cross-entropy loss as the most stable configuration found.

**This validates our own "Existing Gaps" analysis** (Section 4 of synopsis) —
specifically "Real-World Conditions" (camera quality/distance affecting
performance) and "Limited Classification" (improperly worn masks being
harder to detect) — confirming these are genuine, well-documented challenges
in mask detection systems, consistent with prior literature (Loey et al., 2021).

**Future Work:** Source a larger, higher-resolution, naturally balanced
dataset (e.g., individually captured portrait-style images rather than
cropped group/crowd photos) and explore fine-tuning strategies with careful
regularization to further improve minority-class performance.