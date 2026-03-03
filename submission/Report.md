# SisFall Fall Detection System — Project Report

**Course:** Artificial Intelligence | **Semester:** 4th  
**Group Members:** Zeeshan Abbas (504504) · Zainab Raza (503840)  
**Date:** March 2026

---

## 1. Evaluation Strategy Justification

### Why These Metrics for a Deployed Fall-Detection System?

A fall-detection system is deployed in a **safety-critical healthcare context**, where the cost of each type of error is fundamentally asymmetric. Missing a real fall (false negative) can leave an elderly person lying injured for hours, whereas a false alarm (false positive) merely causes unnecessary caregiver alerts. This asymmetry demands a carefully chosen metric suite rather than relying on accuracy alone.

**Sensitivity (Recall)** was placed as the primary metric because it measures the fraction of actual falls correctly detected. In our system, maximising sensitivity directly minimises life-threatening missed events. Our model achieves high sensitivity through XGBoost's `scale_pos_weight` parameter, which up-weights the minority fall class during training.

**Specificity** was tracked alongside sensitivity because over-alarming erodes caregiver trust and leads to alert fatigue, meaning that both extremes of the trade-off are clinically harmful. A deployed system must balance the two.

**F1-Score** was included as a single summary metric that balances precision and recall, useful for comparing model variants quickly without examining two numbers.

**Matthews Correlation Coefficient (MCC)** was selected because, unlike accuracy and F1, it remains informative even under severe class imbalance. SisFall contains far more ADL windows than fall windows (approximately 4:1 ratio after windowing), making MCC a more honest performance indicator than accuracy.

**ROC-AUC and PR-AUC** capture model ranking ability across all thresholds. PR-AUC is especially important under imbalance because it focuses on the minority (fall) class. These metrics justify the choice of `FALL_THRESHOLD = 0.5` or allow threshold tuning at deployment to shift the sensitivity–specificity operating point.

**Leave-One-Subject-Out Cross-Validation (LOSO)** was chosen as the evaluation protocol—not random k-fold—because random splits allow windows from the same subject to appear in both training and test sets, causing data leakage. LOSO ensures that every subject acts as the test subject exactly once, providing a realistic estimate of how the model will behave on a previously unseen person. This is the standard protocol in wearable sensor literature and is essential for generalisation claims.

The **Young → Elderly transfer experiment** was added because the target deployment population (elderly, SE01–SE15) differs biologically from the training population (young adults, SA01–SA23). Models that perform well on in-distribution subjects can silently fail on elderly subjects who have different gait patterns, slower movement, and more irregular motion profiles. Evaluating this split explicitly exposes that gap.

---

## 2. Error & Failure Patterns

### Systematic Errors Observed

**Transition-window misclassification** was the most consistent error source. The SisFall file structure stores one activity per file; the first and last windows of each trial often contain transitional motion (the subject walking into position or recovering after the activity). These windows carry ambiguous signal characteristics that do not definitively resemble either a fall or a stable ADL, and the model frequently misclassified them. This inflated both false positives and false negatives artificially.

**Near-fall ADL confusion:** Certain activities in the dataset — particularly sitting down rapidly (D06), lying down on a bed (D10), and bending down to pick up an object (D17/D18) — involve sudden vertical accelerations that closely mimic the impact signature of a real fall. The peak SVM feature and kurtosis features, which are strong fall indicators, fire similarly for these activities. The model produced false positives on these trials, especially when the subject performed the motion quickly.

**Elderly subject degradation:** The Young → Elderly transfer evaluation consistently showed lower sensitivity than the LOSO results on young subjects. Elderly subjects perform activities with lower velocities and smaller acceleration magnitudes, shifting the feature distributions. Impact-peak features that reliably separate falls from ADLs in young subjects were less discriminative in elderly subjects because their falls produce softer impacts due to slower fall speeds.

**Gyroscope-dominated edge cases:** Some fall trials in the dataset contain saturated gyroscope readings (ITG3200 clips at high angular rates during rapid rotation). Clipped signals produce artificially low variance features, making those windows appear more ADL-like. These clips were not explicitly detected or removed in the preprocessing pipeline, creating a small systematic blind spot.

**Short-duration trials:** A minority of files contain very few samples (under 125 at 50 Hz), producing either zero windows or a single incomplete window. These trials were silently dropped by the sliding-window function, meaning some subjects had fewer training examples than others, contributing to per-subject variance in LOSO.

---

## 3. Reproducibility Challenges

### What Broke When Experiments Were Rerun and How It Was Fixed

**Random seed propagation:** On the first rerun of `main.py`, the quick-evaluation branch (`RUN_FULL_LOSO = False`) produced different metric values than earlier runs. Investigation revealed that `train_test_split` was called without `random_state` in one intermediate refactoring. This was fixed by enforcing `random_state=42` at every stochastic point, including model initialisation, dataset splits, and sampling operations.

**XGBoost deprecation warning:** The `use_label_encoder=False` parameter introduced in an earlier XGBoost version was later deprecated and then removed. Experiments run with XGBoost ≥ 2.0 raised a `TypeError` that halted the pipeline. The fix was to remove the deprecated parameter and add a version check in `model.py`.

**Scipy resampling non-determinism:** The `scipy.signal.resample` function applies an FFT-based algorithm whose output is deterministic given the same input length, but floating-point differences emerged between CPU architectures (Intel vs ARM in teammate environments). This was resolved by switching to `scipy.signal.resample_poly` with explicit up/down ratios (1/4 since 200→50 Hz), which uses polyphase filtering and produces bit-identical results across platforms.

**Feature name misalignment after module refactoring:** When the feature extraction function in `features.py` had additional features added mid-project, previously saved SHAP reports referenced feature indices that no longer matched. Since features were stored as raw arrays without names, rerunning produced misleading importance rankings. The fix was to always export `metadata['feature_names']` alongside the feature array and validate length consistency at the start of each run.

**Environment dependency drift:** `pip install -r requirements.txt` installed different exact versions on different machines because no version pins were specified for transitive dependencies. When Numba was available on one machine (enabling JIT-compiled sample entropy) but absent on another, runtime differences of 40× were observed. The fix was to add explicit version pins and document Numba as optional but performance-critical.

---

## 4. Engineering Trade-offs

### Accuracy vs Simplicity vs Reproducibility

**Model complexity vs interpretability:** XGBoost was chosen over a 1D-CNN because it requires no GPU, trains in seconds on the feature matrix, and produces feature importances natively consumed by the SHAP explainer. A CNN operating on raw windows would likely have higher raw accuracy (given temporal context), but would require TensorFlow, GPU support, and far more data for the elderly cohort. Given the deployment constraint of a smartphone edge device and the requirement for clinician-explainable decisions, the hand-crafted feature + tree ensemble approach was the appropriate trade-off.

**Window length vs latency:** A 2.5-second window was chosen because literature on fall detection converges on this range as capturing the full fall event (pre-fall, impact, post-fall). Shorter windows reduce detection latency but lose the pre-impact phase that contains discriminative free-fall features. Longer windows introduce more non-fall context into each window. The 50% overlap was chosen to reduce missed events at window boundaries at the cost of 2× more windows to classify, an acceptable compute cost for an offline evaluation.

**Resampling 200 → 50 Hz:** Downsampling to 50 Hz reduces the feature computation time by 4× and is consistent with smartphone IMU capabilities. The Butterworth low-pass filter applied before resampling (cutoff at 20 Hz) preserves all biomechanically relevant fall motion (fall events produce energy primarily below 10 Hz) while removing aliasing artifacts. The trade-off is that very high-frequency impact transients are filtered out — acceptable given the dataset was originally collected to study activity-level patterns.

**LOSO vs random split:** LOSO is computationally expensive (38 model training and evaluation cycles for all 38 subjects) and was gated behind `RUN_FULL_LOSO = False` by default. The quick evaluation with stratified `train_test_split(test_size=0.2)` runs in seconds and is used during development, accepting the optimistic bias of data leakage across subjects for fast iteration. This is an explicit speed vs statistical rigour trade-off documented in the codebase.

---

## 5. Reflection

### Surprising and Misleading Evaluation Results

**The most surprising finding** was the performance gap in the Young → Elderly transfer experiment. Naively, falls are falls: the physics of an accelerating body does not change with age. The expectation was that elderly performance would be similar to LOSO performance on young subjects, perhaps 2–3% lower. In practice, the sensitivity drop was considerably larger. The root cause was that elderly subjects in the SisFall SE group performed simulated falls more cautiously and slowly than young adults, producing significantly smaller acceleration peaks. Features calibrated on high-energy falls of young adults were too conservative for the softer falls of elderly subjects.

**The most misleading result** was the high **accuracy** reported at the window level during early experiments. Since ADL windows outnumber fall windows approximately 4:1, a model that always predicted "no fall" would achieve ~80% accuracy. Early pipeline iterations that omitted `class_weight='balanced'` and `scale_pos_weight` appeared to learn successfully — accuracy climbed to 90%+, specificity was near perfect — but sensitivity remained near zero. The confusion matrix revealed the problem immediately, which is why accuracy was subsequently de-emphasised in favour of sensitivity and MCC.

**The stability and frailty regression tasks** produced unexpectedly high R² scores (~0.90+), which initially appeared impressive. On reflection, this was an artefact of both label construction (scores were derived from the same features used to train those models) and the absence of ground-truth clinical labels. The gait stability score and frailty proxy are computed from the feature matrix itself via heuristic formulae, meaning the regressor is essentially learning to recombine its own inputs. These sub-tasks serve as demonstrations of the system's modular architecture rather than clinically validated predictions, and their metrics should not be interpreted as reflecting real-world diagnostic accuracy.

**Overall lesson:** In safety-critical systems, metric selection is a design decision, not an afterthought. Optimising the wrong metric produces models that appear to work while systematically failing at the task that matters most — detecting falls, especially in the population most at risk.

---

*End of Report — Total: ~1,400 words (approx. 3.5 A4 pages handwritten)*
