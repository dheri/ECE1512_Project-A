# Reproducibility checklists

## Report artifact

- [x] Define bag/instance supervision and distinguish slide prediction from localization.
- [x] Review the five required methods using original papers.
- [x] Attribute the two adaptations to relevant consistency, masked-MIL, spatial-graph and gating literature; distinguish their implementations from prior methods.
- [x] Include original, reproducible architecture figures for the baseline and both adaptations.
- [x] State available feature counts, dimensions, split sizes and uncertain provenance.
- [x] Describe the encoder/aggregator/classifier and all actual training settings.
- [x] Report baseline accuracy, macro AUROC, macro F1, test CE and loss curves.
- [x] Show all nine test confusion matrices with unique class sample sizes and ten-seed mean row percentages; verify each saved matrix against predictions.
- [x] Specify two research proposals and evaluate both on three datasets.
- [x] Report all ten primary seeds, paired differences and sample SD, not only a best run.
- [x] Separate historical and primary C17 cohorts.
- [x] Disclose prior test inspection, missing controls and attention limitations.
- [x] Include deterministic attention examples without a localization-accuracy claim.
- [x] Provide editable prose, figure sources and a PDF within the 20-page limit.
- [x] Include the GitHub repository URL: https://github.com/dheri/ECE1512_Project-A
- [ ] Complete the author's final review and confirm any course-specific disclosure requirements.

## Code artifact

- [x] Include completed notebook, baseline pipeline, both extensions and tests.
- [x] Preserve original model/training/evaluation definition fingerprint.
- [x] Supply ordered frozen splits and the C17 patient-group record.
- [x] Package all primary predictions, histories and selected-epoch records.
- [x] Package historical results separately, with cohort labels.
- [x] Include nine seed-42 representative checkpoints, with unchanged parameter tensors.
- [x] Provide feature acquisition instructions and exact feature/split fingerprints.
- [x] Record dependency versions and GPU/CPU execution settings.
- [x] Regenerate tables and analytical plots without training.
- [x] Provide a manifest-verified submission archive builder and separate report PDF.
- [x] Recompute saved metrics and verify representative artifact/source hashes.
- [x] Run synthetic workflows without raw features; skip the clearly labeled optional real-data test when its files are absent.
- [x] Require explicit confirmation and a new output directory before reproduction training.
- [ ] Bundle the remaining checkpoints, if required by the final submission policy. They remain in the original experiment archive.
- [ ] Verify a fresh full 90-run reproduction in the recipient's environment. No extra scientific training was performed while preparing this package.

## Validation scope

The current report source is `report/report_lecture.md`; the earlier longer draft in `report/report.md` is retained for reference. Poster content, speaking notes and Q&A preparation are under `presentation/`. The poster uses the same primary results and fixed attention selection as the report; it is not a new experiment.

The saved-result audit does not rerun neural inference on all original checkpoints. It verifies predictions, metrics, selected epochs, fingerprints and the representative checkpoint files. The original completed experiment collection was verified before packaging. CPU synthetic regression tests exercise small generated bags, not an additional experimental training cohort.

Feature acquisition, image-level feature extraction, a pristine external test cohort, patient-level confidence intervals, calibration analysis and localization ground truth are outside this submission's completed evidence.
