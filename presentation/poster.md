# ABMIL for whole-slide classification

## Author
Parteek Dheri · ECE1512 · University of Toronto

## Takeaway
Gated local context improves mean accuracy on three datasets, but not every metric.

## Introduction
- **Task:** predict a slide label from thousands of patch features, without patch labels.
- **Goal:** complete ABMIL [1], then test a better objective and a local-context architecture.
- **Literature [1–5]:** motivates testing a loss change and a local-context module.

## Pipeline
Frozen patch features: N × 384
Learned projection + ReLU: N × 256
Optional gated local context: 8 neighbors
Gated attention: weighted bag vector
Linear classifier: 2 / 4 / 7 classes

## Methods
- **Data [6–9]:** C16 / C17 / BRACS; 129 / 200 / 86 test slides; frozen splits.
- **C17:** patient-disjoint train/validation/test; centers 3–4 held out for test.
- **Protocol:** 30 epochs; AdamW; 10 paired seeds (42–51); 90 runs. Select first maximum validation AUROC + F1.

## Research ideas
- **Consistency:** full-bag CE + KL(detached full prediction || 90%-retained view). No extra inference parameters.
- **Spatial:** gated neighbor residual before attention; starts at zero to preserve the initial baseline prediction.

## Accuracy caption
Test accuracy (%), mean ± sample SD over ten seeds. The report includes all three methods and all evaluation metrics.

## Difference caption
Spatial minus ABMIL accuracy; points are paired training seeds. Bars show mean ± sample SD, not confidence intervals.

## Attention caption
ABMIL, C17 validation: true class 1, predicted 2. Fixed first class-1 example, seed 42. Coordinate attention is not tumor probability or validated localization.

## Conclusions
- **Spatial:** +2.02 / +2.30 / +1.16 percentage-point mean accuracy changes on C16 / C17 / BRACS.
- **C17:** macro F1 improves by 0.0381; improves in 9/10 seeds.
- **Consistency:** accuracy changes −0.39 / −0.35 / +0.47 pp; no broad advantage.

## Tradeoffs
- BRACS spatial AUROC falls by 0.0081.
- C17 spatial test CE increases: 0.8059 to 1.0075.
- C17 class-1 / class-2 spatial recall remains only 0.1187 / 0.2278.

## Limitations
- Fixed splits and earlier test inspection: exploratory findings, not independent confirmation.
- No final gated shuffled-coordinate control, localization labels or external validation.

## References
[1] Ilse et al. ABMIL. ICML, 2018.
[2] Lu et al. CLAM. Nature BME, 2021.
[3] Li et al. DSMIL. CVPR, 2021.
[4] Shao et al. TransMIL. NeurIPS, 2021.
[5] Zhang et al. DTFD-MIL. CVPR, 2022.
[6] Ehteshami Bejnordi et al. C16. JAMA, 2017.
[7] Bandi et al. C17. IEEE TMI, 2019.
[8] Brancati et al. BRACS. Database, 2022.
[9] Ye. WSI Feature v1. Zenodo 23130543, 2026.
[10] Ye. ECE1512 Lecture 5 and starter code, 2026.

## Footer
Code, full references and complete results: github.com/dheri/ECE1512_Project-A · Baseline adapted from course starter materials [10].
