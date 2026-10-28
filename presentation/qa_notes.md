# Project A: likely questions and short answers

## What did you implement?

The missing ABMIL model/training/evaluation parts in the starter notebook, a reproducible experiment pipeline and two extensions: masked-view prediction consistency and gated local spatial context. The starter structure and utilities remain attributed to the course.

## Why MIL instead of ordinary image classification?

A whole slide has many patches but only a slide label. MIL handles the variable-size bag without pretending that every patch has the slide's diagnosis. The supplied patch features are already extracted.

## What do the projection, attention and classifier each do?

The projection adapts the frozen features. Attention forms a slide-dependent weighted bag representation. The classifier maps that representation to class logits.

## Which encoder did you use?

The supplied release provides 384-dimensional features, but does not identify the exact encoder, checkpoint, magnification or coordinate units. I do not infer a specific backbone from the dimension or filename.

## How do the five reviewed methods differ?

ABMIL learns pooling; CLAM adds clustering constraints; DSMIL relates patches to a critical instance; TransMIL models correlations and position; DTFD-MIL distills pseudo-bag features into a second MIL tier. This project implements none of the latter four architectures.

## Why should masking preserve the diagnosis?

It is a hypothesis for modest random deletion, not a guarantee. Rare diagnostic evidence may be removed. The mixed result could reflect a weak regularization signal or an incorrect invariance assumption; the experiment does not establish which explanation is decisive.

## What exactly is the consistency loss?

Full-bag CE plus KL from the detached full-bag distribution to the masked distribution, with weight and temperature one. There is no EMA teacher or hard-label masked CE. The two gradient contributions accumulate before one optimizer step.

## Is the spatial gate the same as ABMIL's gate?

No. ABMIL's tanh/sigmoid gate computes attention scores. The added scalar sigmoid gate controls a neighborhood residual before attention. The final context layer starts at zero, preserving baseline logits at initialization.

## Why eight neighbors?

Eight is the fixed local-neighborhood setting used in this study, not a demonstrated optimum. The model uses up to eight other patches, fewer for small bags, and zero neighbor context for a singleton. No physical neighborhood radius is claimed.

## Does the gain prove that geometry helps?

No. The final gated model lacks a matched shuffled-coordinate control. Extra capacity or regularization could explain part of the gain. Earlier shuffled controls used a different, ungated architecture and were mixed.

## Is the comparison fair?

Splits, seed grid, optimizer, epochs, checkpoint selection and metrics match. Compute does not: consistency adds a pass, while spatial context adds parameters and neighborhood processing. It is an equal-epoch comparison, not an equal-runtime comparison.

## How did you prevent patient leakage?

The primary C17 split groups patients: 54/6/40 train/validation/test patients with zero overlap; centers 3–4 are held out for test. C16 and BRACS use frozen splits, without claiming independently verified patient grouping where metadata is insufficient.

## Why ten seeds?

Training initialization and stochasticity change the result. Within this same cohort, the first three C17 seeds suggested a −1.33 pp spatial accuracy change; ten seeds give +2.30 pp. More seeds help describe training variation, not new-patient generalization.

## Are the error bars confidence intervals?

No. They are sample SDs across ten paired training seeds on fixed test slides. No statistical-significance claim or patient-level interval is reported.

## What is the strongest result?

Spatial context increases mean accuracy on all three datasets. C17 also improves macro AUROC and F1; F1 increases in nine of ten seeds. The claim remains conditional on the available features, fixed splits and tested protocol.

## Why does C17 cross-entropy worsen when accuracy improves?

Accuracy uses the largest class probability; CE uses the probability assigned to the true class. More correct argmax decisions can coexist with larger probability errors on some slides. No calibration study was performed.

## Why are C17 accuracy and macro F1 so different?

Classes 0 and 3 account for 166/200 test slides. Minority recalls are poor: spatial class-1 and class-2 recall are only 0.1187 and 0.2278. Macro F1 gives each class equal weight, exposing errors that overall accuracy can hide.

## Does attention identify tumors?

It identifies relatively influential patches for the model. The figures show coordinates, not tissue overlays or annotated lesions. Without region-level ground truth, neither localization accuracy nor causal faithfulness is established.

## Was the attention example cherry-picked?

No. It is seed 42's first saved validation example with the specified class label: C16 = 1, C17 = 1, BRACS = 6. Selection does not depend on correctness, confidence or attention. The poster uses the fixed C17 baseline example.

## What happened to the earlier 108 experiments?

They remain separate supporting evidence. The first 90 historical runs had C17 train/validation patient overlap; the later 18 weighting runs and all 90 primary runs use corrected patient-disjoint partitions. Historical runs are not pooled as extra primary seeds.

## Was this an untouched-test confirmatory study?

No. Earlier test inspection informed follow-up designs. The ten-seed comparison is exploratory and estimates training-seed variation more reliably; it does not restore an untouched test set.

## What would you test next?

A matched gated shuffled-coordinate/capacity control, independently held-out patients or centers and probability-quality analysis would address the main unresolved claims. These are proposals, not completed experiments.
