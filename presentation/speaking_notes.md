# Project A poster presentation

Parteek Dheri

## Rehearsal plan

Use three balanced parts: problem and baseline; proposed methods; results and conclusion. The supplied documents do not give a talk-duration limit. The five-minute plan below is a rehearsal target, not a course requirement. Adjust it when the presentation slot is confirmed.

This version assumes one presenter. If presenting in a group, confirm the actual members and divide the three parts evenly; do not invent collaborators.

## Part 1 · Problem and baseline · approximately 0:00–1:40

Point to **Introduction**, then follow the top of **Methods**.

“This project classifies whole-slide pathology images using only slide-level labels. A slide contains many patches, so the challenge is to retain diagnostic evidence without patch annotations.

“I completed the supplied ABMIL pipeline. Frozen 384-dimensional patch features pass through a learned 256-dimensional projection. Gated attention weights the patches, and a linear classifier predicts the slide label.

“The literature suggests different ways to improve this: CLAM adds instance constraints; DSMIL relates patches to a critical instance; TransMIL models correlations; DTFD-MIL forms pseudo-bags and distills features. I used these ideas to ask two narrower questions rather than reproduce all four architectures.

“The baseline mean accuracies are 87.67% on C16, 77.45% on C17 and 42.21% on BRACS. C17 macro F1 is only 0.5035, so accuracy alone hides weak minority-class performance.”

Transition: “I tested one change to the training objective and one to the patch representation.”

## Part 2 · Two ideas and controlled evaluation · approximately 1:40–3:20

Point to **Two research ideas**, then return to the protocol.

“The first idea is prediction consistency. I retain 90% of a bag and encourage its prediction to match the detached full-bag prediction. Training combines full-bag cross-entropy with a unit-weight KL term. The model architecture and inference procedure are unchanged.

“The second idea is local spatial context. For each patch, I average up to eight neighboring features, concatenate this with the original feature and pass it through a gated residual module before attention. The residual starts at zero, so initial predictions match the paired baseline.

“Both extensions use the same saved splits and metrics as ABMIL. There are ten paired training seeds for each of three methods on three datasets: 90 runs, each completing 30 epochs. Checkpoints maximize validation AUROC plus F1, never a test metric.

“C17 has patient-disjoint training, validation and test sets. Equal epochs do not mean equal computation: both extensions cost more during training.”

Transition: “The spatial model gives the clearer gain, but the negative findings are important.”

## Part 3 · Results, caveats and conclusion · approximately 3:20–5:00

Point to **Results**, then **Tradeoffs**, and finish at **Conclusions**.

“Gated spatial context improves mean accuracy by 2.02, 2.30 and 1.16 percentage points on C16, C17 and BRACS. The dots show the paired seed differences; not every seed improves. The error bars are sample standard deviations, not confidence intervals.

“C17 macro F1 improves by 0.0381 and is higher in nine of ten seeds. However, its cross-entropy gets worse, and minority-class recall remains poor. BRACS AUROC also falls. So I would not describe the extension as uniformly better.

“Consistency gives small, mixed changes and no broad advantage at the tested settings. That is a useful negative result, not a reason to hide the method.

“The attention example shows candidate patch importance at coordinates. It is a fixed validation example, not a best-looking slide, and is not validated tumor localization.

“My conclusion is that a small spatial residual is promising on these features, but metric tradeoffs, fixed splits and earlier test inspection limit the claim. The main deliverable is a complete, auditable MIL pipeline and a controlled comparison.”

## Delivery checklist

- Open with the task, not the implementation history.
- Follow the three parts; point to a visual only when explaining it.
- Say “percentage points” for accuracy differences.
- Say “training-seed variation,” not “patient confidence interval.”
- Keep eye contact, use a steady voice and pause between parts.
- End with the conclusion, not a list of unfinished experiments.
- For questions, answer first; then point to the supporting table or limitation.
