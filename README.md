# ECE1512 Project A

Parteek Dheri

ABMIL baseline evaluation on CAMELYON16, CAMELYON17 and BRACS, with full-to-masked prediction consistency and gated local spatial context.

## Start here

- **Report:** [PDF](report/ECE1512_ProjectA_Report.pdf), [editable source](report/report.md), organized under Lecture 5's three assignment sections
- **Main evidence:** [90-run table](results/primary/all_runs.csv), [paired differences](results/primary/paired_comparison.csv)
- **Completed notebook:** [ECE1512_ProjectA_Working.ipynb](ECE1512_ProjectA_Working.ipynb)
- **Data acquisition and provenance:** [DATA.md](DATA.md)
- **Separate report/code checklists:** [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
- **Attribution:** [ATTRIBUTION.md](ATTRIBUTION.md)

Repository: [dheri/ECE1512_Project-A](https://github.com/dheri/ECE1512_Project-A). This is the prepared submission copy; it has not been uploaded to Quercus. Complete the author's final review and ensure graders can access the repository before submitting.

This repository is the **report-and-code submission only**. `report/report.md` is the sole report source. Presentation materials, superseded report drafts and unused draft figures/tables are maintained separately and are not included here.

## Study

The primary study contains 90 fresh runs: 3 methods × 3 datasets × 10 paired seeds (42–51), each completing 30 epochs. The checkpoint is the **first maximum validation macro AUROC + macro F1**. Every seed is reported. Test evaluation uses full bags and ordinary cross-entropy.

| Dataset | ABMIL accuracy | Consistency-90 accuracy | Gated spatial accuracy |
| --- | ---: | ---: | ---: |
| C16 | 87.67 ± 3.38% | 87.29 ± 3.19% | 89.69 ± 2.81% |
| C17 | 77.45 ± 4.65% | 77.10 ± 4.22% | 79.75 ± 2.72% |
| BRACS | 42.21 ± 2.74% | 42.67 ± 3.05% | 43.37 ± 3.38% |

Values are means ± sample SD, not confidence intervals. Spatial context improves mean accuracy, but BRACS AUROC decreases and C17 test cross-entropy increases. Consistency changes are small and mixed. Minority-class C17 recall remains poor.

All primary C17 partitions are patient-disjoint. The earlier 108 runs are kept under `results/historical/`, not pooled with the primary study. The first 90 historical runs used overlapping C17 training/validation patients; the subsequent 18 weighting runs used the corrected partition. Earlier test inspection informed follow-up designs, so the final comparison remains exploratory.

## Contents

```text
report/                  Editable report, PDF, source figures and generated tables
reference_splits/        Ordered final split lists and reference configurations
dataset_csv/             Pseudonymous slide metadata used by the loader
results/primary/         Results, histories and saved predictions for all 90 primary runs
results/historical/      Results and test predictions for 108 earlier supporting runs
checkpoints/             Nine representative seed-42 checkpoints and their hashes
provenance/              Source and artifact fingerprints
checks/                  Saved audit and validation summaries
scripts/                 Figure generation, report/ZIP builds and result audit
tests/                   Synthetic model, objective and workflow tests
reproduce.py             Portable, explicit new-training entry point
```

Raw feature files, original WSIs, training logs, remote launch scripts and session notes are not included. The remaining 81 primary checkpoints and historical checkpoints are retained in the original experiment archive, not duplicated here. Saved predictions support metric verification without them.

The saved results, configurations, split records and representative checkpoints support the report's claims and reproduction commands. Historical results support the explicitly separated earlier-controls discussion; they are not extra primary seeds. All report figures are used in the PDF, in the six-example attention audit, or as larger versions of the nine-panel confusion comparison.

### Why `checks/` is included

These small generated records document completed verification, not additional experiments:

- `result_audit.json`: saved metrics, split separation, checkpoints and source fingerprints.
- `confusion_validation.json`: all 90 confusion matrices checked against saved predictions.
- `pdf_validation.json`: final report source, page count and PDF fingerprint; also used by the ZIP builder.
- `test_validation.json`: test results and the optional real-feature skip.

## Environment

The scientific runs used Python 3.12.13, PyTorch 2.5.1+cu121, an NVIDIA GTX 1080 Ti, two CPU threads, zero loader workers and one serial GPU worker. Core dependency versions are in `requirements.txt`. The portable package was also checked in the local CPU container with Python 3.12 and PyTorch 2.14.1.

Use a separate environment. Install the PyTorch build appropriate to your hardware before installing the remaining requirements. Do not assume a newer CUDA wheel supports this GPU. Optional original-WSI overlays require OpenSlide, which is not needed for coordinate maps, saved-result auditing or training on features.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# Install a CPU or CUDA PyTorch wheel appropriate to your system.
python -m pip install -r requirements.txt
```

## Verify and regenerate without training

Run from the repository root:

```bash
python -m unittest discover -s tests -v
python scripts/audit_results.py
python scripts/make_figures.py
python scripts/make_architecture_figures.py
python scripts/make_confusion_figures.py
```

The real-feature forward test skips clearly when external course features are absent. The included synthetic workflows remain enabled. The audit recomputes both validation and test metrics for every primary run, checks all selected epochs, ordered partitions, paired rows, sample SDs, source fingerprints and representative checkpoint hashes. It also recomputes test metrics for the 108 historical runs. It needs no raw features and performs no scientific training.

To rebuild the reading PDF, use Node 22 and an isolated renderer install. This does not change the training environment:

```bash
npm install --prefix tmp/pdf_runtime pdfmake@0.2.20
python scripts/build_report.py \
  --pdfmake "$(pwd)/tmp/pdf_runtime/node_modules/pdfmake" \
  --output "$(pwd)/outputs/ECE1512_ProjectA_Report.pdf"
```

The builder copies the verified PDF into `report/`. It checks the PDF signature, deliberate ten-page layout and the 20-page limit when `pdfinfo` is available. The architecture figure script draws the baseline and both adaptations. The confusion figure script validates all 90 saved test matrices against predictions, checks the reported accuracies, and generates the nine-panel comparison plus larger dataset-specific versions. Its `n` labels count unique slides per true class; cells show ten-seed mean row percentages, not an ensemble or ten times as many independent slides. These scripts need no features, inference or training.

## Reproduce training

First obtain the exact feature files described in `DATA.md`. The default command starts the full fresh 90-run grid, in seed → dataset → method order:

```bash
python reproduce.py \
  --feature-root "$(pwd)/feature" \
  --output "$(pwd)/results/reproduced_tenseed" \
  --device cuda --confirm-training
```

The output directory must not exist. Omitting `--confirm-training` fails before a run starts. No checkpoint is selected using test metrics. Model/training definitions and extension implementations are reused from the study; execution does not require SSH, a named host or remote controller state.

A smaller explicitly labeled reproduction is possible:

```bash
python reproduce.py --datasets C16 --methods baseline consistency90 spatial_gated \
  --seeds 42 --output "$(pwd)/results/reproduced_C16_seed42" \
  --device cuda --confirm-training
```

That command is three new runs, not a replacement for the reported ten-seed results. CPU mode is supported but can be slow. Exact bitwise reproduction across different PyTorch/CUDA/SciPy versions and hardware is not promised.

## Representative attention export

Nine seed-42 checkpoints preserve all original parameter tensors. Only embedded path configuration was made portable; the manifest records source, packaged-file and tensor hashes. The helper supplies the matching frozen split list:

```bash
python scripts/export_representative.py --method spatial_gated --dataset C17 \
  --features feature/C17/patch_feats_pretrain_medical_ssl.h5 \
  --output outputs/spatial_C17_seed42_attention
```

The helper creates a temporary checkpoint/split directory and never changes the packaged checkpoint. Source PNGs used by the report are already included, so rebuilding the report does not rerun inference.

## Editing for review

Edit `report/report.md` for the report's prose, citations and layout. Tables are generated from the saved results; do not hand-edit numerical values. Rebuild figures if their presentation changes, then rebuild the PDF and rerun the audit. The `<!-- page -->` and column markers define the editable page layout. Required citations and course-source attribution should remain intact.

`python scripts/package_manifest.py` verifies the current package fingerprints. After an intentional reviewed edit, refresh them with `python scripts/package_manifest.py --write`.

Before final submission: review the report, confirm any course-specific disclosure requirements, check repository access for graders, and rerun the report/package checks. The repository URL is included in the report; Quercus submission remains pending.

## Build submission files

After the final checks and an intentional manifest refresh, create separate report and code files:

```bash
python scripts/package_manifest.py
python scripts/build_submission.py --output-dir outputs/submission
```

The builder refuses to overwrite existing delivery files. It verifies every package fingerprint, rejects presentation artifacts and legacy report sources, copies the final report, builds `ECE1512_ProjectA_Code.zip`, and checks every ZIP member against its source hash. The ZIP contains this repository's packaged files, not `.git`, raw features, credentials, temporary files or new training outputs. Keep the separately supplied report PDF as the report submission. Presentation materials are excluded.
