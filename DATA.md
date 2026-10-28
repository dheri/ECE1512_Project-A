# Data and provenance

## Feature release

Linfeng Ye, **WSI Feature**, version 1, Zenodo, October 4, 2026:
https://zenodo.org/records/23130543

Acquire C16.zip, C17.zip and BRACS.zip directly from the source under its terms. The archives total about 14.3 GB. No new download is necessary if you already have the matching files. Place the extracted HDF5 files as follows:

```text
feature/C16/patch_feats_pretrain_medical_ssl.h5
feature/C17/patch_feats_pretrain_medical_ssl.h5
feature/BRACS/patch_feats_pretrain_medical_ssl.h5
```

These large files and original tissue images are intentionally excluded from Git. This repository does not grant a redistribution license for course data or source datasets.

Each slide group has a `feat` array shaped `[N,384]`, a `coords` array shaped `[N,2]`, and integer `label` attribute. Features are frozen. The release description does **not** establish the exact image encoder, checkpoint, pretraining corpus, magnification, patch footprint or coordinate units. Do not infer a ViT/DINO backbone from the feature dimension or filename. Optional tissue overlays assume a level-0 coordinate frame and patch footprint; those assumptions are not verified by the release.

## Exact HDF5 SHA-256 fingerprints

| Dataset | Slides | SHA-256 |
| --- | ---: | --- |
| C16 | 398 | `f22c22c4dc7c5314a2b4b16b8322f8c9cab8e8330d0942a5ab0cbf4f7b27fd6f` |
| C17 | 499 | `6c9dd83f6af398773a45da57e4f1f69d6269ecfa74821d8d4f72799f92b86e42` |
| BRACS | 537 | `72a787e628f30a8d440fb698bd2de05edfe82e082c2bb6865c5aea4dd8e8e5bc` |

Use `sha256sum feature/*/*.h5` to compare your local files. This reads the complete files and can take time.

## Frozen primary partitions

Ordered membership is stored in `reference_splits/{C16,C17,BRACS}/splits.json`. Its fingerprint hashes compact, sorted-key JSON and includes list ordering.

| Dataset | Train / validation / test | Split fingerprint |
| --- | --- | --- |
| C16 | 242 / 27 / 129 | `1e78888dc9c3339e7132d1a18e8776cc36c12f31315bd599720378fd12925acb` |
| C17 | 269 / 30 / 200 | `341f834431027671ae92e8084af1134d2139c74a23ed7b185a43902c65b652e8` |
| BRACS | 389 / 62 / 86 | `6da8b481d706c885a9cac45f35fc7b50e823c7d57560a11ece4210d254ae1d81` |

C17 centers 3 and 4 form the fixed test partition. Development centers 0–2 use StratifiedGroupKFold with ten folds, shuffle enabled and seed 42. The first all-class fold is fold 0. Its patient groups are 54/6/40 with zero overlap. Test labels and test metrics did not select the partition. `reference_splits/C17/patient_partition.json` records the rule and patient groups. Labels remain numeric because their exact mapping to diagnoses has not been independently verified.

C16 keeps the available test slides and frozen development split. BRACS preserves the available split metadata. Patient-group independence is not claimed for datasets without comparable grouping metadata.

## Metadata

`dataset_csv/camelyon17.csv` contains pseudonymous slide IDs, center, case ID and labels. Labels are read from HDF5 during training. Center assignments were cross-checked against the CAMELYON17 training-file manifest: patients 000–019/020–039/040–059/060–079/080–099 correspond to centers 0/1/2/3/4.

`dataset_csv/bracs.csv` contains the available 537 slides and inherited `split_info`. All available slide IDs and labels matched the supplied source metadata. Ten metadata rows without supplied feature bags are omitted. The original BRACS publication describes 547 whole slides.

## Older C17 cohort

The first 90 historical runs used a slide-level development split with 27 patients shared by training and validation. The test patients were still disjoint. Its split fingerprint is `185395475d66095bf4d0419210a3b4f30cc8f4c6a82883ea3f7fe4855d154169`.

The later 18 class-weighting/control runs and all 90 primary runs use the corrected patient-disjoint partition. Historical results are labeled by cohort and must not be pooled as additional primary seeds.

## Original datasets

- CAMELYON16: Bejnordi et al., JAMA 2017, doi:10.1001/jama.2017.14585.
- CAMELYON17: Bandi et al., IEEE TMI 2019, doi:10.1109/TMI.2018.2867350; https://camelyon17.grand-challenge.org/
- BRACS: Brancati et al., Database 2022, doi:10.1093/database/baac093.

The feature subset and slide-level experiments are not a reproduction of the official challenge evaluation protocols.
