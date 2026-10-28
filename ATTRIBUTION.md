# Attribution and scope

## Course materials

The baseline notebook structure, starter utility interface, YAML inputs and assignment instructions derive from Linfeng Ye's ECE1512 Project A course materials, University of Toronto, October 2026. `utils.py` identifies provided metric/configuration utilities. The completed notebook implements the model, training and evaluation exercises. Project additions include lazy feature loading, fixed-split runners, validation, attention exports, consistency training, spatial context and reporting.

The original model/training/evaluation definitions are reused, not silently replaced in the submission. Their recorded SHA-256 fingerprint is:

`c7fdc641c63142edf4718e742c5b131ece045aa895eae0beb6a41d757717c89d`

The clean notebook removes saved execution outputs and automatic package installation, loads the final frozen splits and describes the completed extensions. These setup/documentation changes do not alter that fingerprint.

## Research references

The report cites ABMIL, CLAM, DSMIL, TransMIL, DTFD-MIL and the dataset publications. CLAM, DSMIL, TransMIL and DTFD-MIL code and published figures are not copied into this repository. The implemented extensions are not claimed to reproduce those architectures.

Original-paper source checks and bibliographic details are recorded in `provenance/literature_sources.md`. The feature release is credited separately in `DATA.md`.

## Data and licensing

Dataset and course-material rights remain with their respective owners. Raw features and tissue images are not redistributed. No blanket software or dataset license is asserted for this mixed course submission. Confirm source terms and course publication requirements before making the repository public.

## Reporting

The report identifies exploratory test reuse, older C17 patient overlap, incomplete feature provenance and negative findings. There is no watermark. Removing administrative session material does not remove required citations, source attribution or any disclosure required by the course.
