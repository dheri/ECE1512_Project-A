# Final poster and oral presentation

## Files

- [Submission image](ECE1512_ProjectA_Poster.png): 5120 × 2880 PNG, below 10 MB.
- [Reading PDF](ECE1512_ProjectA_Poster.pdf): one-page vector poster.
- [Editable poster content](poster.md).
- [Speaking notes](speaking_notes.md): three balanced parts, with a provisional five-minute rehearsal plan.
- [Q&A preparation](qa_notes.md).

The course documents do not specify a talk-duration limit. Confirm the allotted time before rehearsal. No separate slide deck is required by the supplied instructions or included here.

## Requirements used

Primary assignment guidance: `Lecture5_ProjectA.pdf`, especially Assignment Outlines 1–3 and Deliverables.

Poster layout: `20260831_ECE1512_PosterTemplate.pdf` supplies Introduction, Methods, Results, Conclusions and Bibliography. Its 4:3 print dimensions are adapted to the lecture's explicit 16:9 digital-poster limit.

Presentation assessment: `2026F_ECE1512-Poster-Presentation-Review-Evaluation.pdf` covers content, motivation/goals, system overview, results, conclusions, organization, readable/cited visuals, delivery, balanced speaking and Q&A. The notes address these criteria without treating delivery as something a file can verify.

The lecture gives December 4, 2026 for presentation materials and December 16 for the presentation. The poster must be no larger than 5120 × 2880 pixels and 10 MB. Original course PDFs remain outside this repository.

## Rebuild without training

From the repository root, with the documented Python plotting environment and isolated PDF renderer:

```bash
python scripts/make_poster_figure.py
python scripts/build_poster.py \
  --pdfmake "$(pwd)/tmp/pdf_runtime/node_modules/pdfmake" \
  --output "$(pwd)/outputs/ECE1512_ProjectA_Poster.pdf"
```

Use `--node /absolute/path/to/node` if Node is not on PATH. The builder checks one-page pagination, font glyph support, the PDF signature, exact PNG dimensions and a conservative 10,000,000-byte image limit. It copies the finished PDF and PNG here. The figure uses existing paired results only; no training, feature download or inference occurs.

Full references and all three methods' metrics are in the [current report source](../report/report_lecture.md). Earlier exploratory cohorts are not pooled into the poster's primary comparison. Check repository access for evaluators and review the files before submitting; nothing has been uploaded to Quercus.
