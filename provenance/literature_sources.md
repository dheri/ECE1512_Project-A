# Literature source record

Bibliographic and method summaries were checked against the following original publisher pages or author full texts. Accessed October 26, 2026. Published performance values are not treated as comparable to the supplied feature/subset protocol.

| Reference | Primary source | Method details checked |
| --- | --- | --- |
| ABMIL, ICML 2018, PMLR 80:2127–2136 | https://proceedings.mlr.press/v80/ilse18a.html and linked paper PDF | Permutation-invariant bag embedding; learned/gated attention; bag likelihood; attention is not a direct instance probability. |
| CLAM, Nature Biomedical Engineering 5:555–570, 2021 | https://www.nature.com/articles/s41551-020-00682-w and https://arxiv.org/html/2004.09666 | Attention-based pseudo-instance clustering, class-specific attention branches, optional mutual-exclusivity supervision. Publisher page verifies journal citation; open author preprint supplies the full method. |
| DSMIL, CVPR 2021:14318–14328 | https://openaccess.thecvf.com/content/CVPR2021/html/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_CVPR_2021_paper.html and linked PDF | Critical-instance max stream, learned similarity aggregation, averaged stream scores; contrastive extraction and multiscale features are separate contributions. |
| TransMIL, NeurIPS 34:2136–2147, 2021 | https://arxiv.org/html/2106.00908v2 | Two transformer layers, Nyström attention, square token arrangement, class token, PPEG kernels 3/5/7. |
| DTFD-MIL, CVPR 2022:18802–18812 | https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_DTFD-MIL_Double-Tier_Feature_Distillation_Multiple_Instance_Learning_for_Histopathology_Whole_CVPR_2022_paper.html and linked PDF | Pseudo-bags inherit parent labels; noisy positives; two tiers; derived instance scores and MaxS/MaxMinS/MAS/AFS distillation. |
| CAMELYON16, JAMA 2017 | doi:10.1001/jama.2017.14585; references on https://camelyon17.grand-challenge.org/ | Dataset context and citation. The course subset is not represented as the complete original challenge. |
| CAMELYON17, IEEE TMI 38(2):550–560, 2019 | doi:10.1109/TMI.2018.2867350; https://camelyon17.grand-challenge.org/ | Patient-level challenge endpoint differs from this four-class slide experiment. |
| BRACS, Database 2022:baac093 | https://academic.oup.com/database/article/doi/10.1093/database/baac093/6762252 | Citation and original 547-WSI dataset context; available course feature subset has 537. |
| Course features | https://zenodo.org/records/23130543 | Linfeng Ye, WSI Feature, v1, October 4, 2026. Description does not document exact encoder/checkpoint/pretraining or coordinate units. |

Publisher PDFs are not redistributed. No implementation from the four reviewed follow-on architectures was imported. The course starter materials and locally verified experimental outputs are separate sources described in `ATTRIBUTION.md` and the report.
