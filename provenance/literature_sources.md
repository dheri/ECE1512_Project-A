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

## Extension prior-art checks

The following primary sources were checked during the report revision to ground the implemented extensions in prior work. This is a related-work attribution, not a claim that the experiments reproduced these papers or that these sources were followed before implementation.

| Reference | Primary source | Relationship and important differences |
| --- | --- | --- |
| UDA, NeurIPS 2020; report [11] | https://proceedings.neurips.cc/paper/2020/hash/44feb0096faa8326192570788b38c1d1-Abstract.html and linked PDF | Original/augmented prediction consistency with a fixed, no-gradient target. Its soft-target cross-entropy and KL have the same student gradient for a detached target. UDA is semi-supervised; our labeled-bag experiment is not a UDA reproduction. |
| MHIM-MIL, ICCV 2023:4078–4087; report [12] | https://openaccess.thecvf.com/content/ICCV2023/html/Tang_Multiple_Instance_Learning_Framework_with_Masked_Hard_Instance_Mining_for_ICCV_2023_paper.html and linked PDF, Sections 3.3–3.4 | Masked WSI bags, attention-guided hard-instance mining, EMA teacher, bag-representation consistency and masked-student classification. Our version uses uniform random deletion, same-current-model detached class predictions and full-bag CE, without an EMA teacher or hard mining. |
| Patch-GCN, MICCAI 2021:339–349; report [13] | https://link.springer.com/chapter/10.1007/978-3-030-87237-3_33, Sections 3.1–3.2 | Coordinate-based eight-neighbor spatial graph, residual multilayer message passing and attention pooling for survival prediction. Our adapter uses one neighbor-mean correction before ABMIL classification, not the complete graph/survival architecture. The matching neighbor count does not establish historical design provenance. |
| GraphSAGE, NeurIPS 2017; report [14] | https://arxiv.org/html/1706.02216v4, Algorithm 1 and aggregation functions | Neighbor aggregation, including an elementwise mean, and combination with a node's own representation. Supports the mean/concatenation principle, not an exact reproduction of GraphSAGE layers, normalization or training. |
| Highway Networks, 2015; report [15] | https://arxiv.org/html/1505.00387v2, Section 2 | Learned transform/carry gating, including negative gate bias initialization. Our scalar gate scales only an added correction; the identity bypass remains unscaled, unlike the coupled Highway carry gate. |

The three architecture figures are original implementation-specific drawings generated by `scripts/make_architecture_figures.py`. No published figure or implementation code from these five papers was copied. The contribution is the controlled adaptation and evaluation, not a claim to invent consistency training, neighborhood aggregation or gating.
