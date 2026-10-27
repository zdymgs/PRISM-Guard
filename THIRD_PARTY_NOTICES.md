# Third-party notices

The MIT license in this repository covers PRISM-authored code. Libraries, external model implementations, pretrained weights and datasets retain their own licenses.

## Software dependencies

| Component | License | Source |
|---|---|---|
| NumPy, SciPy, scikit-learn, pandas | BSD family; bundled components have their own notices | [NumPy](https://github.com/numpy/numpy), [SciPy](https://github.com/scipy/scipy), [scikit-learn](https://github.com/scikit-learn/scikit-learn), [pandas](https://github.com/pandas-dev/pandas) |
| PyTorch | BSD-3-Clause | [LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| torchvision, including MC3-18 | BSD-3-Clause | [v0.24.1 LICENSE](https://github.com/pytorch/vision/blob/v0.24.1/LICENSE) |
| PyYAML, pytest | MIT | [PyYAML](https://github.com/yaml/pyyaml), [pytest](https://github.com/pytest-dev/pytest) |
| threadpoolctl, joblib | BSD-3-Clause | [threadpoolctl](https://github.com/joblib/threadpoolctl), [joblib](https://github.com/joblib/joblib) |
| Matplotlib | Matplotlib license, based on the PSF license | [LICENSE](https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE) |
| Pillow | HPND / PIL software license | [LICENSE](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| OpenCV | Apache-2.0; binary distributions include additional notices | [LICENSE](https://github.com/opencv/opencv/blob/4.x/LICENSE) |
| Transformers, Hugging Face Hub, safetensors | Apache-2.0 | [Transformers](https://github.com/huggingface/transformers), [Hub](https://github.com/huggingface/huggingface_hub), [safetensors](https://github.com/huggingface/safetensors) |
| Ultralytics YOLO11n-pose | AGPL-3.0, or a separate Enterprise license | [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE), [licensing](https://www.ultralytics.com/license) |

Challenge inference calls Ultralytics for pose-based input preparation. The source tree provides calling code and excludes Ultralytics source copies and YOLO weights. The challenge bundle published as a release asset embeds the YOLO11n-pose checkpoint, which remains under AGPL-3.0. Use and distribution of a combined application must follow the applicable Ultralytics terms.

MC3-18 is called through torchvision. Obtain its Kinetics pretrained weights separately and follow the [official model documentation](https://docs.pytorch.org/vision/0.24/models/generated/torchvision.models.video.mc3_18.html).

## Teacher backbones

| Backbone | Implementation and model terms |
|---|---|
| VideoMAE | Original [VideoMAE source](https://github.com/MCG-NJU/VideoMAE/blob/main/LICENSE) and the [base](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics) / [large](https://huggingface.co/MCG-NJU/videomae-large-finetuned-kinetics) model cards use CC-BY-NC-4.0. The Transformers implementation has its separate Apache-2.0 license. |
| VideoMAEv2 | [Source](https://github.com/OpenGVLab/VideoMAEv2/blob/master/LICENSE): MIT. The current [VideoMAE2 model card](https://huggingface.co/OpenGVLab/VideoMAE2) declares Apache-2.0. Follow the terms of the exact checkpoint revision you obtain. |
| InternVideo2 | [Source](https://github.com/OpenGVLab/InternVideo/blob/main/LICENSE): Apache-2.0. The [distillation model repository](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models) does not clearly state a separate weight license; confirm its terms before use. |
| V-JEPA2 | [Source](https://github.com/facebookresearch/vjepa2/blob/main/LICENSE) and the referenced [model card](https://huggingface.co/facebook/vjepa2-vitl-fpc16-256-ssv2): MIT. |
| LaViLa | [Source](https://github.com/facebookresearch/LaViLa/blob/main/LICENSE): MIT. Check the terms of the particular pretrained checkpoint separately. |
| DINOv2 | Referenced [model card](https://huggingface.co/facebook/dinov2-base): Apache-2.0. |

Pretrained weights and external backbone source trees are obtained separately. Included upstream license texts retain their original copyright notices.

## Datasets

- **CUHK-X:** [License v2.0](https://github.com/openaiotlab/CUHK-X/blob/main/LICENSE), noncommercial use with provider authorization; dataset redistribution is prohibited. Cite the [original paper](https://dl.acm.org/doi/10.1145/3745756.3809209) and acknowledge AIoT Lab, Department of Information Engineering, The Chinese University of Hong Kong.
- **MMAct:** [official terms](https://mmact19.github.io/2019/), academic use through the provider; redistribution is prohibited.
- **MM-Fit:** obtain each modality from the [official project](https://mmfit.github.io/) and follow the terms attached to that release.

This repository distributes code and schematic diagrams only. Dataset files, derived features, posterior banks, predictions and weights are excluded.
