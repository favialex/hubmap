# HuBMAP + HPA - Hacking the Human Body

This repository contains my code for the 2022 __HuBMAP + HPA - Hacking the Human Body__ Competition hosted on [kaggle](https://www.kaggle.com/competitions/hubmap-organ-segmentation/).


Libraries used in this repository:
- [MONAI](https://monai.io/) is used for image loading, data transformation and efficient dataset and datalaoder generation.
- [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch) for their intuitive approach of generating state-of-the-art segmentation models.
- [Catalyst](https://catalyst-team.com/) for simple and flexible model training.
- A few components from [fastai](https://www.fast.ai/), some transformations from [albumentations](https://albumentations.ai/) and [PyTorch](https://pytorch.org/).


Repository structure:

```
root
  - data
    - ...
  - logs
    - exp_01
    - exp_02
    - ...
  - nbs
    - exp_01
    - exp_02
    - ...
```

Only the code is provided here.