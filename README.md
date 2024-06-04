# TMT

A novel Token-disentangling Mutual Transformer(TMT) for multimodal emotion recognition. The TMT can effectively disentangle inter-modality emotion consistency features and intra-modality
emotion heterogeneity features and mutually fuse them for more comprehensive multimodal emotion representations by introducing two primary modules, namely multimodal emotion Token disentanglement and Token mutual Transformer. The Models folder contains our overall TMT model code, and the subNets folder contains the Transforme structure used in it.

### Features

- Train, test and compare in a unified framework.
- Supports Our TMT model.
- Supports 3 datasets: [MOSI](https://ieeexplore.ieee.org/abstract/document/7742221), [MOSEI](https://aclanthology.org/P18-1208.pdf), and [CH-SIMS](https://aclanthology.org/2020.acl-main.343/).
- Easy to use, provides Python APIs and commandline tools.
- Experiment with fully customized multimodal features extracted by [MMSA-FET](https://github.com/thuiar/MMSA-FET) toolkit.

## 1. Get Started

> **Note:** From version 2.0, we packaged the project and uploaded it to PyPI in the hope of making it easier to use. If you don't like the new structure, you can always switch back to `v_1.0` branch. 

### 1.1 Use Python API
At present, only the model and test code are uploaded, and the training code will be released after the paper is published.
- Import and use in any python file:

  ```python
  python test.py
  python test_acc5.py

- For more detailed usage, please contact ygh2@cug.edu.cn.



### 1.3 Clone & Edit the Code

- Clone this repo and install requirements.
  ```bash
  $ git clone https://github.com/cug-ygh/TMT
  ```


## 2. Datasets

TMT currently supports MOSI, MOSEI, and CH-SIMS dataset. Use the following links to download raw videos, feature files and label files. You don't need to download raw videos if you're not planning to run end-to-end tasks. 

- [BaiduYun Disk](https://pan.baidu.com/s/1XmobKHUqnXciAm7hfnj2gg) `code: mfet`
- [Google Drive](https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk?usp=sharing)

SHA-256 for feature files:

```text
`MOSI/Processed/unaligned_50.pkl`:  `78e0f8b5ef8ff71558e7307848fc1fa929ecb078203f565ab22b9daab2e02524`
`MOSI/Processed/aligned_50.pkl`:    `d3994fd25681f9c7ad6e9c6596a6fe9b4beb85ff7d478ba978b124139002e5f9`
`MOSEI/Processed/unaligned_50.pkl`: `ad8b23d50557045e7d47959ce6c5b955d8d983f2979c7d9b7b9226f6dd6fec1f`
`MOSEI/Processed/aligned_50.pkl`:   `45eccfb748a87c80ecab9bfac29582e7b1466bf6605ff29d3b338a75120bf791`
`SIMS/Processed/unaligned_39.pkl`:  `c9e20c13ec0454d98bb9c1e520e490c75146bfa2dfeeea78d84de047dbdd442f`
```

Our uses feature files that are organized as follows:

```python
{
    "train": {
        "raw_text": [],              # raw text
        "audio": [],                 # audio feature
        "vision": [],                # video feature
        "id": [],                    # [video_id$_$clip_id, ..., ...]
        "text": [],                  # bert feature
        "text_bert": [],             # word ids for bert
        "audio_lengths": [],         # audio feature lenth(over time) for every sample
        "vision_lengths": [],        # same as audio_lengths
        "annotations": [],           # strings
        "classification_labels": [], # Negative(0), Neutral(1), Positive(2). Deprecated in v_2.0
        "regression_labels": []      # Negative(<0), Neutral(0), Positive(>0)
    },
    "valid": {***},                  # same as "train"
    "test": {***},                   # same as "train"
}
```
## 3. Citation


Please cite our paper if you find our work useful for your research:

@article{YIN2024108348,
title = {Token-disentangling Mutual Transformer for multimodal emotion recognition},
journal = {Engineering Applications of Artificial Intelligence},
volume = {133},
pages = {108348},
year = {2024},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2024.108348},
url = {https://www.sciencedirect.com/science/article/pii/S0952197624005062},
author = {Guanghao Yin and Yuanyuan Liu and Tengfei Liu and Haoyu Zhang and Fang Fang and Chang Tang and Liangxiao Jiang},
keywords = {Multimodal emotion recognition, Multimodal emotion Token disentanglement, Token mutual Transformer, Token separation learning, Bi-directional query learning},
abstract = {Multimodal emotion recognition presents a complex challenge, as it involves the identification of human emotions using various modalities such as video, text, and audio. Existing methods focus mainly on the fusion information from multimodal data, but ignore the interaction of the modality-specific heterogeneity features that contribute differently to emotions, leading to sub-optimal results. To tackle this challenge, we propose a novel Token-disentangling Mutual Transformer (TMT) for robust multimodal emotion recognition, by effectively disentangling and interacting inter-modality emotion consistency features and intra-modality emotion heterogeneity features. Specifically, the TMT consists of two main modules: multimodal emotion Token disentanglement and Token mutual Transformer. In the multimodal emotion Token disentanglement, we introduce a Token separation encoder with an elaborated Token disentanglement regularization, which effectively disentangle the inter-modality emotion consistency feature Token from each intra-modality emotion heterogeneity feature Token; consequently, the emotion-related consistency and heterogeneity information can be performed independently and comprehensively. Furthermore, we devise the Token mutual Transformer with two cross-modal encoders to interact and fuse the disentangled feature Tokens by using bi-directional query learning, which delivers more comprehensive and complementary multimodal emotion representations for multimodal emotion recognition. We evaluate our model on three popular three-modality emotion datasets, namely CMU-MOSI, CMU-MOSEI, and CH-SIMS, and the experimental results affirm the superior performance of our model compared to state-of-the-art methods, achieving state-of-the-art recognition performance. Evaluation Codes and models are released at https://github.com/cug-ygh/TMT.}
}






