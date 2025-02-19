# UDL Project Group 8

## Overview
This repository contains the code and scripts for our UDL Project, which involves training and evaluating an unsupervised deep learning model. Specifically, we adapt latent diffusion architecture in [Lovelace, et al. (2023)](https://arxiv.org/abs/2212.09462) with Glow ([Kingma & Dhariwal 2018](https://arxiv.org/abs/1807.03039)) instead of a diffusion component.

The final model is hosted on Hugging Face for easy access.

## Installation
To set up the environment, install the required dependencies using:

```bash
pip install -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl
```

## Downloading the Final Model
The trained model is available on Hugging Face. You can download it using the `huggingface_hub` library:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="rizkiduwinanto/lnf4lg")
```

Alternatively, you can manually download the model from: [Hugging Face Model Page](https://huggingface.co/rizkiduwinanto/lnf4lg)

## Running the Project
To execute the training scripts, use the provided job scripts:

```bash 
bash jobscript.sh  # Run training
bash jobscript_flow.sh  # Run flow-based job
```

```python
python3 src/main_lm_finetune.py
python3 src/main_flow_training.py
```

To execute evaluation and inference scripts, use the provided job scripts:
```python
python3 src/lm_inference.py
python3 src/flow_inference.py
```
## Logs
Log files are saved as `logs/log.txt` and `logs/log_flow.txt` for monitoring the execution. Also available in `logs`  folder are the samples of flow inferences in and scores of evaluation.

## Credits

This repository builds upon and adapts code from the following publicly available sources:


- [Latent Diffusion for Language (GitHub)](https://github.com/justinlovelace/latent-diffusion-for-language)
- [Glow-PyTorch (GitHub)](https://github.com/rosinality/glow-pytorch)
