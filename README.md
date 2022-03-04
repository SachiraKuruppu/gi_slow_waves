# Introduction

Use deep learning techniques and scalograms to detect gastrointestinal slow waves from extracellular bioelectrical recordings.

The code in this repository is written using the pytorch lightning framework, and contains lightning modules based on resnet50 model, lightning data modules, and dataset modules. 
The datasets for training are currently not publicly available.

# Using model checkpoints

Model checkpoints can be downloaded from: https://drive.google.com/drive/folders/1EEwP3_R8k0-gA-yjmIl2hCi3nj3_Fv9s?usp=sharing

Refer the jupyter notebooks on how to use the model checkpoints.

# Generating data

Data can be generated using `./data_modules/generate_dataset.py`, which requires as an input, a [GEMS](https://sites.google.com/site/gimappingsuite/home) .mat data file with marked slow waves.

# Notebooks

## Centered slow wave detection
Use transfer learning to train and validate a resnet 50 classifier to detect `slow wave` and `no slow wave` classes from scalogram images where slow waves are centered (if present). Uses the model in `resnet_sw_detect.py`

## Slow wave AT detection*
Use transfer learning to train and validate a resnet 50 model. The model contains a classification and a regression output. The classification predicts if a given scalogram contains a slow waves (slow wave does not have to be centered on the scalogram). The regression outputs the activation time of the slow waves. In cases where there are multiple slow waves in a scalogram, regression outputs the activation time of the left most slow wave. Uses the models in `resnet_sw_activation_time*.py`.
