# Introduction

Use deep learning techniques and scalograms to detect gastrointestinal slow waves from extracellular bioelectrical recordings.

The code in this repository is written using the pytorch lightning framework, and contains lightning modules based on resnet50 model, lightning data modules, and dataset modules. 
The datasets for training are currently not publicly available.

# Using pre-trained models

Model checkpoints can be downloaded from: https://drive.google.com/drive/folders/1EEwP3_R8k0-gA-yjmIl2hCi3nj3_Fv9s?usp=sharing

Refer the jupyter notebooks on how to use the pre-trained models.

# Generating data

Data can be generated using `./data_modules/generate_dataset.py`, which requires as an input, a [GEMS](https://sites.google.com/site/gimappingsuite/home) .mat data file with marked slow waves.
