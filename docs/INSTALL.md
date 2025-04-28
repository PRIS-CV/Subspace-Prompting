# Installation

### Acknowledgement: This readme file for installing datasets is modified from [MaPLe's](https://github.com/muzairkhattak/multimodal-prompt-learning) official repository.

This codebase is tested on Ubuntu 20.04.2 LTS with python 3.8. Follow the below steps to create environment and install dependencies.

* Setup conda environment (recommended).
```bash
# Create a conda environment
conda create -y -n SuPr python=3.8

# Activate the environment
conda activate SuPr

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

* Clone SuPr code repository and install requirements
```bash
# Clone PromptSRC code base
git clone https://github.com/PRIS-CV/Subspace-Prompting

cd Subspace_Prompting/
# Install requirements


```

* Install dassl library.
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# Clone this repo
# original source: https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```
