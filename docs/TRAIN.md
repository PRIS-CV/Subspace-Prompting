
# 🚀 Running SuPr

This section provides detailed instructions on running **SuPr** experiments across different scenarios: base-to-novel transfer, cross-dataset/domain generalization, and few-shot learning.

---

# 📋 Table of Contents
- [🚀 Running SuPr](#-running-supr)
- [📋 Table of Contents](#-table-of-contents)
  - [🖥️ GPU and Memory Requirements](#️-gpu-and-memory-requirements)
  - [(1) 🏆 Base-to-Novel Experiments](#1--base-to-novel-experiments)
    - [Step-by-Step Instructions](#step-by-step-instructions)
    - [🔥 SuPr + PromptSRC](#-supr--promptsrc)
  - [(2) 🌐 Cross-Dataset / Domain Generalization Experiments](#2--cross-dataset--domain-generalization-experiments)
    - [Step-by-Step Instructions](#step-by-step-instructions-1)
  - [(3) 🎯 Few-Shot Learning Experiments](#3--few-shot-learning-experiments)
    - [Step-by-Step Instructions](#step-by-step-instructions-2)

---

## 🖥️ GPU and Memory Requirements

- All experiments are trained with a **batch size of 4** on a **single NVIDIA 4090** GPU, with the exception of ImageNet. 
- **ImageNet** experiments require approximately **30 GB** of GPU memory. For ImageNet, we recommend using a **single NVIDIA A800**.
- We provide two implementations for projection:
  - **SVD**-based projection
  - **Least squares**-based projection  
  > **Tip:** Although mathematically equivalent, the least squares method is more GPU memory-efficient.

---

## (1) 🏆 Base-to-Novel Experiments

### Step-by-Step Instructions

1. **Configuration**  
   Use the configuration file located at:  
   ```
   configs/trainers/SuPr/vit_b16_ep10_batch4_4+4ctx.yaml
   ```

2. **Update Dataset Path**  
   Change the dataset path in:
   - `scripts/supr/base2new.sh` (for SuPr)
   - `scripts/supr_ens/base2new.sh` (for SuPrEns)  
   
   (Modify **line 4** to point to your local dataset directory.)

3. **Training Commands**  
   Run the following command to train SuPr (repeat for seeds 1, 2, and 3):

   ```bash
   # Set dataset (e.g., imagenet)
   # Available datasets: [caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]

   # Train SuPr
   sh scripts/supr/base2new.sh imagenet

   # Train SuPr Ens
   sh scripts/supr_ens/base2new.sh imagenet
   ```

4. **Output Directory**  
   Results will be saved automatically at:
   ```
   Base results: output/base2new/${TRAINER}/${CFG}/train_base/${DATASET}/shots_${SHOTS}/seed${SEED}
   Novel results: output/base2new/${TRAINER}/${CFG}/test_new/${DATASET}/shots_${SHOTS}/seed${SEED}
   ```

5. **Result Aggregation**  
   After finishing training for all seeds, run:

   ```bash
   # Aggregate base-to-novel results
   python parse_test_res.py -type base2new output/base2new/SuPr/vit_b16_ep10_batch4_4+4ctx/test_new/caltech101/shots_16
   ```

---

### 🔥 SuPr + PromptSRC

To run SuPr combined with PromptSRC:

1. **Configuration**  
   Use the configuration file:  
   ```
   configs/trainers/SuPr/vit_b16_ep20_batch4_4+4ctx_promptsrc.yaml
   ```

2. **Training Command**  
   ```bash
   # Train SuPr+PromptSRC
   sh scripts/supr_src/base2new.sh imagenet
   ```

---

## (2) 🌐 Cross-Dataset / Domain Generalization Experiments

### Step-by-Step Instructions

1. **Configuration**  
   Use the configuration file at:  
   ```
   configs/trainers/SuPr/vit_b16_ep12_batch8_4+4ctx_cross_datasets.yaml
   ```

2. **Update Dataset Path**  
   Change the dataset path in:  
   ```
   scripts/supr/cross_dg.sh (line 4)
   ```

3. **Training Command**  
   Run the following script:

   ```bash
   # This script will:
   # 1. Train SuPr on ImageNet (3 seeds)
   # 2. Evaluate on 10 cross-datasets
   # 3. Perform DG evaluation on ImageNetV2, ImageNet-Sketch, ImageNet-A, and ImageNet-R

   sh scripts/supr/cross_dg.sh
   ```

4. **Output Directory**  
   Results will be saved at:
   ```
   output/cross_dg/${TRAINER}/${CFG}/${DATASET}/shots_${SHOTS}/seed${SEED}
   ```

5. **Result Aggregation**  

   ```bash
   # Aggregate cross-dataset results
   python parse_test_res.py -type cross output/cross_dg/SuPr/vit_b16_ep12_batch8_4+4ctx_cross_datasets/caltech101/shots_16

   # Aggregate domain generalization results
   python parse_test_res.py -type dg output/cross_dg/SuPr/vit_b16_ep12_batch8_4+4ctx_cross_datasets/imagenet/shots_16
   ```

---

## (3) 🎯 Few-Shot Learning Experiments

### Step-by-Step Instructions

1. **Configuration**  
   Use the configuration file at:  
   ```
   configs/trainers/SuPr/vit_b16_ep25_batch8_4+4ctx_few_shot.yaml
   ```

2. **Update Dataset Path**  
   Change the dataset path in:  
   ```
   scripts/supr/few_shot.sh (line 4)
   ```

3. **Training Command**  
   ```bash
   # dataset=imagenet
   # Other available datasets: [caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]

   sh scripts/supr/fewshot.sh imagenet
   ```

4. **Output Directory**  
   Results will be saved at:
   ```
   output/fewshot/${TRAINER}/${CFG}/${DATASET}/shots_${SHOTS}/seed${SEED}
   ```

5. **Result Aggregation**  

   ```bash
   # Aggregate few-shot results
   python parse_test_res.py -type fewshot output/fewshot/SuPr/vit_b16_ep25_batch8_4+4ctx_few_shot/imagenet/shots_4
   ```

---

> **Tip:** Always run experiments across **three random seeds** to ensure reproducibility and statistically stable results.
>  
> **Warning:** Be sure to update dataset paths correctly before launching the scripts. Missing this may lead to training failures or empty outputs.

---
