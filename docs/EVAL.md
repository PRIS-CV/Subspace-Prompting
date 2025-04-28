
# 📑 Evaluating and Reproducing SuPr Results

We provide ready-to-use bash scripts under the [scripts/](../scripts) directory for evaluating **SuPr**, **SuPr+PromptSRC**, and **SuPrEns** models using our pre-trained checkpoints.

Please ensure that you update the `DATA` variable in each script to match your dataset path, and run all commands from the project root directory `Subspace_Prompting/`.

We have already provided:
- Precomputed evaluation results under [output/](../output)
- Aggregated and summarized statistics under [parse_results/](../parse_results)

Below, we guide you through reproducing these results by yourself.

---

## 🔥 SuPr Reproduction Guide

We now explain how to reproduce our reported results step-by-step.

---

### 🛠️ Preliminary Setup

To reproduce the results (taking ImageNet as an example), follow these steps:

1. **Create the environment and install dependencies**  
   - Follow the instructions in [INSTALL.md](../docs/INSTALL.md) to set up the environment and install the `Dassl.pytorch` library.

2. **Prepare datasets**  
   - Follow the dataset preparation guidelines provided in [DATASETS.md](../docs/DATASETS.md).

3. **Download pre-trained weights**  
   - Download the zipped folder containing all pre-trained weights from this [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/syed_wasim_mbzuai_ac_ae/Em_3tkSj6T9AmhVjmzKTL3gBYNehhvfJl8ke2pU3U0nabA?e=9ecjQA).
   - After extraction, the folder structure should look like:

```text
weights/
|–– SuPr/
|   |–– base2new/
|   |   |–– imagenet/
|   |       |–– shots_16/
|   |           |–– seed1/
|   |           |–– seed2/
|   |           |–– seed3/
|   ...
|   |–– cross_dg/
|   |–– fewshot/
|
|–– SubspacePromptSRC/
|   |–– base2new/
|   ...
|
|–– SuPrEns/
|   |–– base2new/
|   ...
```

> **Important:**  
> If you place the `weights/` folder outside the `Subspace_Prompting/` root directory,  
> remember to update the `${WEIGHTSPATH}` variable inside the following scripts:
> - `scripts/supr/reproduce_base2novel_setting.sh`
> - `scripts/supr/reproduce_fewshot.sh`
> - `scripts/supr/reproduce_xd.sh`
> - `scripts/supr_src/reproduce_base2novel_setting.sh`
> - `scripts/supr_ens/reproduce_base2novel_setting.sh`

---

### ⚡ Reproducing Experiments

After setting up, run the following command from the `Subspace_Prompting/` root directory:

```bash
bash reproduce.sh
```

This command will automatically start evaluation across all settings, using the provided pre-trained models.

The evaluation logs and results will be saved under the `output/` directory.

---

### 📈 Aggregating Results

After running evaluations, you can aggregate the results across seeds and tasks by running:

```bash
# Base-to-Novel Evaluation Results

# SuPr
python parse_test_res.py -type base2new output/base2new/SuPr/reproduce_vit_b16_ep10_batch4_4+4ctx/test_new/caltech101/shots_16

# SuPr+PromptSRC
python parse_test_res.py -type base2new output/base2new/SubspacePromptSRC/reproduce_vit_b16_ep20_batch4_4+4ctx_promptsrc/test_new/imagenet/shots_16

# SuPr Ensemble
python parse_test_res.py -type base2new output/base2new/SuPrEns/reproduce_vit_b16_ep10_batch4_4+4ctx/test_new/imagenet/shots_16


# Cross-Dataset Generalization Results
python parse_test_res.py -type cross output/cross_dg/SuPr/reproduce_vit_b16_ep12_batch8_4+4ctx_cross_datasets/caltech101/shots_16

# Domain Generalization Results
python parse_test_res.py -type dg output/cross_dg/SuPr/reproduce_vit_b16_ep12_batch8_4+4ctx_cross_datasets/imagenet/shots_16
```

The aggregated results will be automatically compiled into Excel spreadsheets for easy reporting.

---

> **Tip:** If you want to evaluate on other datasets beyond ImageNet, simply adjust the dataset names and paths accordingly in the scripts.

> **Warning:** Ensure that datasets are correctly prepared and accessible by the scripts, otherwise evaluation may fail.

---
