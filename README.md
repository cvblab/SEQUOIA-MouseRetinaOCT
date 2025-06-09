# SEQUOIA-RodentOCT

## Overview

**Title:** *A Curated OCT Dataset of Ex-Vivo Mice Retinas for Deep Learning Models*

**Description:**  
Optical coherence tomography (OCT) is a fundamental technology in ophthalmic research, providing high-resolution 3D cross-sections of the retina. Algorithms for retinal cell layer segmentation provide critical information on retinal structure and pathology.

Although novel segmentation algorithms based on deep learning approaches have proven effective in analysing human retinal OCT B-scans, their application to rodent retinas remains limited — primarily due to the lack of publicly available relevant OCT datasets. This limitation constrains the development of generalisable AI models adapted to preclinical animal studies, which are crucial for improving diagnosis and treatment of retinal diseases.

To address this gap, we present a **novel dataset** composed of OCT volumes of *ex-vivo* rodent retinas embedded in aqueous media — a sample type rarely represented in existing datasets, which predominantly focus on human retinas. Our dataset is meticulously annotated to support the segmentation of the **retinal nerve fibre layer (RNFL)**, a critical task for understanding retinal health and disease progression.

Given the limited availability of expert annotations, we implemented a **teacher-student training strategy** to generate pseudo-masks, enabling large-scale validation of the dataset with minimal expert input. This dataset offers a scalable and foundational resource for developing artificial intelligence-driven models in rodent retinal imaging and supports further advancements in preclinical ophthalmic diagnostics.

---

## Usage

### 1. Clone the repository

```bash
git clone https://github.com/cvblab/SEQUOIA-RodentOCT

```

### 2. Install dependencies

Make sure you have Python 3.8+ installed. Then run:

```bash
pip install -r requirements.txt
```



### 3. File Structure

The repository is divided into three main folders:

```
SEQUOIA-RodentOCT/
│
├── 1_preprocessing/              # MATLAB scripts to load OCT volumes and convert them to .mat format
│   └── load_oct_to_mat.m         # Main function for preprocessing volumes
│
├── 2_segmentation_model/         # PyTorch implementation of the teacher-student segmentation strategy
│   ├── main.py                   # Main script for training
│   ├── model.py                  # U-Net architecture.
│   ├── train.py                  # Training script.
│   ├── dataset.py                # Contains the main functions to load the data.
│   ├── inference.py              # Inference and pseudo-mask generation.
│   └── compute_metrics.py        # Script for the evaluation.
│
└── 3_postprocessing/             # MATLAB scripts for postprocessing pseudo-masks
    └── pseudo_mask_refinement.m  # Code for refining the pseudo-masks obtained during inference.
```
> **Note:** Some preprocessing steps are written in MATLAB. You will need a compatible version of MATLAB R2019b or superior to execute those scripts.
### 4. Run the code

To train the model, run the following command in the terminal:

```bash
python main.py
```

Make sure you configure `main.py` or individual scripts with the correct paths to your data and model files.

To evaluate its performance, run the following command in the terminal:

```bash
python compute_metrics.py
```
---

## Acknowledgment

TBC

---

## Citation

TBC

If you use this dataset, code, or methodology in your research, please cite:

```
@article{SEQUOIA-RodentOCT,
  title={A Curated OCT Dataset of Ex-Vivo Mice Retinas for Deep Learning Models},
  author={Ferguson, A. and García, T. and et al.},
  journal={To be updated upon publication},
  year={2024}
}
```

*A DOI or final citation will be added once the dataset is officially published.*

---

## License

TBC

This repository is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for more information.

---


