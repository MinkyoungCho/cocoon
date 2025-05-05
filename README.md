# Cocoon: Robust Multi-Modal Perception with Uncertainty-Aware Sensor Fusion (ICLR'25) 

Cocoon is a new object- and feature-level uncertainty-aware multimodal fusion framework designed for 3D OD tasks. See how we achieve uncertainty quantification and comparison for heterogeneous representations!

- [paper](https://openreview.net/pdf?id=DKgAFfCs5F) / [website](https://minkyoungcho.github.io/cocoon/) / [poster](https://iclr.cc/media/PosterPDFs/ICLR%202025/30466.png)

---

## Overview

This repo provides a modular pipeline for Cocoon, built on top of a base model ([FUTR3D](https://github.com/NVlabs/FUTR3D)).

<img width="800" alt="image" src="https://github.com/user-attachments/assets/3704fd18-a187-45bd-9b55-bb26ce7b6500" />


---

## Pipeline Stages

### 0. Conda Environment Setup

- Setup conda virtual environment called `cocoon`.
- Pytorch version: https://pytorch.org/get-started/previous-versions/#linux-and-windows-40
- This follows [FUTR3D's installation guide](https://github.com/Tsinghua-MARS-Lab/futr3d#installation).
```
conda create -n cocoon python=3.7
conda activate cocoon
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

git clone https://github.com/MinkyoungCho/cocoon.git
cd cocoon
pip3 install -v -e .
```

### 1. Dataset Preparation

- Organize the NuScenes dataset
   - Follow the link to process the nuScenes data: [nuscenes dataset prep guide](https://github.com/open-mmlab/mmdetection3d/blob/main/docs/en/advanced_guides/datasets/nuscenes.md#dataset-preparation)
     
- Split original training set 
   - Use **600 scenes** for the **proper training set**.  
   - Use **100 scenes** as the **calibration set** for uncertainty quantification.  
   - Refer to **Appendix A** in the paper for more details.
     
- Usage
   - The **proper training set** is used for pretraining the base model.
   - The **calibration set** is used for uncertainty quantification.
 
- Commands:
   - Ensure that no directory named nuscenes is present in `./data`
     ```
      cd ./data/ && mv nuscenes nuscenes_original
     ```
  - Generate Proper Training Set (s1, s2, s3)
     ```
      # replace `<python_path>/site-packages/nuscenes/utils/splits.py` with `splits_train_test.py`.
      cd ./data && mkdir nuscenes_propertrain_val && cd nuscenes_propertrain_val
      ln -sf ../nuscenes_original/maps/ .
      ln -sf ../nuscenes_original/samples/ .
      ln -sf ../nuscenes_original/sweeps/ .
      ln -sf ../nuscenes_original/v1.0-trainval .
      cd ../../
      python tools/create_data.py nuscenes --version v1.0-trainval --root-path ./data/nuscenes_propertrain_val --out-dir ./data/nuscenes_propertrain_val --extra-tag nuscenes
   
      # Before starting pre-traing, fine-tuning, and inference stages, make the symbolic link
      cd ./data/
      ln -sf nuscenes_propertrain_val nuscenes 
      ```

   - Generate Calibration Set (s3 -- with this dataset, we can generate NC score pool) 
   
      ```
      # replace `<python_path>/site-packages/nuscenes/utils/splits.py` with `splits_calib.py`.
      cd ./data && mkdir nuscenes_calib && cd nuscenes_calib
      ln -sf ../nuscenes_original/maps/ .
      ln -sf ../nuscenes_original/samples/ .
      ln -sf ../nuscenes_original/sweeps/ .
      ln -sf ../nuscenes_original/v1.0-trainval .
      cd ../../
      python tools/create_data.py nuscenes --version v1.0-trainval --root-path ./data/nuscenes_calib --out-dir ./data/nuscenes_calib --extra-tag nuscenes
      
      # Before starting calibration stage, make the symbolic link
      cd ./data/
      ln -sf nuscenes_calib nuscenes 
      ```

- The folder structure after processing should be as below 

```
cocoon
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl
```


### 2. Nonconformity (NC) Score Pool Construction

- Use the calibration data samples to create a **Nonconformity (NC) score pool**.
- Run this step using the **pretrained base model**.
- This step is essential for enabling uncertainty quantification.



### 3. Base Model Pretraining

- Pretrain the base model following the original [FUTR3D training command](https://github.com/Tsinghua-MARS-Lab/futr3d?tab=readme-ov-file#train).
- Use proper training set (composed of 600 scenes) for this stage.
- **Note**: This stage does **not** include dynamic weighting.


```
bash run_s1_basetrain.sh
```


### 4. Feature Impression \& Feature Aligner Training 
- Jointly train the feature impression and feature aligner with the proposed loss function (Eq. 3)
- **Note**: This stage does **not** include dynamic weighting.

```
bash run_s2_train_fa_fi.sh
```



### 5. Inference with Uncertainty-Aware Sensor Fusion 

- Perform post-training inference with **dynamic weighting enabled**.
- This step applies uncertainty-aware post-processing based on calibrated outputs.
- NC score pool construction can be performed with this code set.

```
bash run_s3_test.sh
```

---

## Acknowledgment

For the implementation, we rely heavily on MMCV, MMDetection, MMDetection3D, and **FUTR3D**. 

---

## Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@inproceedings{cho2025cocoon,
  title={Cocoon: Robust Multi-Modal Perception with Uncertainty-Aware Sensor Fusion},
  author={Cho, Minkyoung and Cao, Yulong and Sun, Jiachen and Zhang, Qingzhao and Pavone, Marco and Park, Jeong Joon and Yang, Heng and Mao, Zhuoqing},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
