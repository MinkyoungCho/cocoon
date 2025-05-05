#!/bin/bash

# Step2 is to train Feature Impression and Feature Aligner which are required for uncertainty quantification

ln -sf plugin/futr3d/models/detectors/s2_train_fa_fi.py plugin/futr3d/models/detectors/futr3d.py
ln -sf plugin/futr3d/models/head/s2_train_fa_fi.py plugin/futr3d/models/head/futr3d_head.py
ln -sf plugin/futr3d/models/utils/transformer_s2_train_fa_fi.py plugin/futr3d/models/utils/futr3d_transformer.py
ln -sf  plugin/futr3d/models/utils/attention_s2_train_fa_fi.py plugin/futr3d/models/utils/futr3d_attention.py

python tools/train.py plugin/futr3d/configs/lidar_cam/config_s2_train_fa_fi.py