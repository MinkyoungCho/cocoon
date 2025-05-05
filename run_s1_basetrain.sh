#!/bin/bash

# Step1 is to train the original FUTR3D model

ln -sf plugin/futr3d/models/detectors/s1_basetrain.py plugin/futr3d/models/detectors/futr3d.py
ln -sf plugin/futr3d/models/head/s1_basetrain.py plugin/futr3d/models/head/futr3d_head.py
ln -sf plugin/futr3d/models/utils/transformer_s1_basetrain.py plugin/futr3d/models/utils/futr3d_transformer.py
ln -sf  plugin/futr3d/models/utils/attention_s1_basetrain.py plugin/futr3d/models/utils/futr3d_attention.py


# bash tools/dist_train.sh plugin/futr3d/configs/lidar_only/config_s1_basetrain.py 8  # when using 8 GPUs
python tools/train.py plugin/futr3d/configs/lidar_cam/config_s1_basetrain.py  # when using 1 GPU