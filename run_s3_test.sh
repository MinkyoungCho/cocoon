#!/bin/bash

# Step4 is to test the cocoon-equipped model on the test set

ln -sf plugin/futr3d/models/detectors/s3_test.py plugin/futr3d/models/detectors/futr3d.py
ln -sf plugin/futr3d/models/head/s3_test.py plugin/futr3d/models/head/futr3d_head.py
ln -sf plugin/futr3d/models/utils/transformer_s3_test.py plugin/futr3d/models/utils/futr3d_transformer.py
ln -sf  plugin/futr3d/models/utils/attention_s3_test.py plugin/futr3d/models/utils/futr3d_attention.py


python tools/test.py plugin/futr3d/configs/lidar_cam/config_s3_test.py