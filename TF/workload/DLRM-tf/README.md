Step-by-Step
============

This document is used to list steps of reproducing Tensorflow DLRM model.


# Prerequisite

### 1. Installation

  ```Shell
  pip install intel-tensorflow==2.2     # model load too long time on tensorflow2.3

  ```

### 2. Prepare Dataset
  All datasets can be downloaded from Google drive [here](https://drive.google.com/drive/folders/1taJ91txiMAWBMUtezc_N5gaYuTEpvW_e?usp=sharing). 

  Or find from: `/home2/tensorflow-broad-product/dpg/DLRM/dataset/kaggle_processed.npz`

### 3. Prepare Pre-trained model
  You can use find pre-trained model from: `/home2/tensorflow-broad-product/dpg/DLRM/dlrm`, it is `tf.keras` saved_model.


### benchmark
  ```python
  python dlrm_criteo.py --model_path /home2/tensorflow-broad-product/dpg/DLRM/dlrm



  python dlrm_criteo.py -h

  usage: prepare_model.py [-h]
                          [--model_name {resnet18_v1,resnet50_v1,squeezenet1.0,mobilenet1.0,mobilenetv2_1.0,inceptionv3}]
                          [--model_path MODEL_PATH] [--image_shape IMAGE_SHAPE]

  Prepare pre-trained model for MXNet ImageNet Classifier

  optional arguments:
    -h, --help            show this help message and exit
    --model_name {resnet18_v1,resnet50_v1,squeezenet1.0,mobilenet1.0,mobilenetv2_1.0,inceptionv3}
                          model to download, default is resnet18_v1
    --model_path MODEL_PATH
                          directory to put models, default is ./model
    --image_shape IMAGE_SHAPE
                          model input shape, default is 3,224,224

  ```
  
