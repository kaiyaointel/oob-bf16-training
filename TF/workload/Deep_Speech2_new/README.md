Step-by-Step
============

This document is used to list steps of reproducing Tensorflow DeepSpeech2 model.


# Prerequisite

### 1. Prepare envs

  
  ```Shell

  pip install intel-tensorflow==2.2
  git clone https://github.com/tensorflow/models.git
  cp ds2.patch models/research/deep_speech
  cd models/research/deep_speech
  pip install -r requirements.txt
  git apply ds2.patch

  ```

### 2. Prepare Dataset

  Find from: `/home2/tensorflow-broad-product/oob_tf_models/dpg/Deep_Speech2/librispeech_data/

### 3. Prepare Pre-trained model
  You can use find pre-trained model from: `/home2/tensorflow-broad-product/oob_tf_models/dpg/Deep_Speech2/model/


### benchmark
  ```python
python deep_speech.py --train_data_dir=/path_to_librispeech_data/final_eval_dataset.csv --eval_data_dir=/path_to_librispeech_data/final_eval_dataset.csv  --wer_threshold=0.23 --seed=1 --only_dev --model_dir /path_to_model/ --batch_size=1
  ```
  
