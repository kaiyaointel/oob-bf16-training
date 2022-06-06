# install dependency package

1. torchaudio install, download the corresponding to install
```
git clone https://github.com/pytorch/audio.git && \
    cd audio && \
    # checkout the corresponding version
    git submodule update --init --recursive && \
    pip install --no-deps -e . && \
cd ..
```

2. warp-ctc install:
```
git clone https://github.com/SeanNaren/warp-ctc.git && \
    cd warp-ctc && \
    mkdir -p build && cd build && cmake .. && make && \
    cd ../pytorch_binding && python setup.py install
cd ../../
```
3. pip install package
```
pip install python-Levenshtein librosa
```

4. apply patch
```
cd PT/mlperf_training
git apply ../workload/deepspeech/deepspeech.patch
```

# dataset and pretrained model
```
cd speech_recognition
rsync -avz ../../workload/deepspeech/*.csv .
ln -sf /home2/pytorch-broad-models/DeepSpeech2/LibriSpeech_dataset .
ln -sf /home2/pytorch-broad-models/DeepSpeech2/models .
# or download all via 'bash ./download_dataset.sh'
```

# run real time inference
```bash
cd pytorch
python train.py \
    --model_path ../models/deepspeech_10.pth \
    --seed 1 \
    --batch_size ${bs} \
    --evaluate \
    --channels_last 1 \
    --precision bfloat16

usage: train.py [-h] [--checkpoint] [--save_folder SAVE_FOLDER] 
                [--model_path MODEL_PATH] [--continue_from CONTINUE_FROM] 
                [--seed SEED] [--acc ACC]
                [--start_epoch START_EPOCH] [--channels_last CHANNELS_LAST]

DeepSpeech training

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint          Enables checkpoint saving of model
  --save_folder SAVE_FOLDER
                        Location to save epoch models
  --model_path MODEL_PATH
                        Location to save best validation model
  --continue_from CONTINUE_FROM
                        Continue from checkpoint model
  --seed SEED           Random Seed
  --acc ACC             Target WER
  --channels_last       Use the channels last format
  --start_epoch START_EPOCH
                        Number of epochs at which to start from
```
