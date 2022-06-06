## Install dependency packages
Install torch and torchvision, you can ignore it if you installed these packages already.
```
pip install -r requirements.txt
```
## Install espeak for linux
Please refer https://github.com/espeak-ng/espeak-ng/blob/master/docs/building.md

## Dataset
LJ Speech
Dummy dataset available in this directory, consisting of example sentences.


## Truncing dataset (optional)
Original LJ Speech dataset consists of 13100 sentences. Processing them one by one,
as the `benchmark.py` script does, can take many hours.
In order to avoid that, a truncated dataset was prepared using `dataset_trunc.py`.
The script randomly samples a subset from original dataset, verifies if the sentence lengths
of the subset's statistics (mean, media, stdev) are similair to the original - by default
difference is expected to be no greater than 1%.


## Parameters for dataset_trunc.py
### -n                    Number of elements in truncated data
### --dataset-name        Original dataset name, use ljspeech for LJ Speech (default: dummy)
### --metadata-path       Path to dataset metadata file - file containing sentences (default: ./dummy_data.csv)
### --mean-limit          Acceptable difference percentage for mean sentence length between original and subset
### --median-limit        Acceptable difference percentage for median sentence length between original and subset
### --stdev-limit         Acceptable difference percentage for stdev of sentence length between original and subset
### --tolerate            Saves truncated dataset regardles of differences
### --retries             Number of max retries for generating acceptable subset (default: 10)
### --output, -o          Output file (default: ./truncated_metadata.csv)


## Patch
```
cd TTS
git apply ../workload/TTS/TTS.diff
```

## Run inference
```
../workload/InceptionV3/TTS_inf.sh
```

## Parameters for benchmark.py
### --model-path          Tacotron2 model filename
### --config-path         Tacotron2 config filename
### --dataset-name        Dataset name, use ljspeech for LJ Speech (default: dummy)
### --metadata-path       Dataset metadata filename - file containing sentences (default: ./dummy_data.csv)
