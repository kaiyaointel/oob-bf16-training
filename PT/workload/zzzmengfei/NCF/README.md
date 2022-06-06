# install dependency package

1. pip install package

```
pip install mlperf_compliance numpy_indexed
```

3. apply patch

```
cd PT/mlperf_training
git apply PT/workload/NCF/NCF.patch

```

# dataset

### 1. Steps to download

```
# Download ./download_dataset.sh and unzip ml-20m.zip.
cd PT/mlperf_training/recommendation
bash ./download_dataset.sh
```

### 2. Step to expand the dataset (default is x16 users, x32 items, here we use x1 users, x1 items)

go to `PT/mlperf_training/data_generation/fractal_graph_expansions` directory and run:

```bash
pip install -r requirements.txt
DATA_DIR=PATH_TO_ml-20 USER_MUL=1 ITEM_MUL=1 ./data_gen.sh
```

# run real time inference

```bash
python ncf.py \
    PATH_TO_ml-20x1x1 \
    --layers 256 256 128 64 \
    -f 64 \
    --seed 1 \
    --user_scaling 1 \
    --item_scaling 1 \
    --random_negatives \
    --evaluate --ipex \
    --precision float32

usage: ncf.py [-h] [-e EPOCHS] [-b BATCH_SIZE] [--valid-batch-size VALID_BATCH_SIZE] [-f FACTORS] [--layers LAYERS [LAYERS ...]] [-n NEGATIVE_SAMPLES]
              [-l LEARNING_RATE] [-k TOPK] [--no-cuda] [--seed SEED] [--threshold THRESHOLD] [--valid-negative VALID_NEGATIVE] [--processes PROCESSES]
              [--workers WORKERS] [--beta1 BETA1] [--beta2 BETA2] [--eps EPS] [--user_scaling USER_SCALING] [--item_scaling ITEM_SCALING] [--cpu_dataloader]
              [--random_negatives] [--ipex] [--evaluate] [--precision PRECISION]
              data

Train a Nerual Collaborative Filtering model

positional arguments:
  data                  path to test and training data files

optional arguments:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        number of epochs for training
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        number of examples for each iteration
  --valid-batch-size VALID_BATCH_SIZE
                        number of examples in each validation chunk
  -f FACTORS, --factors FACTORS
                        number of predictive factors
  --layers LAYERS [LAYERS ...]
                        size of hidden layers for MLP
  -n NEGATIVE_SAMPLES, --negative-samples NEGATIVE_SAMPLES
                        number of negative examples per interaction
  -l LEARNING_RATE, --learning-rate LEARNING_RATE
                        learning rate for optimizer
  -k TOPK, --topk TOPK  rank for test examples to be considered a hit
  --no-cuda             use available GPUs
  --seed SEED, -s SEED  manually set random seed for torch
  --threshold THRESHOLD, -t THRESHOLD
                        stop training early at threshold
  --valid-negative VALID_NEGATIVE
                        Number of negative samples for each positive test example
  --processes PROCESSES, -p PROCESSES
                        Number of processes for evaluating model
  --workers WORKERS, -w WORKERS
                        Number of workers for training DataLoader
  --beta1 BETA1, -b1 BETA1
                        beta1 for Adam
  --beta2 BETA2, -b2 BETA2
                        beta1 for Adam
  --eps EPS             eps for Adam
  --user_scaling USER_SCALING
  --item_scaling ITEM_SCALING
  --cpu_dataloader      pre-process data on cpu to save memory
  --random_negatives    do not check train negatives for existence in dataset
  --ipex                use ipex
  --evaluate            evaluate only
  --precision PRECISION
```
