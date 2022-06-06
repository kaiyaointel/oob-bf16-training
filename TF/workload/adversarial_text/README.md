install dependency package

## 3rd-party repo

```
 git submodule update --init

 cp diff.patch ../models/research/adversarial_text
```


## patch

```
cd ../models/research/adversarial_text
git apply diff.patch
```

## dataset and pre-trained models

### dataset
/home2/tensorflow-broad-product/mlp/adversarial_text/IMBD

### pre-trained models
/home2/tensorflow-broad-product/mlp/adversarial_text/imdb_pretrain

## run benchmark

```bash
export TRAIN_DIR=/home2/tensorflow-broad-product/mlp/adversarial_text/imdb_pretrain
export IMDB_DATA_DIR=/home2/tensorflow-broad-product/mlp/adversarial_text/IMBD

python evaluate.py \
        --checkpoint_dir=$TRAIN_DIR \
        --eval_data=test \
        --run_once \
        --num_examples=200 \
        --data_dir=$IMDB_DATA_DIR \
        --vocab_size=87007 --embedding_dims=256 --rnn_cell_size=1024 \
        --batch_size=1 \
        --num_timesteps=50 \
        --normalize_embeddings
```