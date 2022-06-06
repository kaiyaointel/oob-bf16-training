# Pretrained model

Use random weights

# Run

```
cd PT/GCN
git apply ../workload/GCN/GCN.patch
cd GCN
python -u train.py \
         --no-cuda \
         --ipex \
         --evaluate\
         --precision ${precision}
```

# Help info

```
python train.py -h
usage: train.py [-h] [--no-cuda] [--fastmode] [--seed SEED] [--epochs EPOCHS]
                [--lr LR] [--weight_decay WEIGHT_DECAY] [--hidden HIDDEN]
                [--dropout DROPOUT] [--evaluate] [--ipex]
                [--precision PRECISION] [--warmup WARMUP]
                [--max_iters MAX_ITERS]

optional arguments:
  -h, --help            show this help message and exit
  --no-cuda             Disables CUDA training.
  --fastmode            Validate during training pass.
  --seed SEED           Random seed.
  --epochs EPOCHS       Number of epochs to train.
  --lr LR               Initial learning rate.
  --weight_decay WEIGHT_DECAY
                        Weight decay (L2 loss on parameters).
  --hidden HIDDEN       Number of hidden units.
  --dropout DROPOUT     Dropout rate (1 - keep probability).
  --evaluate            evaluation only
  --ipex                use ipex
  --precision PRECISION
                        precision, "float32" or "bfloat16"
  --warmup WARMUP       number of warmup
  --max_iters MAX_ITERS
                        max number of iterations to run
```
