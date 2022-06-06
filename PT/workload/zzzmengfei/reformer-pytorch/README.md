# Pretrained model

Use random weights

# Run

```
cd PT/reformer-pytorch
cp ../workload/reformer/benchmark.py .
python -u benchmark.py \
         --ipex \
         --batch_size {bs}\
         --precision ${precision}
```

# Help info

```
python ./benchmark.py -h
create model...
usage: benchmark.py [-h] [--with_cuda] [-b BATCH_SIZE]
                    [--local_rank LOCAL_RANK] [--ipex] [--precision PRECISION]
                    [--warmup WARMUP] [--max_iters MAX_ITERS]

enwik8

optional arguments:
  -h, --help            show this help message and exit
  --with_cuda           use CPU in case there's no GPU support
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        mini-batch size (default: 32)
  --local_rank LOCAL_RANK
                        local rank passed from distributed launcher
  --ipex                use ipex
  --precision PRECISION
                        precision, "float32" or "bfloat16"
  --warmup WARMUP       number of warmup
  --max_iters MAX_ITERS
                        max number of iterations to run 
```
