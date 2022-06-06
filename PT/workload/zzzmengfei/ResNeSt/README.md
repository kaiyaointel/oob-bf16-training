# Dataset

Dummy data

# Pretrained model

Load weights automatically

# Run

```
python -u scripts/torch/verify.py \
         --model ${model} \
         --batch-size ${bs} \
         --no-cuda \
         --ipex \
         --jit \
         --dummy \
         --precision ${precision} \
         --workers 1 \
```

# Help info

```
python scripts/torch/verify.py -h
usage: verify.py [-h] [--base-size BASE_SIZE] [--crop-size CROP_SIZE]
                 [--model MODEL] [--batch-size N] [--workers N] [--no-cuda]
                 [--seed S] [--resume RESUME] [--verify VERIFY] [--ipex]
                 [--jit] [--precision PRECISION] [--warmup WARMUP]
                 [--max_iters MAX_ITERS] [--dummy]

Deep Encoding

optional arguments:
  -h, --help            show this help message and exit
  --base-size BASE_SIZE
                        base image size
  --crop-size CROP_SIZE
                        crop image size
  --model MODEL         network model type (default: densenet)
  --batch-size N        batch size for training (default: 128)
  --workers N           dataloader threads
  --no-cuda             disables CUDA training
  --seed S              random seed (default: 1)
  --resume RESUME       put the path to resuming file if needed
  --verify VERIFY       put the path to resuming file if needed
  --ipex                use ipex
  --jit                 use ipex
  --precision PRECISION
                        precision, "float32" or "bfloat16"
  --warmup WARMUP       number of warmup
  --max_iters MAX_ITERS
                        max number of iterations to run
  --dummy               use dummy data
```
