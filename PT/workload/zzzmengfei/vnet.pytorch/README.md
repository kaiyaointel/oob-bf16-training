# install dependency package

1. apply patch

```
cd PT/vnet.pytorch
git apply ../workload/vnet/vnet.patch
```

2. install requirements

```
pip install -r requirements.txt
```


# run real time inference

```bash
python -u main.py --no-cuda \
				  --ipex \
				  --precision bfloat16 \ # optional
				  --jit     # optional, jit will decreace the perf

'''
Throughput is: 86.841739 imgs/s
'''

usage: main.py [-h] [--batchSz BATCHSZ] [--gpu_ids GPU_IDS] [--nEpochs NEPOCHS]
               [--start-epoch N] [--resume PATH] [-e] [--weight-decay W] [--no-cuda]
               [--seed SEED] [--opt {sgd,adam,rmsprop}] [--ipex] [--precision PRECISION]
               [--jit]
 
optional arguments:
  -h, --help            show this help message and exit
  --batchSz BATCHSZ
  --gpu_ids GPU_IDS
  --nEpochs NEPOCHS
  --start-epoch N       manual epoch number (useful on restarts)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --weight-decay W, --wd W
                        weight decay (default: 1e-8)
  --no-cuda
  --seed SEED
  --opt {sgd,adam,rmsprop}
  --ipex                use intel pytorch extension
  --precision PRECISION
                        precision, float32, bfloat16
  --jit                 enable ipex jit fusionpath
```
