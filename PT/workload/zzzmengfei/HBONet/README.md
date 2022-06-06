# install dependency package

1. apply patch

```
cd PT/HBONet
git apply ../workload/HBONet/HBONet.patch
```

# run real time inference

```bash
python imagenet.py \
                        --dummy \
                        -b ${bs} \
                        -a hbonet \
                        -e \
                        --ipex \
                        --jit \
                        --width-mult ${width_mult} \
                        --max_iters 500 \
                        --precision ${precision} \

```

# Help Info

```
python imagenet.py -h
usage: imagenet.py [-h] [-d DIR] [--data-backend BACKEND] [-a ARCH] [-j N]
                   [--epochs N] [--start-epoch N] [-b N] [--lr LR]
                   [--momentum M] [--wd W] [-p N] [--resume PATH] [-e]
                   [--pretrained] [--world-size WORLD_SIZE] [--rank RANK]
                   [--dist-url DIST_URL] [--dist-backend DIST_BACKEND]
                   [--seed SEED] [--lr-decay LR_DECAY] [--step STEP]
                   [--schedule SCHEDULE [SCHEDULE ...]] [--gamma GAMMA]
                   [--warmup] [-c PATH] [--cuda] [--width-mult WIDTH_MULT]
                   [--input-size INPUT_SIZE] [--weight WEIGHT] [--dummy]
                   [--ipex] [--precision PRECISION] [--jit]
                   [--max_iters MAX_ITERS] [--warmup_iters WARMUP_ITERS]

PyTorch ImageNet

optional arguments:
  -h, --help            show this help message and exit
  -d DIR, --data DIR    path to dataset
  --data-backend BACKEND
  -a ARCH, --arch ARCH  model architecture: alexnet | densenet121 |
                        densenet161 | densenet169 | densenet201 | googlenet |
                        inception_v3 | mnasnet0_5 | mnasnet0_75 | mnasnet1_0 |
                        mnasnet1_3 | mobilenet_v2 | resnet101 | resnet152 |
                        resnet18 | resnet34 | resnet50 | resnext101_32x8d |
                        resnext50_32x4d | shufflenet_v2_x0_5 |
                        shufflenet_v2_x1_0 | shufflenet_v2_x1_5 |
                        shufflenet_v2_x2_0 | squeezenet1_0 | squeezenet1_1 |
                        vgg11 | vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn
                        | vgg19 | vgg19_bn | wide_resnet101_2 |
                        wide_resnet50_2 | hbonet (default: resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --wd W, --weight-decay W
                        weight decay (default: 1e-4)
  -p N, --print-freq N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --world-size WORLD_SIZE
                        number of nodes for distributed training
  --rank RANK           node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --seed SEED           seed for initializing training.
  --lr-decay LR_DECAY   mode for learning rate decay
  --step STEP           interval for learning rate decay in step mode
  --schedule SCHEDULE [SCHEDULE ...]
                        decrease learning rate at these epochs.
  --gamma GAMMA         LR is multiplied by gamma on schedule.
  --warmup              set lower initial learning rate to warm up the
                        training
  -c PATH, --checkpoint PATH
                        path to save checkpoint (default: checkpoints)
  --cuda                Use CUDA
  --width-mult WIDTH_MULT
                        MobileNet model width multiplier.
  --input-size INPUT_SIZE
                        MobileNet model input resolution
  --weight WEIGHT       path to pretrained weight (default: none)
  --dummy               Use dummy data
  --ipex                Use IPEX
  --precision PRECISION
                        Precision, "float32" or "bfloat16"
  --jit                 Use jit script model
  --max_iters MAX_ITERS
                        max iterations to run
  --warmup_iters WARMUP_ITERS
                        iterations to warmup

```
