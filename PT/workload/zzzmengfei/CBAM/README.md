# Run

```
cd PT/attention-module
git apply ../workload/CBAM/CBAM.patch
python -u train_imagenet.py \
                                            --arch resnet --depth 50 \
                                            --batch-size ${bs} --lr 0.1 \
                                            --att-type ${model} \
                                            --prefix RESNET50_IMAGENET_CBAM \
                                            --evaluate \
                                            --ipex \
                                            --jit \
                                            --dummy \
                                            --max_iters 100 \
                                            --warmup 5 \
                                            --precision ${precision}
```

# Help info

```
python train_imagenet.py -h
usage: train_imagenet.py [-h] [--data DIR] [--arch ARCH] [--depth D]
                         [--ngpu G] [-j N] [--epochs N] [--start-epoch N]
                         [-b N] [--lr LR] [--momentum M] [--weight-decay W]
                         [--print-freq N] [--resume PATH] [--seed BS] --prefix
                         PFX [--evaluate] [--att-type {BAM,CBAM}] [--cuda]
                         [--dummy] [--ipex] [--jit] [--precision PRECISION]
                         [--warmup WARMUP] [--max_iters MAX_ITERS]

PyTorch ImageNet Training

optional arguments:
  -h, --help            show this help message and exit
  --data DIR            path to dataset
  --arch ARCH, -a ARCH  model architecture: alexnet | densenet121 |
                        densenet161 | densenet169 | densenet201 | googlenet |
                        inception_v3 | mnasnet0_5 | mnasnet0_75 | mnasnet1_0 |
                        mnasnet1_3 | mobilenet_v2 | resnet101 | resnet152 |
                        resnet18 | resnet34 | resnet50 | resnext101_32x8d |
                        resnext50_32x4d | shufflenet_v2_x0_5 |
                        shufflenet_v2_x1_0 | shufflenet_v2_x1_5 |
                        shufflenet_v2_x2_0 | squeezenet1_0 | squeezenet1_1 |
                        vgg11 | vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn
                        | vgg19 | vgg19_bn | wide_resnet101_2 |
                        wide_resnet50_2 (default: resnet18)
  --depth D             model depth
  --ngpu G              number of gpus to use
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256)
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)
  --print-freq N, -p N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  --seed BS             input batch size for training (default: 64)
  --prefix PFX          prefix for logging & checkpoint saving
  --evaluate            evaluation only
  --att-type {BAM,CBAM}
  --cuda                use cuda
  --dummy               use dummy data
  --ipex                use ipex
  --jit                 use ipex
  --precision PRECISION
                        precision, "float32" or "bfloat16", default is
                        "float32"
  --warmup WARMUP       number of warmup
  --max_iters MAX_ITERS
                        max number of iterations to run
```
