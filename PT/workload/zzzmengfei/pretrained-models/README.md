# install dependency package

install torch and torchvision, you can ignore it if you install these packages already.

# dataset

imagenet

## patch

```
cd PT/pretrained-models
git apply ../workload/pretrained-models/vggm.diff
python setup.py install
```

# run training and inference

```
python -u examples/imagenet_eval.py \
                   -e --performance \
                   --pretrained None \
                   --mkldnn \
                   -j 1 \
                   -b 1 \
                   -i 100 \
                   -a ${MODEL_PATH} \
                   --dummy \
```

You can set different models by set -a $model_name which model_name is one of "vggm", "inceptionresnet50", "se_resnext50_32x4d", "dpn92".

## help info

```python
python examples/imagenet_eval.py -h                                                                                                             [16/48802]
usage: imagenet_eval.py [-h] [--data DIR] [--arch ARCH] [-j N] [--epochs N]
                        [--start-epoch N] [-b N] [--lr LR] [--momentum M]
                        [--weight-decay W] [--print-freq N] [--resume PATH]
                        [-e] [--pretrained PRETRAINED]
                        [--do-not-preserve-aspect-ratio] [--mkldnn] [--jit]
                        [--cuda] [-i N] [-w N] [--precision PRECISION] [-t]
                        [--performance] [--dummy]

PyTorch ImageNet Training

optional arguments:
  -h, --help            show this help message and exit
  --data DIR            path to dataset
  --arch ARCH, -a ARCH  model architecture: alexnet | bninception |
                        cafferesnet101 | densenet121 | densenet161 |
                        densenet169 | densenet201 | dpn107 | dpn131 | dpn68 |
                        dpn68b | dpn92 | dpn98 | fbresnet152 |
                        inceptionresnetv2 | inceptionv3 | inceptionv4 |
                        nasnetalarge | nasnetamobile | pnasnet5large | polynet
                        | resnet101 | resnet152 | resnet18 | resnet34 |
                        resnet50 | resnext101_32x4d | resnext101_64x4d |
                        se_resnet101 | se_resnet152 | se_resnet50 |
                        se_resnext101_32x4d | se_resnext50_32x4d | senet154 |
                        squeezenet1_0 | squeezenet1_1 | vgg11 | vgg11_bn |
                        vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19 | vgg19_bn
                        | vggm | xception (default: fbresnet152)
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
  -e, --evaluate        evaluate model on validation set
  --pretrained PRETRAINED
                        use pre-trained model
  --do-not-preserve-aspect-ratio
                        do not preserve the aspect ratio when resizing an
                        image
  --mkldnn              use mkldnn weight cache
  --jit                 enable Intel_PyTorch_Extension JIT path
  --cuda                disable CUDA
  -i N, --iterations N  number of total iterations to run
  -w N, --warmup-iterations N
                        number of warmup iterations to run
  --precision PRECISION
                        precision, float32, int8, bfloat16
  -t, --profile         Trigger profile on current topology.
  --performance         measure performance only, no accuracy.
  --dummy               using dummu data to test the performance of inference

```
