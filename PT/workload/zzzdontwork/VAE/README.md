# Dataset

[CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

Need below files:
img_align_celeba.zip
list_attr_celeba.txt
identity_CelebA.txt
list_bbox_celeba.txt
list_landmarks_align_celeba.txt
list_eval_partition.txt

# Pretrained model

We use random weights to banchmark

# Run

```
python -u test.py \
         -b ${bs} \
         --no-cuda \
         --ipex \
         --jit \
         --precision ${precision} \
         ${dataset}
```

# Help info

```
python test.py -h

usage: test.py [-h] [--dataset {celeba}] [--model {VanillaVAE}]
[--resume PATH] [-b N] [--lr LR] [--wd W] [--sh S]
[--in-channels N] [--latent-dim N] [-t] [-w N] [-p N]
[--img-size N] [--no-cuda] [--gpu GPU] [--ipex] [--jit]
[--precision PRECISION]
DIR

Generic runner for VAE models

positional arguments:
DIR                   path to dataset

optional arguments:
-h, --help            show this help message and exit
--dataset {celeba}    Kind of dataset
--model {VanillaVAE}  Kind of model
--resume PATH         path to latest checkpoint (default: none)
-b N, --batch-size N  batch size (default: 144), this is the total
--lr LR, --learning-rate LR
initial learning rate
--wd W, --weight-decay W
weight decay (default: 0.0)
--sh S, --scheduler-gamma S
weight decay (default: 0.95)
--in-channels N       The input chanels
--latent-dim N        The latent-dim
-t, --profile         Trigger profile on current topology.
-w N, --warmup-iterations N
number of warmup iterations to run
-p N, --print-freq N  print frequency (default: 10)
--img-size N          print frequency (default: 64)
--no-cuda             disable CUDA
--gpu GPU             GPU id to use.
--ipex                use ipex
--jit                 use ipex
--precision PRECISION
                      precision, "float32" or "bfloat16"
```
