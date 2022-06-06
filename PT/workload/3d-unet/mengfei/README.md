# install dependency package

## 1. libs

```
pip install hdbscan scikit-image
```

## 2. patch

```
cd PT/3d-unet
git apply ../workload/3D-UNet/UNet3D.patch
```

## 3. install 3DUNet

```
cd 3d-unet && python setup.py install
```

# dataset & pre-trained model

## dataset

```
cd resources && cp random3D.h5 random3D_copy.h5
```

## Pretrained model

you can find `/tmp/pengxiny/3DUNet/3dunet/best_checkpoint.pytorch`, easy to use local dataset by symlink like:

```
cd pytorch3dunet && mkdir 3dunet
ln -s /tmp/pengxiny/3DUNet/3dunet/best_checkpoint.pytorch ./3dunet/best_checkpoint.pytorch
```

## Run

```python
cd PT/3d-unet/pytorch3dunet
python predict.py --config ../resources/test_config_dice.yaml 
                                --ipex \
                                --jit \
                                --precision float32 \

usage: predict.py [-h] --config CONFIG [--mkldnn] [--jit] [--profiling]
                  [--batch_size BATCH_SIZE] [--num_warmup NUM_WARMUP]

UNet3D

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the YAML config file
  --mkldnn              Use intel pytorch extension.
  --jit                 enable jit optimization in intel pytorch extension.
  --profiling           Do profiling.
  --batch_size BATCH_SIZE
                        input batch size.
  --num_warmup NUM_WARMUP
                        num of warmup, default is 10.

```
