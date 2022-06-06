# install dependency package

## 1. libs
```
pip install pandas
```

## 2. patch
```
cp inference_arch1_binary.py wide_deep/pytorch_widedeep/
cp wide_deep.patch wide_deep/
git apply wide_deep.patch
```

## 3. install wide&deep
```
pip install -e .
```

# dataset & pre-trained model

you can find `/home2/pytorch-broad-models/widedeep/`, easy to use local dataset by symlink like:
```
cd pytorch_widedeep
ln -s /home2/pytorch-broad-models/widedeep/data data
ln -s /home2/pytorch-broad-models/widedeep/model model
```

# Run

```bash

python -u inference_arch1_binary.py --inf --mkldnn --batch_size 1




python inference_arch1_binary.py -h

usage: inference_arch1_binary.py [-h] [--batch_size BATCH_SIZE]
                                 [--nepoch NEPOCH] [--pretrained PRETRAINED]
                                 [--save_dir SAVE_DIR] [--inf] [--mkldnn]
                                 [--jit]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        input batch size
  --nepoch NEPOCH       number of epochs to train for
  --pretrained PRETRAINED
                        path to pretrained model
  --save_dir SAVE_DIR   Where to store models
  --inf                 inference only
  --mkldnn              Use MKLDNN to get boost.
  --jit                 enable Intel_PyTorch_Extension JIT path
```

