# PnasNet-tf

## Prepare

Git clone this repository, and `cd` into directory for remaining commands
```bash
git clone https://github.com/chenxi116/PNASNet.TF.git && cd PNASNet.TF/
# commit: 338371f
```

Install python packages:
```bash
pip install tensorflow==1.15.2
pip install torchvision
```

### Patch
```bash
git apply PnasNet.patch
```


## pre-trained model and data
### model
```bash
midir data && cd data
wget https://storage.googleapis.com/download.tensorflow.org/models/pnasnet-5_large_2017_12_13.tar.gz
tar -zxvf pnasnet-5_large_2017_12_13.tar.gz
```

### dataset
we use ImageNet, you can find here: mlt-ace.sh.intel.com:/lustre/dataset/imagenet/img/val/


## Running

To do test:
```bash
python main.py --valdir /lustre/dataset/imagenet/img/val/ 
```

we use words/s as the throughput.


## help info
```bash
python main.py -h

usage: main.py [-h] [--valdir VALDIR] [--image_size IMAGE_SIZE]
               [--num_warmup NUM_WARMUP] [--batch_size BATCH_SIZE]
               [--num_iters NUM_ITERS]

optional arguments:
  -h, --help            show this help message and exit
  --valdir VALDIR       path to ImageNet val folder
  --image_size IMAGE_SIZE
                        image size
  --num_warmup NUM_WARMUP
                        num of warmup, default is 10.
  --batch_size BATCH_SIZE
                        batch size, default is 1.
  --num_iters NUM_ITERS
                        total inference iters
```