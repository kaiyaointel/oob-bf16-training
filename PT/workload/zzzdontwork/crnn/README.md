# install dependency package

1. lmdb

```
pip install lmdb
```

2. warp-ctc
   install:

```
git clone https://github.com/SeanNaren/warp-ctc.git && \
    cd warp-ctc && \
    mkdir -p build && cd build && cmake .. && make && \
    cd ../pytorch_binding && python setup.py install
```

3. apply patch

```
git apply crnn.patch
```

# dataset & pre-trained model

```
cd ./crnn
```

You need to put dataset and pre-trained model at ./data/ , and you can use Shared account:
**shareduser@mlt-ace.sh.intel.com or shareduser@10.239.60.9, Password: 1**

dataset: `ace:/home2/pytorch-broad-models/CRNN/data/IIIT5k`

model: `ace:/home2/pytorch-broad-models/CRNN/data/crnn.pth`

# run real time inference

```bash
python -u train.py --inf \
                   --pretrained data/crnn.pth \
                   --trainRoot $dataset data/IIII5k/ \
                   --valRoot data/IIII5k/ \
                   --batchSize 1 \
                   --ipex \
                   --precision bfloat16 \ # optional
                   --jit     # jit will decreace the perf

'''
Throughput is: 25.691916 imgs/s
'''

usage: train.py [-h] --trainRoot TRAINROOT --valRoot VALROOT
                [--workers WORKERS] [--batchSize BATCHSIZE] [--imgH IMGH]
                [--imgW IMGW] [--nh NH] [--nepoch NEPOCH] [--cuda]
                [--ngpu NGPU] [--pretrained PRETRAINED] [--alphabet ALPHABET]
                [--expr_dir EXPR_DIR] [--displayInterval DISPLAYINTERVAL]
                [--n_test_disp N_TEST_DISP] [--valInterval VALINTERVAL]
                [--saveInterval SAVEINTERVAL] [--lr LR] [--beta1 BETA1]
                [--adam] [--adadelta] [--keep_ratio] [--manualSeed MANUALSEED]
                [--random_sample] [--inf] [--ipex] [--num_warmup NUM_WARMUP]
                [--max_iter MAX_ITER] [--jit]

optional arguments:
  -h, --help            show this help message and exit
  --trainRoot TRAINROOT
                        path to dataset
  --valRoot VALROOT     path to dataset
  --workers WORKERS     number of data loading workers
  --batchSize BATCHSIZE
                        input batch size
  --imgH IMGH           the height of the input image to network
  --imgW IMGW           the width of the input image to network
  --nh NH               size of the lstm hidden state
  --nepoch NEPOCH       number of epochs to train for
  --cuda                enables cuda
  --ngpu NGPU           number of GPUs to use
  --pretrained PRETRAINED
                        path to pretrained model (to continue training)
  --alphabet ALPHABET
  --expr_dir EXPR_DIR   Where to store samples and models
  --displayInterval DISPLAYINTERVAL
                        Interval to be displayed
  --n_test_disp N_TEST_DISP
                        Number of samples to display when test
  --valInterval VALINTERVAL
                        Interval to be displayed
  --saveInterval SAVEINTERVAL
                        Interval to be displayed
  --lr LR               learning rate for Critic, not used by adadealta
  --beta1 BETA1         beta1 for adam. default=0.5
  --adam                Whether to use adam (default is rmsprop)
  --adadelta            Whether to use adadelta (default is rmsprop)
  --keep_ratio          whether to keep ratio for image resize
  --manualSeed MANUALSEED
                        reproduce experiemnt
  --random_sample       whether to sample the dataset with random sampler
  --inf                 inference only
  --ipex                Use ipex to get boost.
  --precision           Run model with "float32" or "bfloat16"
  --num_warmup NUM_WARMUP
                        number of warm up, default is 5
  --max_iter MAX_ITER   max iterations to run, default is 0. 0 means max iterations is length of dataset 
  --jit                 Use Pytorch jit to get boost
```
