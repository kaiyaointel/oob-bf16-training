# OOB models benchmarking

Inference (realtime) scripts with Pytorch


## Prerequisites
### PyTorch
1. pip install torch
2. [build from source](https://github.com/pytorch/pytorch#from-source)

### IPEX (Optional)
1. [build from source](https://github.com/intel/intel-extension-for-pytorch#installation)

## Benchmarking
### Quckly generate realtime performance
```
git clone https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/extended-broad-product.git
cd extended-broad-product
cd PT/workload/gen-efficientnet-pytorch
./launch_benchmark.sh --workspace=${PWD}/logs --precision=bfloat16 --channels_last=1 \
    --batch_size=1 --cores_per_instance=4 --model_name=resnet50
```
### Step by step
find info in the corresponding [workload/gen-efficientnet-pytorch/launch_benchmark.sh](https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/extended-broad-product/-/blob/master/PT/workload/gen-efficientnet-pytorch/launch_benchmark.sh)
```
git clone https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/extended-broad-product.git
cd extended-broad-product/PT/gen-efficientnet-pytorch
# prepare workload
git submodule update --init .
patch -p1 < ../workload/gen-efficientnet-pytorch/gen.patch
cp ../workload/gen-efficientnet-pytorch/main.py .
pip install -r ../workload/requirements.txt
# install corresponding torchvision refer to https://github.com/pytorch/vision#installation
# such as torch1.9(build from source)
pip install --no-deps https://github.com/pytorch/vision/archive/refs/tags/v0.9.1-rc1.tar.gz
# benchmark command
python ./main.py -e --performance --pretrained --dummy --no-cuda -j 1 \
    -w 10 -i 200  -a resnet50 -b 1 --precision bfloat16 --channels_last 1
```
### Parser Tools
1. oneDNN verbose: [dnnl_parser.py](https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/extended-broad-product/-/blob/master/PT/dnnl_parser.py)
2. timeline: [profile_parser.py](https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/extended-broad-product/-/blob/master/PT/profile_parser.py)
