# Prepare
Capture root directory for path and model_dir
```
root=`pwd`
```

Git clone this repository, `cd` into directory and checkout r0.7 branch
```
git clone --recurse-submodules https://github.com/mlperf/inference.git && cd inference
git checkout r0.7
cd vision/classification_and_detection
```

Install python packages:
```
pip install tensorflow
```

Build and install the benchmark:
```
cd ../../loadgen ; CFLAGS="-std=c++1y" python setup.py develop --user ; cd ../vision/classification_and_detection
python setup.py develop
```

Patch
```
cp /root/resnet50.patch .
git apply resnet50.patch
```

Set environment variables to tell the benchmark where to find model and data and to set the batch size:
```
export MODEL_DIR=/home2/tensorflow-broad-product/oob_tf_models/mlp/ResNet50_v1_5/model_dir/resnet50_v1_5.pb
export DATA_DIR=/home2/tensorflow-broad-product/oob_tf_models/mlp/ResNet50_v1_5/Imagenet_2012Val
export EXTRA_OPS="--max-batchsize 1"
```

# Running
```
./run_local.sh tf resnet50 cpu 
```
Throughput is measured in queries per second (qps)
