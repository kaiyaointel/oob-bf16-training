# Prepare

Capture root directory for path and model_dir
```
root=`pwd`
```

Clone tensorflow models
```
git clone https://github.com/tensorflow/models.git && cd models/research/
```

Install required packages:
```
pip install tensorflow
apt-get update && apt install -y protobuf-compiler
pip install cython matplotlib
protoc object_detection/protos/*.proto --python_out=.
```

Pre trained model:
```
/home2/tensorflow-broad-product/oob_tf_models/ckpt/centernet_hg104_1024x1024_coco17
```

Copy benchmarking script
```
cp /root/run_object_detection_saved_model.py .
```

# Running

To run inference using pretrained model:
```
python3 run_object_detection_saved_model.py --num_iter 500 --num_warmup 10
```

Iterations and warmup can be passed by following (Default: num_iter=500, num_warmup=10):
```
--num_iter 5 --num_warmup 5
```
 
Throughput is measured in images/s
