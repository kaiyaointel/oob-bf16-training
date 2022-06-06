# Prepare

Capture root directory for path and model_dir
```
root=`pwd`
```

Git clone this repository, and `cd` into directory for remaining commands
```
git clone https://github.com/Visual-Behavior/detr-tensorflow.git && cd detr-tensorflow
```

Install required packages:
```
pip install tensorflow
pip install -r requirements.txt
pip --no-cache-dir install opencv-python imgaug
apt-get update && apt-get install -y libgl1-mesa-glx
```

Patch
```
cp /root/detr.patch .
git apply detr.patch
```

Dataset:
```
/home2/tensorflow-broad-product/oob_tf_models/dpg/DETR
```

# Running

To run inference using pretrained model:
```
python eval.py --data_dir /home2/tensorflow-broad-product/oob_tf_models/dpg/DETR --img_dir val2017 --ann_file annotations/instances_val2017.json
```

Iterations and warmup can be passed by following (Default: num_iter=500, num_warmup=10):
```
--num_iter 10 --num_warmup 5
```

Default batch size = 1
Throughput is measured in samples per second
