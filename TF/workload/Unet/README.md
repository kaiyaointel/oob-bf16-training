# Prepare

Capture root directory for path and model_dir
```
root=`pwd`
```

Git clone this repository, and `cd` into directory for remaining commands
```
git clone https://github.com/divamgupta/image-segmentation-keras.git && cd image-segmentation-keras
```

Install required packages:
```
pip install tensorflow
apt-get update && apt-get install -y libgl1-mesa-glx
apt-get install -y libsm6 libxext6 libxrender-dev
pip --no-cache-dir install opencv-python imgaug
python setup.py install
```

Benchmark script
```
cp /root/run_benchmark.py .
```

Dataset:
```
/home2/tensorflow-broad-product/oob_tf_models/oob/Unet/dataset
```

Pre-trained-model:
```
/home2/tensorflow-broad-product/oob_tf_models/oob/Unet/checkpoint
```

# Running

To run inference using pretrained model:
```
python run_benchmark.py --num_warmup 5 --num_iter 5
```

Iterations and warmup can be passed by following (Default: num_iter=500, num_warmup=10):
```
--num_iter 5 --num_warmup 5
```
Please Note: Number of iterations is multiples of 100 because the test data has 100 images
 
Throughput is measured in it/s
