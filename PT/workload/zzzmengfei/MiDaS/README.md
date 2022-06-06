# install dependency package

1. apply patch

```
cd PT/MiDaS
git apply ../workload/midasnet/midasnet.patch
```

2. install requirements

```
conda install opencv --no-deps
```


# pre-trained model && dataset

```
ln -s /home2/pytorch-broad-models/midasnet/model/model-small-70d6b9c8.pt ./model-small-70d6b9c8.pt
cp /home2/pytorch-broad-models/midasnet/dataset/input.tar.gz ./input
tar -zxvf input/input.tar.gz
rm -f input/input.tar.gz
```

# run real time inference

```bash
python -u run.py --model_weights model-small-70d6b9c8.pt  \
				 --model_type small \
				 --ipex \
				 --precision bfloat16 \ # optional
				 --optimize     # optional, jit will decreace the perf

'''
Throughput is: 13.792641 imgs/s
'''

usage: run.py [-h] [-i INPUT_PATH] [-o OUTPUT_PATH] [-m MODEL_WEIGHTS] [-t MODEL_TYPE]
              [--optimize] [--no-optimize] [--ipex] [--precision PRECISION] [-w N]
 
optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_PATH, --input_path INPUT_PATH
                        folder with input images
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        folder for output images
  -m MODEL_WEIGHTS, --model_weights MODEL_WEIGHTS
                        path to the trained weights of model
  -t MODEL_TYPE, --model_type MODEL_TYPE
                        model type: large or small
  --optimize
  --no-optimize
  --ipex                use intel pytorch extension
  --precision PRECISION
                        precision, float32, bfloat16
  -w N, --warmup_iterations N
                        number of warmup iterations to run
```
