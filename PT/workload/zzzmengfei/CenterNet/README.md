# Preparation

cd PT/CenterNet

git apply ../workload/CenterNet/Centernet.patch

## Compiling Corner Pooling Layers

```
cd <CenterNet dir>/models/py_utils/_cpools/
python setup.py install --user
```

## Compiling NMS

```
cd <CenterNet dir>/external
make
```

## Installing MS COCO APIs

```
cd <CenterNet dir>/data/coco/PythonAPI
make
```

## Downloading MS COCO Data

- Download the training/validation split we use in our paper from [here](https://drive.google.com/file/d/1dop4188xo5lXDkGtOZUzy2SHOD_COXz4/view?usp=sharing) (originally from [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn/tree/master/data))
- Unzip the file and place `annotations` under `<CenterNet dir>/data/coco`
- Download the images (2014 Train, 2014 Val, 2017 Test) from [here](http://cocodataset.org/#download)
- Create 3 directories, `trainval2014`, `minival2014` and `testdev2017`, under `<CenterNet dir>/data/coco/images/`
- Copy the training/validation/testing images to the corresponding directories according to the annotation files

## Pretrained Model

You can download it from [BaiduYun CenterNet-52](https://pan.baidu.com/s/1xZHB7jq7Hmi0qKu46qnotw) (code: 680t) or [Google Drive CenterNet-52](https://drive.google.com/open?id=14vJYw4P9sxDoltjp5zDkOS3QjUa2zZIP) and put it under `<CenterNet dir>/cache/nnet` (You may need to create this directory by yourself if it does not exist)

# Run

```
cd PT/CenterNet
python -u test.py \
                                            CenterNet-52 \
                                            --testiter 480000 \
                                            --split validation \
                                            --max_iters 20 \
                                            --warmup 10 \
                                            --batch_size ${bs} \
                                            --evaluate \
                                            --ipex \
                                            --precision ${precision}
```

# Help info

```
 python test.py -h
usage: test.py [-h] [--testiter TESTITER] [--split SPLIT] [--suffix SUFFIX]
               [--debug] [--evaluate] [--cuda] [--ipex]
               [--precision PRECISION] [--warmup WARMUP]
               [--max_iters MAX_ITERS] [--batch_size BATCH_SIZE]
               cfg_file

Test CenterNet

positional arguments:
  cfg_file              config file

optional arguments:
  -h, --help            show this help message and exit
  --testiter TESTITER   test at iteration i
  --split SPLIT         which split to use
  --suffix SUFFIX
  --debug
  --evaluate            evaluate performance only
  --cuda                Use CUDA
  --ipex                use ipex
  --precision PRECISION
                        precision, "float32" or "bfloat16"
  --warmup WARMUP       number of warmup
  --max_iters MAX_ITERS
                        max number of iterations to run
  --batch_size BATCH_SIZE
                        batch size

```
