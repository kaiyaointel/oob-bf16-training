# install dependency package

1. apply patch

```
cd PT/pytorch-ssd
git apply ../workload/ssd-vgg16/ssd.patch
```

2. install requirements

```
pip install pandas opencv-python
```


# pre-trained model

```
ln -s /home2/pytorch-broad-models/ssd-vgg16/model/vgg16-ssd-mp-0_7726.pth ./models/vgg16-ssd-mp-0_7726.pth
ln -s /home2/pytorch-broad-models/ssd-vgg16/model/voc-model-labels.txt ./models/voc-model-labels.txt
```

# run real time inference

```bash
python -u eval_ssd.py --net vgg16-ssd \
					  --dataset /home2/pytorch-broad-models/VOC2007 \
					  --trained_model models/vgg16-ssd-mp-0_7726.pth \
					  --label_file models/voc-model-labels.txt \
					  --ipex \
					  --precision bfloat16 \ # optional
					  --jit     # optional, jit will decreace the perf

'''
Throughput is: 17.372409 imgs/s
'''

usage: eval_ssd.py [-h] [--net NET] [--trained_model TRAINED_MODEL]
                   [--dataset_type DATASET_TYPE] [--dataset DATASET] [--label_file LABEL_FILE]
                   [--use_cuda USE_CUDA] [--use_2007_metric USE_2007_METRIC]
                   [--nms_method NMS_METHOD] [--iou_threshold IOU_THRESHOLD]
                   [--eval_dir EVAL_DIR] [--mb2_width_mult MB2_WIDTH_MULT] [--ipex]
                   [--precision PRECISION] [--jit]

optional arguments:
  -h, --help            show this help message and exit
  --net NET             The network architecture, it should be of mb1-ssd, mb1-ssd-lite,
                        mb2-ssd-lite or vgg16-ssd.
  --trained_model TRAINED_MODEL
  --dataset_type DATASET_TYPE
                        Specify dataset type. Currently support voc and open_images.
  --dataset DATASET     The root directory of the VOC dataset or Open Images dataset.
  --label_file LABEL_FILE
                        The label file path.
  --use_cuda USE_CUDA
  --use_2007_metric USE_2007_METRIC
  --nms_method NMS_METHOD
  --iou_threshold IOU_THRESHOLD
                        The threshold of Intersection over Union.
  --eval_dir EVAL_DIR   The directory to store evaluation results.
  --mb2_width_mult MB2_WIDTH_MULT
                        Width Multiplifier for MobilenetV2
  --ipex                use intel pytorch extension
  --precision PRECISION
                        precision, float32, bfloat16
  --jit                 enable ipex jit fusionpath
```
