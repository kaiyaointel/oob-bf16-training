# Prepare

## pretrained model

Download pretrained model follow up [this README](https://github.com/yjh0410/yolov2-yolov3_PyTorch/blob/master/weights/README.md#yolo-v2-v3-and-tiny-model)

## dataset

Download the VOC2007 dataset from [http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar](http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar)
then unzip and copy: cp -R ./VOCdevkit to required destination

## patch

```bash
cd PT/YOLOV2
git apply ../workload/YOLOV2/YOLOV2.patch
```

# Benchmaring

```python
python ./eval_voc.py \
    --ipex \
    --precision float32 \
    --trained_model ${checkpoint} \
    --voc_root ${dataset} \
```

## help info

```python
python ./eval_voc.py -h
usage: eval_voc.py [-h] [-v VERSION] [-d DATASET]
                   [--trained_model TRAINED_MODEL] [--save_folder SAVE_FOLDER]
                   [--gpu_ind GPU_IND] [--top_k TOP_K] [--cuda]
                   [--voc_root VOC_ROOT] [--cleanup CLEANUP] [--ipex]
                   [--precision PRECISION] [--max_iters MAX_ITERS]
                   [--warmup WARMUP]

YOLO-v2 Detector Evaluation

optional arguments:
  -h, --help            show this help message and exit
  -v VERSION, --version VERSION
                        yolo_v2, yolo_v3, slim_yolo_v2, tiny_yolo_v3.
  -d DATASET, --dataset DATASET
                        VOC or COCO dataset
  --trained_model TRAINED_MODEL
                        Trained state_dict file path to open
  --save_folder SAVE_FOLDER
                        File path to save results
  --gpu_ind GPU_IND     To choose your gpu.
  --top_k TOP_K         Further restrict the number of predictions to parse
  --cuda                Use cuda
  --voc_root VOC_ROOT   Location of VOC root directory
  --cleanup CLEANUP     Cleanup and remove results files following eval
  --ipex                Use ipex
  --precision PRECISION
                        precision, "float32" or "bfloat16"
  --max_iters MAX_ITERS
                        max number to run.
  --warmup WARMUP       warmup number.
```
