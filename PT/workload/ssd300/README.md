# ssd300
```
pip install pycocotools
pip install mlperf_compliance
pip install apex (CUDA only)
cd ../../ssd300/single_stage_detector/ssd
git apply ../../../workload/ssd300/ssd300-training.patch
python train.py --epochs 1
```