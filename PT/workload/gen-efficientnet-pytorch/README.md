# Efficient-Net Group (enabled 4 workloads)  
## Xeon  
```
pip install --no-deps Pillow
cd ../../gen-efficientnet-pytorch
python setup.py install
cp ../workload/gen-efficientnet-pytorch/main.py .
```
Apply patch
```
git apply ../workload/gen-efficientnet-pytorch/geneffnet-training.patch
```
### densenet169  
```
python main.py --data /home2/pytorch-broad-models/COCO2017 -a densenet169 --batch-size 32 --performance --epochs 1
```
### densenet201  
```
python main.py --data /home2/pytorch-broad-models/COCO2017 -a densenet201 --batch-size 32 --performance --epochs 1
```
### efficientnet_b5  
```
python main.py --data /home2/pytorch-broad-models/COCO2017 -a efficientnet_b5 --batch-size 32 --performance --epochs 1
```
### efficientnet_b7  
```
python main.py --data /home2/pytorch-broad-models/COCO2017 -a efficientnet_b7 --batch-size 32 --performance --epochs 1
```
### inception_v3  
```
python main.py --data /home2/pytorch-broad-models/COCO2017 -a inception_v3 --batch-size 32 --performance --epochs 1
```
### Dataset  
```
/home2/pytorch-broad-models/COCO2017
```
## CUDA  
```
pip uninstall torch
pip3 install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html -U
cd ../../gen-efficientnet-pytorch
python setup.py install
cp ../workload/gen-efficientnet-pytorch/main-cuda.py .
```
Apply patch
```
git apply ../workload/gen-efficientnet-pytorch/geneffnet-training.patch
```
### densenet169  
```
python main-cuda.py --data /home2/pytorch-broad-models/COCO2017 -a densenet169 --batch-size 32 --performance --epochs 1 --gpu 0
```
### densenet201  
```
python main-cuda.py --data /home2/pytorch-broad-models/COCO2017 -a densenet201 --batch-size 32 --performance --epochs 1 --gpu 0
```
### efficientnet_b5  
```
python main-cuda.py --data /home2/pytorch-broad-models/COCO2017 -a efficientnet_b5 --batch-size 32 --performance --epochs 1 --gpu 0
```
### efficientnet_b7  
```
python main-cuda.py --data /home2/pytorch-broad-models/COCO2017 -a efficientnet_b7 --batch-size 32 --performance --epochs 1 --gpu 0
```
### inception_v3  
```
python main-cuda.py --data /home2/pytorch-broad-models/COCO2017 -a inception_v3 --batch-size 32 --performance --epochs 1 --gpu 0
```
### Dataset  
```
/home2/pytorch-broad-models/COCO2017
```