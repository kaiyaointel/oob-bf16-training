# 3d-unet  
### Setup
```
cd ../../3d-unet
git apply ../workload/3d-unet/unet3d-training.patch
python setup.py install
cd resources && cp random3D.h5 random3D_copy.h5
cd ../pytorch3dunet
```
### Run
```
python train.py --config ../resources/train_config_ce.yaml --bf16-train-cpu
python train.py --config ../resources/train_config_ce.yaml --bf16-train-cuda
python train.py --config ../resources/train_config_ce.yaml
```
### Dataset  
```
resources/random3D.h5  
resources/random3D_copy.h5  
```