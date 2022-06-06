## YOLOV2  
Enter dir  
```
cd ../../YOLOV2
```
Apply patch  
```
cp ../workload/YOLOV2/yolov2-training.patch .
git apply yolov2-training.patch
```
Copy this empty file to /models (otherwise models.yolo_v2 can't be imported)
```
cp ../workload/YOLOV2/__init__.py ./models
```
Copy weights and rename
```
cp /home2/pytorch-broad-models/YOLOV2/yolo_v2_250epoch_77.1_78.1.pth ./backbone/weights
mv backbone/weights/yolo_v2_250epoch_77.1_78.1.pth backbone/weights/darknet19_72.96.pth
```
or
```
cp ../workload/YOLOV2/darknet19_72.96.pth ./backbone/weights
```
Run train
```
python train_voc.py
```
### Dataset  
```
/home2/pytorch-broad-models/YOLOV2/VOCdevkit/VOC2007
```