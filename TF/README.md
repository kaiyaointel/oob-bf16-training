# TF models benchmarking

Inference (real time) scripts with Intel-TensorFlow.

## Prerequisites

The model scripts can be run on Linux and require the following
dependencies to be installed:
* [Bazel](https://github.com/bazelbuild/bazel/releases)
* [Python](https://www.python.org/downloads/) 3.5 or later
* [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) v2.x
* `scp/rsync` for copy pre-trained models


## Install Intel-TensorFlow Sample(from tensorflow 2.4.0)
```
WORKSPACE=$PWD

# GPU: alternative TensorFlow 2.4.0
git clone -b utb https://gitlab.devtools.intel.com/TensorFlow/Direct-Optimization/private-tensorflow.git

cd private-tensorflow

bazel clean --expunge --async
echo "yes" |python configure.py
bazel build --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 --copt=-O3 --copt=-Wformat --copt=-Wformat-security \
	--copt=-fstack-protector --copt=-fPIC --copt=-fpic --linkopt=-znoexecstack --linkopt=-zrelro \
	--linkopt=-znow --linkopt=-fstack-protector --config=mkl --define build_with_mkl_dnn_v1_only=true \
	--copt=-DENABLE_INTEL_MKL_BFLOAT16 --copt=-march=native //tensorflow/tools/pip_package:build_pip_package

rm -rf whl_dir && mkdir whl_dir
./bazel-bin/tensorflow/tools/pip_package/build_pip_package whl_dir
pip install whl_dir/tensorflow*.whl

```

## Benchmarking
benchmark tensorflow freezed model with 3 ways.
* with pb file directly. [Support Models](#tensorflow-pb-directly)
```python
python tf_benchmark.py --model_path ${PATH_TO_MODEL}

'''
will get below:
Throughput: 16.0 fps
'''
```

* provide input/output info in `model_detail.py`. [Support Models](#tensorflow-pb-with-input-and-output)
```python
 python tf_benchmark.py --model_path ${PATH_TO_RN50} --model_name resnet_v1-50
    ## model_detail.py
    # resnet50
    {
        'model_name': 'resnet_v1-50',
        'model_dir': 'resnet_v1-50.pb',
        'input': {'map/TensorArrayStack/TensorArrayGatherV3': generate_data([224, 224, 3])},
        'output': ['softmax_tensor']
    }
```
> Tips: model name must be align with the name in the dict of model_detail.py.
> For example: `--model_name resnet_v1-50` align with `'model_name': 'resnet_v1-50'`

* use saved_model. [Support Models](#tensorflow-saved-model)
```python
python tf_savemodel_benchmark.py --model_path CKPT/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_coco_2018_01_28/saved_model/

'''
ls saved_model/
saved_model.pb  variables
'''
```

### TensorFlow PB Directly
Shared: `shareduser@mlt-ace.sh.intel.com` or `shareduser@10.239.60.9`, Password: 1

|	Model	|	Period	|	Model Location	|
|	---------------------	|	---------------	|	-----------------------------	|
|	HugeCTR	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/mlp/HugeCTR/HugeCTR.pb	|
|	i3d-flow	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/action_recognition/i3d/flow/tf/i3d-flow.pb	|
|	i3d-rgb	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/action_recognition/i3d/rgb/tf/i3d-rgb.pb	|
|	densenet-121	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/classification/densenet/121/tf/densenet-121.pb	|
|	densenet-161	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/classification/densenet/161/tf/densenet-161.pb	|
|	densenet-169	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/classification/densenet/169/tf/densenet-169.pb	|
|	googlenet-v1	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/classification/googlenet/v1/tf/googlenet-v1.pb	|
|	googlenet-v2	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/classification/googlenet/v2/tf/googlenet-v2.pb	|
|	googlenet-v3	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/classification/googlenet/v3/tf/googlenet-v3.pb	|
|	googlenet-v4	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/classification/googlenet/v4/tf/googlenet-v4.pb	|
|	image-retrieval-0001	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/classification/image-retrieval-0001/image-retrieval-0001.pb	|
|	inception-resnet-v2	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/classification/inception-resnet/v2/tf/inception-resnet-v2.pb	|
|	nasnet-a-large-331	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/classification/nasnet/large/tf/nasnet-a-large-331.pb	|
|	nasnet-a-mobile-224	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/classification/nasnet/mobile/tf/nasnet-a-mobile-224.pb	|
|	resnet-101	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/classification/resnet/v1/101/tf/resnet-101.pb	|
|	resnet-152	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/classification/resnet/v1/152/tf/resnet-152.pb	|
|	resnet-50	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/classification/resnet/v1/50/tf/official/resnet-50.pb	|
|	resnet-v2-101	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/classification/resnet/v2/101/tf/resnet-v2-101.pb	|
|	resnet-v2-152	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/classification/resnet/v2/152/tf/resnet-v2-152.pb	|
|	resnet-v2-50	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/classification/resnet/v2/50/tf/224x224/resnet-v2-50.pb	|
|	squeezenet1_1	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/classification/squeezenet/1.1/tf/squeezenet1_1.pb	|
|	vgg16	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/classification/vgg/16/tf/vgg16.pb	|
|	vgg19	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/classification/vgg/19/tf/vgg19.pb	|
|	3d-pose-baseline	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/human_pose_estimation/3d-pose-baseline/tf/3d-pose-baseline.pb	|
|	cpm-person	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/human_pose_estimation/cpm/person/tf/cpm-person.pb	|
|	openpose-pose	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/human_pose_estimation/openpose/pose/tf/openpose-pose.pb	|
|	mask_rcnn_inception_v2_coco	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/instance_segmentation/mask_rcnn/mask_rcnn_inception_v2_coco/tf/mask_rcnn_inception_v2_coco.pb	|
|	intel-labs-nonlocal-dehazing	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/IntelLabs/FastImageProcessing/NonlocalDehazing/intel-labs-nonlocal-dehazing.pb	|
|	learning-to-see-in-the-dark-fuji	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/IntelLabs/LearningToSeeInTheDark/Fuji/learning-to-see-in-the-dark-fuji.pb	|
|	learning-to-see-in-the-dark-sony	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/IntelLabs/LearningToSeeInTheDark/Sony/learning-to-see-in-the-dark-sony.pb	|
|	bert-base-uncased_L-12_H-768_A-12	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/language_representation/bert/base/uncased_L-12_H-768_A-12/tf/bert-base-uncased_L-12_H-768_A-12.pb	|
|	vehicle-attributes-barrier-0103	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/object_attributes/vehicle_attributes/tf/vehicle-attributes-barrier-0103.pb	|
|	vehicle-license-plate-detection-barrier-0123	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/object_detection/barrier/tf/0123/vehicle-license-plate-detection-barrier-0123.pb	|
|	faster_rcnn_inception_v2_coco	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/object_detection/common/faster_rcnn/faster_rcnn_inception_v2_coco/tf/faster_rcnn_inception_v2_coco.pb	|
|	faster_rcnn_resnet101_coco	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/object_detection/common/faster_rcnn/faster_rcnn_resnet101_coco/tf/faster_rcnn_resnet101_coco.pb	|
|	faster_rcnn_resnet50_coco	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/object_detection/common/faster_rcnn/faster_rcnn_resnet50_coco/tf/faster_rcnn_resnet50_coco.pb	|
|	faster_rcnn_resnet50_lowproposals_coco	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/object_detection/common/faster_rcnn/faster_rcnn_resnet50_lowproposals_coco/tf/faster_rcnn_resnet50_lowproposals_coco.pb	|
|	retinanet	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/object_detection/common/retinanet/tf/retinanet.pb	|
|	rfcn-resnet101-coco	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/object_detection/common/rfcn/rfcn_resnet101_coco/tf/rfcn-resnet101-coco.pb	|
|	inceptionv2_ssd	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/object_detection/common/ssd_inceptionv2/tf/inceptionv2_ssd.pb	|
|	ssd_resnet50_v1_fpn_coco	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/object_detection/common/ssd_resnet50/ssd_resnet50_v1_fpn_coco/tf/ssd_resnet50_v1_fpn_coco.pb	|
|	yolo-v2	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/object_detection/yolo/yolo_v2/tf/yolo-v2.pb	|
|	yolo-v3	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/object_detection/yolo/yolo_v3/tf/yolo-v3.pb	|
|	yolo-v2-tiny-ava-0001	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/PublicCompressed/detection/tinyYOLOv2/fp32_sparsity00/yolo-v2-tiny-ava-0001.pb	|
|	yolo-v2-tiny-ava-sparse-30-0001	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/PublicCompressed/detection/tinyYOLOv2/fp32_sparsity30/yolo-v2-tiny-ava-sparse-30-0001.pb	|
|	yolo-v2-tiny-ava-sparse-60-0001	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/PublicCompressed/detection/tinyYOLOv2/fp32_sparsity60/yolo-v2-tiny-ava-sparse-60-0001.pb	|
|	yolo-v2-ava-sparse-35-0001	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/PublicCompressed/detection/YOLOv2/fp32_sparsity35/yolo-v2-ava-sparse-35-0001.pb	|
|	yolo-v2-ava-sparse-70-0001	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/PublicCompressed/detection/YOLOv2/fp32_sparsity70/yolo-v2-ava-sparse-70-0001.pb	|
|	icnet-camvid-ava-0001	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/PublicCompressed/semantic_segmentation/icnet-camvid-tf-ws00/icnet-camvid-ava-0001.pb	|
|	icnet-camvid-ava-sparse-30-0001	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/PublicCompressed/semantic_segmentation/icnet-camvid-tf-ws30/icnet-camvid-ava-sparse-30-0001.pb	|
|	icnet-camvid-ava-sparse-60-0001	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/PublicCompressed/semantic_segmentation/icnet-camvid-tf-ws60/icnet-camvid-ava-sparse-60-0001.pb	|
|	darknet19	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/PublicInHouse/classification/darknet19/darknet19.pb	|
|	darknet53	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/PublicInHouse/classification/darknet53/darknet53.pb	|
|	DSSD_12	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/PublicInHouse/object_detection/common/dssd/DSSD_12/tf/DSSD_12.pb	|
|	tiny_yolo_v1	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/PublicInHouse/object_detection/common/yolo/v1_tiny/tf/tiny_yolo_v1.pb	|
|	tiny_yolo_v2	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/PublicInHouse/object_detection/common/yolo/v2_tiny/tf/tiny_yolo_v2.pb	|
|	TCN	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/PublicInHouse/sequence_modelling/tcn/tf/TCN.pb	|
|	rmnet_ssd	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/Retail/action_detection/pedestrian/rmnet_ssd/0028_tf/tf/rmnet_ssd.pb	|
|	icv-emotions-recognition-0002	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/Retail/object_attributes/emotions_recognition/0002/tf/icv-emotions-recognition-0002.pb	|
|	ava-face-recognition-3.0.0	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/Security/feature_extraction/ava/tf/ava-face-recognition-3.0.0.pb	|
|	yolo-v2-tiny-vehicle-detection-0001	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/Security/object_detection/barrier/yolo/yolo-v2-tiny-vehicle-detection-0001/tf/yolo-v2-tiny-vehicle-detection-0001.pb	|
|	ava-person-vehicle-detection-stage2-2.0.0	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/Security/object_detection/common/ava/stage2/tf/ava-person-vehicle-detection-stage2-2.0.0.pb	|
|	person-vehicle-bike-detection-crossroad-yolov3-1020	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/Security/object_detection/crossroad/1020/tf/person-vehicle-bike-detection-crossroad-yolov3-1020.pb	|
|	deeplabv3	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/semantic_segmentation/deeplab/v3/deeplabv3.pb	|
|	ctpn	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/text_detection/ctpn/tf/ctpn.pb	|
|	faster_rcnn_resnet101_fgvc	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ckpt/faster_rcnn_resnet101_fgvc_2018_07_19/faster_rcnn_resnet101_fgvc_2018_07_19/faster_rcnn_resnet101_fgvc.pb	|
|	faster_rcnn_resnet101_kitti	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ckpt/faster_rcnn_resnet101_kitti_2018_01_28/faster_rcnn_resnet101_kitti_2018_01_28/faster_rcnn_resnet101_kitti.pb	|
|	faster_rcnn_resnet101_lowproposals_coco	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ckpt/faster_rcnn_resnet101_lowproposals_coco_2018_01_28/faster_rcnn_resnet101_lowproposals_coco_2018_01_28/faster_rcnn_resnet101_lowproposals_coco.pb	|
|	faster_rcnn_resnet50_fgvc	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ckpt/faster_rcnn_resnet50_fgvc_2018_07_19/faster_rcnn_resnet50_fgvc_2018_07_19/faster_rcnn_resnet50_fgvc.pb	|
|	ssd_inception_v2_coco	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ckpt/ssd_inception_v2_coco_2018_01_28/ssd_inception_v2_coco_2018_01_28/ssd_inception_v2_coco.pb|
|	ssd-resnet34	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ckpt/ssd-resnet34_300x300/ssd_resnet34_300x300.pb	|
|	DeepLab	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/mlp/deeplabv3_mnv2_cityscapes_train/deeplab.pb	|
|	SSD_ResNet50_V1_FPN_640x640(RetinaNet50)	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ckpt/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/SSD_ResNet50_V1_FPN_640x640_RetinaNet50.pb	|
|	DRAW	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/DRAW/DRAW.pb	|
|	Vnet	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/mlp/vnet/vnet.pb	|
|	GraphSage	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/mlp/GraphSage/GraphSage.pb	|
|	WGAN	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/mlp/oob_gan_models/WGAN.pb|	|					
|	CBAM	|	OOB-Q4-20	|  /home2/tensorflow-broad-product/oob_tf_models/mlp/CBAM/CBAM.pb	|
|	ALBERT	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/ALBERT/ALBERT.pb	|
|	BERT-BASE	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/mlp/BERT_BASE/BERT_BASE.pb	|
|	BERT-LARGE	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/mlp/BERT_LARGE/BERT_LARGE.pb	|
|	CharCNN	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/CharCNN/CharCNN.pb	|
|	dilation	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/Dilation/dilation.pb	|
|	faster_rcnn_inception_resnet_v2_atrous_coco	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/object_detection/common/faster_rcnn/faster_rcnn_inception_resnet_v2_atrous_coco/tf/faster_rcnn_inception_resnet_v2_atrous_coco.pb	|
|	faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/ckpt/faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28/faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco.pb	|
|	faster_rcnn_nas_coco	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/ckpt/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_coco_2018_01_28.pb	|
|	faster_rcnn_nas_lowproposals_coco	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/object_detection/common/faster_rcnn/faster_rcnn_nas_lowproposals_coco/tf/faster_rcnn_nas_lowproposals_coco.pb	|
|	GAN	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/mlp/oob_gan_models/GAN.pb	|
|	gmcnn-places2	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/image_inpainting/gmcnn/tf/gmcnn-places2.pb	|
|	LSGAN	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/mlp/oob_gan_models/LSGAN.pb	|
|	mask_rcnn_inception_resnet_v2_atrous_coco	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/instance_segmentation/mask_rcnn/mask_rcnn_inception_resnet_v2_atrous_coco/tf/mask_rcnn_inception_resnet_v2_atrous_coco.pb	|
|	NCF-1B	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/ncf_trained_movielens_1m/NCF-1B.pb	|
|	R-FCN	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/R-FCN/rfcn_resnet101_coco_2018_01_28/R-FCN.pb	|
|	Transformer-LT	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/mlp/transformer_lt_official_fp32_pretrained_model/graph/Transformer-LT.pb	|
|	DynamicMemory	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/oob/checkpoint_dynamic_memory_network/DynamicMemory.pb	|
|	EntityNet	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/oob/checkpoint_entity_network2/EntityNet.pb	|
|	Seq2seqAttn	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/oob/Seq2seqAttn/Seq2seqAttn.pb	|
|	SqueezeNet	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/mlp/SqueezeNet-tf/SqueezeNet.pb	|
|	ssd_resnet34_1200x1200	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/mlp/ssd_resnet34_model/ssd_resnet34_1200x1200.pb	|
|	TextRCNN	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/oob/TextRCNN/TextRCNN.pb	|
|	U-Net	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/mlp/unet/U-Net.pb	|
|	wavenet	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/wavenet/wavenet.pb	|
|	SphereFace	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/SphereFace/SphereFace.pb	|
|	HierAtteNet	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/oob/checkpoint_hier_atten_title/text_hier_atten_title_desc_checkpoint_MHA/HierAtteNet.pb	|
|	ResNet-50_v1.5	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/mlp/ResNet50_v1_5/model_dir/ResNet-50_v1.5.pb	|
|	ResNeXt-50	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/ResNext_50/ResNext_50.pb	|
|	ResNeXt-101	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/ResNext_101/ResNext_101.pb	|
|	COVID-Net	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/oob/COVID-Net/COVID-Net.pb |
|	ACGAN	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/oob/oob_gan_models/ACGAN.pb	|
|	BEGAN	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/oob/oob_gan_models/BEGAN.pb	|
|	CGAN	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/oob/oob_gan_models/CGAN.pb	|
|	DRAGAN	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/oob/oob_gan_models/DRAGAN.pb	|
|	EBGAN	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/oob/oob_gan_models/EBGAN.pb	|
|	infoGAN	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/oob/oob_gan_models/infoGAN.pb	|
|	mask_rcnn_resnet101_atrous_coco	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/instance_segmentation/mask_rcnn/mask_rcnn_resnet101_atrous_coco/tf/mask_rcnn_resnet101_atrous_coco.pb	|
|	mask_rcnn_resnet50_atrous_coco	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/instance_segmentation/mask_rcnn/mask_rcnn_resnet50_atrous_coco/tf/mask_rcnn_resnet50_atrous_coco.pb	|
|	person-vehicle-bike-detection-crossroad-yolov3-1024	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/Security/object_detection/crossroad/1024/tf/person-vehicle-bike-detection-crossroad-yolov3-1024.pb	|
|	srgan	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/image_processing/srgan/tf/srgan	|
|	SSD-ResNet34_1200 x 1200	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/SSD-ResNet34_1200x1200/ssd_resnet34_fp32_1200x1200_pretrained_model.pb	|
|	unet-3d-isensee_2017	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/PublicInHouse/volumetric_segmentation/unet/3d/isensee_2017/tf/unet-3d-isensee_2017.pb	|
|	unet-3d-origin	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/PublicInHouse/volumetric_segmentation/unet/3d/origin/tf/unet-3d-origin.pb	|
|	WGAN_GP	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/oob/oob_gan_models/WGAN_GP.pb	|
|	wide_deep	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/wide_deep/wide_deep.pb	|
|	EfficientDet-D0-512x512	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/EfficientDet/efficientdet-d0/EfficientDet-D0-512x512.pb	|
|	EfficientDet-D1-640x640	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/EfficientDet/efficientdet-d1/EfficientDet-D1-640x640.pb	|
|	EfficientDet-D2-768x768	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/EfficientDet/efficientdet-d2/EfficientDet-D2-768x768.pb	|
|	EfficientDet-D3-896x896	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/EfficientDet/efficientdet-d3/EfficientDet-D3-896x896.pb	|
|	EfficientDet-D4-1024x1024	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/EfficientDet/efficientdet-d4/EfficientDet-D4-1024x1024.pb	|
|	EfficientDet-D5-1280x1280	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/EfficientDet/efficientdet-d5/EfficientDet-D5-1280x1280.pb	|
|	EfficientDet-D6-1280x1280	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/EfficientDet/efficientdet-d6/EfficientDet-D6-1280x1280.pb	|
|	EfficientDet-D7-1536x1536	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/EfficientDet/efficientdet-d7/EfficientDet-D7-1536x1536.pb	|
|	vehicle-license-plate-detection-barrier-0106	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/ov/vehicle-license-plate-detection-barrier-0106/vehicle-license-plate-detection-barrier-0106.pb	|
|	ssd_resnet_101_fpn_oidv4	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/ckpt/ssd_resnet101_v1_fpn/ssd_resnet_101_fpn_oidv4.pb	|
|	key-value-memory-networks	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/key-value-memory-networks/key-value-memory-networks.pb	|
|	NTM-One-Shot	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/NTM-One-Shot/model/NTM-One-Shot.pb	|
|	NetVLAD	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/NetVLAD/NetVLAD.pb	|
|	KeypointNet	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/mlp/keypoint/keypoint.pb (Need **/home2/tensorflow-broad-product/oob_tf_models/mlp/keypoint/cars_with_keypoints** in cur dir) 	|
|	Evolution ensemble	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/simple_net/simple_net.pb	|
|	adv_inception_v3	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/adv_inception_v3/adv_inception_v3.pb	|
|	ens3_adv_inception_v3	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/ens3_inception_v3/ens3_adv_inception_v3.pb	|
|	NCF	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/mlp/NCF/NCF.pb	|
|	YOLOv4	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/mlp/yolov4/YOLOv4.pb	|
|	3DUNet	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/mlp/3D-Unet/3DUNet.pb	|
|	faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/mlp/faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid/faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid.pb	|
|	faster_rcnn_inception_resnet_v2_atrous_oid	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/mlp/faster_rcnn_inception_resnet_v2_atrous_oid/faster_rcnn_inception_resnet_v2_atrous_oid.pb	|
|	faster_rcnn_resnet101_snapshot_serengeti	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/mlp/faster_rcnn_resnet101_snapshot_serengeti/faster_rcnn_resnet101_snapshot_serengeti.pb	|
|	ssd_mobilenet_v1_0.75_depth_300x300_coco	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/mlp/ssd_mobilenet_v1_0.75_depth_300x300_coco/ssd_mobilenet_v1_0.75_depth_300x300_coco.pb	|
|	ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/mlp/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco.pb	|

### TensorFlow PB with input and output. Note that the * model is without optimize "--disable_optimize"
Shared: `shareduser@mlt-ace.sh.intel.com` or `shareduser@10.239.60.9`, Password: 1

|	Model	|	Period	|	Model Location	|
|	---------------------	|	---------------	|	-----------------------------	|
|	efficientnet-b0*	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/classification/efficientnet/b0/tf/efficientnet-b0.pb	|
|	efficientnet-b0_auto_aug*	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/classification/efficientnet/b0_auto_aug/tf/efficientnet-b0_auto_aug.pb	|
|	efficientnet-b5*	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/classification/efficientnet/b5/tf/efficientnet-b5.pb	|
|	efficientnet-b7_auto_aug*	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/classification/efficientnet/b7_auto_aug/tf/efficientnet-b7_auto_aug.pb	|
|	vggvox*	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/voice_recognition/vggvox/vggvox.pb	|
|	aipg-vdcnn	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/AIPG_trained/text_classification/vdcnn/agnews/tf/aipg-vdcnn.pb	|
|	arttrack-coco-multi	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/human_pose_estimation/arttrack/coco/tf/arttrack-coco-multi.pb	|
|	arttrack-mpii-single	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/human_pose_estimation/arttrack/mpii/tf/arttrack-mpii-single.pb	|
|	deepvariant_wgs	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/dna_sequencing/deepvariant/wgs/deepvariant_wgs.pb	|
|	east_resnet_v1_50	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/text_detection/east/tf/east_resnet_v1_50.pb	|
|	facenet-20180408-102900	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/face_recognition/facenet/CASIA-WebFace/tf/facenet-20180408-102900.pb	|
|	handwritten-score-recognition-0003	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/Retail/handwritten-score-recognition/0003/tf/handwritten-score-recognition-0003.pb	|
|	license-plate-recognition-barrier-0007	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/optical_character_recognition/license_plate_recognition/tf/license-plate-recognition-barrier-0007.pb	|
|	optical_character_recognition-text_recognition-tf	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/optical_character_recognition/text_recognition/tf/optical_character_recognition-text_recognition-tf.pb	|
|	PRNet	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/face_reconstruction/PRNet/tf/PRNet.pb	|
|	text-recognition-0012	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/Retail/text_recognition/bilstm_crnn_bilstm_decoder/0012/tf/text-recognition-0012.pb	|
|	Hierarchical_LSTM	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/Hierarchical/Hierarchical_LSTM.pb	|
|	yolo-v3-tiny	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/object_detection/yolo/yolo_v3/yolo-v3-tiny/yolo-v3-tiny-tf/yolo-v3-tiny.pb	|
|	resnet_v2_200	|	OOB-Q3-20	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/Resnet_v2_200/resnet_v2_200.pb	|	
|	CRNN*	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/mlp/CRNN/crnn.pb	|
|	CenterNet	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/mlp/CenterNet/CenterNet.pb	|
|	CapsuleNet	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/CapsuleNet/CapsuleNet.pb	|
|	deepspeech	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/speech_to_text/deepspeech/v1/tf/deepspeech.pb |
|	DIEN(Deep Interest Evolution Network )	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/oob/DIEN/dien.pb	|
|	dense_vnet_abdominal_ct	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/semantic_segmentation/dense_vnet/tf/dense_vnet_abdominal_ct.pb	|
|	TextCNN	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/oob/TextCNN/TextCNN.pb	|
|	TextRNN	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/oob/TextRNN/TextRNN.pb	|
|	Show and Tell	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/oob/show_and_tell/show_and_tell.pb	|
|	MiniGo	|	OOB-Q4-20	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/MiniGo/MiniGo.pb	|
|	NeuMF	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/mlp/NeuMF/NeuMF.pb	|
|	context_rcnn_resnet101_snapshot_serenget	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/ckpt/context_rcnn_resnet101_snapshot_serengeti_2020_06_10/context_rcnn_resnet101_snapshot_serenget.pb	|
|	MANN	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/dpg/MANN/MANN.pb	|
|	pose-ae-multiperson	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/human_pose_estimation/pose-ae/multiperson/tf/pose-ae-multiperson.pb	|
|	pose-ae-refinement	|	OOB-Q1-21	|	/home2/tensorflow-broad-product/oob_tf_models/ov/all_tf_models/human_pose_estimation/pose-ae/refinement/tf/pose-ae-refinement.pb	|


### TensorFlow saved model
Shared: `shareduser@mlt-ace.sh.intel.com` or `shareduser@10.239.60.9`, Password: 1

|	Model	|	Period	|	Model Location	|
|	---------------------	|	---------------	|	-----------------------------	|
|	faster_rcnn_resnet101_ava_v2.1	|	OOB-Q3-20	|/home2/tensorflow-broad-product/oob_tf_models/ckpt/faster_rcnn_resnet101_ava_v2/faster_rcnn_resnet101_ava_v2.1_2018_04_30/saved_model	|

### TensorFlow model benchmark with script

|	Model	|	Period	|	Model BKC	|
|	---------------------	|	---------------	|	-----------------------------	|
|	DLRM	|	OOB-Q3-20	|	[DLRM BKC](https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/extended-broad-product/-/tree/master/TF%2Fworkload%2FDLRM-tf)	|
|	Deep_Speech_2	|	OOB-Q3-20	|	[Deep_Speech_2 BKC](https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/extended-broad-product/-/tree/master/TF%2Fworkload%2FDeep_Speech2)	|
|	VAE-CF	|	OOB-Q3-20	|	[VAE-CF BKC](https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/extended-broad-product/-/tree/master/TF%2Fworkload%2FVAE-CF)	|		
|	adversarial_text	|	OOB-Q4-20	|	[adversarial_text BKC](https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/extended-broad-product/-/tree/master/TF%2Fworkload%2Fadversarial_text)	|
|	GPT2	|	OOB-Q4-20	|	[GPT-2 BKC](https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/extended-broad-product/-/tree/master/TF%2Fworkload%2FGPT-2)	|
|	AttRec	|	OOB-Q4-20	|	[AttRec BKC](https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/extended-broad-product/-/tree/master/TF%2Fworkload%2FAttRec)	|
|	Attention_OCR	|	OOB-Q4-20	|	[Attention_OCR BKC](https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/extended-broad-product/-/tree/master/TF%2Fworkload%2Fattention-ocr)	|
|	Parallel WaveNet	|	OOB-Q4-20	|	[Parallel WaveNet BKC](https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/extended-broad-product/-/tree/master/TF%2Fworkload%2Fparallel_wavenet)	|
|	PNASNet-5	|	OOB-Q4-20	|	[PNASNet-5 BKC](https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/extended-broad-product/-/tree/master/TF%2Fworkload%2FPnasNet-tf)	|
|       CBAM       |       OOB-Q4-20       |       [CBAM BKC](https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/extended-broad-product/-/tree/master/TF%2Fworkload%2FCBAM)     |
|       ResNest50       |       OOB-Q4-20       |       [ResNest50 BKC](https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/extended-broad-product/-/tree/master/TF%2Fworkload%2FResNest)     |
|       ResNest101       |       OOB-Q4-20       |       [ResNest101 BKC](https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/extended-broad-product/-/tree/master/TF%2Fworkload%2FResNest)     |
|       ResNest50-3D       |       OOB-Q4-20       |       [ResNest50-3D BKC](https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/extended-broad-product/-/tree/master/TF%2Fworkload%2FResNest)     |
|	DETR	|	OOB-Q1-21	|	[DETR BKC](https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/extended-broad-product/-/tree/master/TF%2Fworkload%2FDETR)	|
|	DCGAN	|	OOB-Q1-21	|	[DCGAN BKC](https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/extended-broad-product/-/tree/master/TF%2Fworkload%2FDCGAN)	|
|	Elmo	|	OOB-Q1-21	|	[ELMO BKC](https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/extended-broad-product/-/tree/master/TF%2Fworkload%2Felmo)	|
|	Time series LSTM	|	OOB-Q1-21	|	[Time-series-LSTM BKC](https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/extended-broad-product/-/tree/master/TF%2Fworkload%2Ftime_series_LSTM)	|
|	Unet	|	OOB-Q1-21	|	[Unet BKC](https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/extended-broad-product/-/tree/master/TF%2Fworkload%2FUnet)	|
|	centernet_hg104	|	OOB-Q1-21	|	[centernet_hg104 BKC](https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/extended-broad-product/-/tree/master/TF%2Fworkload%2Fcenternet_hg104)	|
|	WD	|	OOB-Q1-21	|	[WD BKC](https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/extended-broad-product/-/tree/master/TF%2Fworkload%2FWD)	|