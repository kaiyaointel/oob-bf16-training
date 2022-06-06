# DSSD_12
python mo_tf.py \
    --input_model ./all_tf_models/PublicInHouse/object_detection/common/dssd/DSSD_12/tf/DSSD_12.pb \
    --reverse_input_channels \
    --input_shape=[1,192,192,3] \
    --input=image_tensor \
    --transformations_config=/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json \
    --tensorflow_object_detection_api_pipeline_config=./all_tf_models/PublicInHouse/object_detection/common/dssd/DSSD_12/tf/pipeline.config \
    --output=detection_scores,detection_boxes,num_detections \
    --output_dir tf_ov/DSSD_12

# efficient-b0
python mo_tf.py \
    --input_model ./all_tf_models/classification/efficientnet/b0/tf/efficientnet-b0.pb \
    --reverse_input_channels \
    --input_shape=[1,224,224,3] \
    --input=0:sub \
    --output=logits \
    --output_dir tf_ov/efficientnet-b0

# efficientnet-b0_auto_aug
python mo_tf.py \
    --input_model ./all_tf_models/classification/efficientnet/b0_auto_aug/tf/efficientnet-b0_auto_aug.pb \
    --reverse_input_channels \
    --input_shape=[1,224,224,3] \
    --input=0:sub \
    --output=logits \
    --output_dir tf_ov/efficientnet-b0_auto_aug

# efficient-b5
python mo_tf.py \
    --input_model ./all_tf_models/classification/efficientnet/b5/tf/efficientnet-b5.pb \
    --reverse_input_channels \
    --input_shape=[1,456,456,3] \
    --input=0:sub \
    --output=logits \
    --output_dir tf_ov/efficientnet-b5

# efficientnet-b7_auto_aug
python mo_tf.py \
    --input_model ./all_tf_models/classification/efficientnet/b7_auto_aug/tf/efficientnet-b7_auto_aug.pb \
    --reverse_input_channels \
    --input_shape=[1,600,600,3] \
    --input=0:sub \
    --output=logits \
    --output_dir tf_ov/efficientnet-b7_auto_aug

# faster_rcnn_inception_v2_coco
python mo_tf.py \
    --reverse_input_channels \
    --input_shape=[1,600,600,3] \
    --input=image_tensor \
    --transformations_config=/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json \
    --tensorflow_object_detection_api_pipeline_config=./all_tf_models/object_detection/common/faster_rcnn/faster_rcnn_inception_v2_coco/tf/pipeline.config \
    --output=detection_scores,detection_boxes,num_detections \
    --input_model=./all_tf_models/object_detection/common/faster_rcnn/faster_rcnn_inception_v2_coco/tf/faster_rcnn_inception_v2_coco.pb \
    --output_dir tf_ov/faster_rcnn_inception_v2_coco

# faster_rcnn_resnet101_ava_v2.1
python mo_tf.py \
    --input_shape=[1,600,600,3] \
    --input=image_tensor \
    --transformations_config=/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json \
    --tensorflow_object_detection_api_pipeline_config=./CKPT/faster_rcnn_resnet101_ava_v2/faster_rcnn_resnet101_ava_v2.1_2018_04_30/pipeline.config \
    --output=detection_scores,detection_boxes,num_detections \
    --input_model=./CKPT/faster_rcnn_resnet101_ava_v2/faster_rcnn_resnet101_ava_v2.1_2018_04_30/frozen_inference_graph.pb \
    --output_dir tf_ov/faster_rcnn_resnet101_ava_v2.1

# faster_rcnn_resnet101_coco
python mo_tf.py \
    --reverse_input_channels \
    --input_shape=[1,600,600,3] \
    --input=image_tensor \
    --transformations_config=/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json \
    --tensorflow_object_detection_api_pipeline_config=./all_tf_models/object_detection/common/faster_rcnn/faster_rcnn_resnet101_coco/tf/pipeline.config \
    --output=detection_scores,detection_boxes,num_detections \
    --input_model=./all_tf_models/object_detection/common/faster_rcnn/faster_rcnn_resnet101_coco/tf/faster_rcnn_resnet101_coco.pb \
    --output_dir tf_ov/faster_rcnn_resnet101_coco

# faster_rcnn_resnet101_fgvc
python mo_tf.py \
    --input_shape=[1,600,600,3] \
    --input=image_tensor \
    --transformations_config=/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json \
    --tensorflow_object_detection_api_pipeline_config=./CKPT/faster_rcnn_resnet101_fgvc_2018_07_19/faster_rcnn_resnet101_fgvc_2018_07_19/pipeline.config \
    --output=detection_scores,detection_boxes,num_detections \
    --input_model=./CKPT/faster_rcnn_resnet101_fgvc_2018_07_19/faster_rcnn_resnet101_fgvc_2018_07_19/frozen_inference_graph.pb \
    --output_dir tf_ov/faster_rcnn_resnet101_fgvc

# faster_rcnn_resnet101_kitti
python mo_tf.py \
    --input_shape=[1,600,600,3] \
    --input=image_tensor \
    --transformations_config=/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json \
    --tensorflow_object_detection_api_pipeline_config=./CKPT/faster_rcnn_resnet101_kitti_2018_01_28/faster_rcnn_resnet101_kitti_2018_01_28/pipeline.config \
    --output=detection_scores,detection_boxes,num_detections \
    --input_model=./CKPT/faster_rcnn_resnet101_kitti_2018_01_28/faster_rcnn_resnet101_kitti_2018_01_28/frozen_inference_graph.pb \
    --output_dir tf_ov/faster_rcnn_resnet101_kitti

# faster_rcnn_resnet101_lowproposals_coco
python mo_tf.py \
    --input_shape=[1,600,600,3] \
    --input=image_tensor \
    --transformations_config=/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json \
    --tensorflow_object_detection_api_pipeline_config=./CKPT/faster_rcnn_resnet101_lowproposals_coco_2018_01_28/faster_rcnn_resnet101_lowproposals_coco_2018_01_28/pipeline.config \
    --output=detection_scores,detection_boxes,num_detections \
    --input_model=./CKPT/faster_rcnn_resnet101_lowproposals_coco_2018_01_28/faster_rcnn_resnet101_lowproposals_coco_2018_01_28/frozen_inference_graph.pb \
    --output_dir tf_ov/faster_rcnn_resnet101_lowproposals_coco


# faster_rcnn_resnet50_coco
python mo_tf.py \
    --reverse_input_channels \
    --input_shape=[1,600,600,3] \
    --input=image_tensor \
    --transformations_config=/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json \
    --tensorflow_object_detection_api_pipeline_config=./all_tf_models/object_detection/common/faster_rcnn/faster_rcnn_resnet50_coco/tf/pipeline.config \
    --output=detection_scores,detection_boxes,num_detections \
    --input_model=./all_tf_models/object_detection/common/faster_rcnn/faster_rcnn_resnet50_coco/tf/faster_rcnn_resnet50_coco.pb \

# faster_rcnn_resnet50_fgvc
python mo_tf.py \
    --input_shape=[1,600,600,3] \
    --input=image_tensor \
    --transformations_config=/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json \
    --tensorflow_object_detection_api_pipeline_config=./CKPT/faster_rcnn_resnet50_fgvc_2018_07_19/faster_rcnn_resnet50_fgvc_2018_07_19/pipeline.config \
    --output=detection_scores,detection_boxes,num_detections \
    --input_model=./CKPT/faster_rcnn_resnet50_fgvc_2018_07_19/faster_rcnn_resnet50_fgvc_2018_07_19/frozen_inference_graph.pb \
    --output_dir tf_ov/faster_rcnn_resnet50_fgvc 

# faster_rcnn_resnet50_lowproposals_coco
python mo_tf.py \
    --reverse_input_channels \
    --input_shape=[1,600,600,3] \
    --input=image_tensor \
    --transformations_config=/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json \
    --tensorflow_object_detection_api_pipeline_config=./all_tf_models/object_detection/common/faster_rcnn/faster_rcnn_resnet50_lowproposals_coco/tf/pipeline.config \
    --output=detection_scores,detection_boxes,num_detections \
    --input_model=./all_tf_models/object_detection/common/faster_rcnn/faster_rcnn_resnet50_lowproposals_coco/tf/faster_rcnn_resnet50_lowproposals_coco.pb \
    --output_dir tf_ov/faster_rcnn_resnet50_lowproposals_coco

# Hierarchical_LSTM
python mo_tf.py \
    --input_shape=[1,1024,27],[1,1024,27] \
    --input=batch_in,batch_out \
    --output=map_1/while/output_module_vars/prediction \
    --input_model=./dpg/Hierarchical/text8_freeze.pb \
    --output_dir tf_ov/Hierarchical_LSTM


# HugeCTR
python mo_tf.py \
    --input_shape=[1,13],[1,26,1] \
    --input=dense-input,sparse-input \
    --output=add_14 \
    --input_model=./mlp/HugeCTR/dcn_model-0_freeze.pb \
    --output_dir tf_ov/HugeCTR

# squeuenet1_1
python mo_tf.py \
    --reverse_input_channels \
    --input_model=./all_tf_models/classification/squeezenet/1.1/tf/squeezenet_v1.1.pb \
    --output_dir=tf_ov/squeuenet1_1 \
    --input_shape=[1,224,224,3] \
    --input=Placeholder \
    --output=softmax_tensor \
    --scale_values=Placeholder[255.0] 

 # ssd_inception_v2_coco
 python mo_tf.py \
     --input_model=./CKPT/ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb \
     --output_dir=tf_ov/ssd_inception_v2_coco/ \
     --input_shape=[1,300,300,3] \
     --transformations_config=/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json \
     --tensorflow_object_detection_api_pipeline_config=./CKPT/ssd_inception_v2_coco_2018_01_28/pipeline.config 

 # ssd-resnet34_300x300
 python mo_tf.py \
     --input_model=./CKPT/ssd_resnet34_fp32_bs1_pretrained_model/ssd_resnet34_fp32_bs1_pretrained_model.pb \
     --output_dir=tf_ov/ssd-resnet34_300x300 \
     --input_shape=[1,300,300,3] \
     --transformations_config=/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json 

# text-recognition-0012
python mo_tf.py \
    --reverse_input_channels \
    --input_shape=[1,32,120,1] \
    --input=Placeholder \
    --output=shadow/LSTMLayers/transpose_time_major \
    --input_model=./all_tf_models/Retail/text_recognition/bilstm_crnn_bilstm_decoder/0012/tf/text-recognition-0012.pb \
    --output_dir=tf_ov/text-recognition-0012

# vehicle-license-plate-detection-barrier-0123
python mo_tf.py \
    --reverse_input_channels \
    --input_shape=[1,256,256,3] \
    --input=Placeholder \
    --mean_values=Placeholder[127.5,127.5,127.5] \
    --scale_values=Placeholder[127.5] \
    --transformations_config=./all_tf_models/object_detection/barrier/tf/0123/model.tfmo.json \
    --output=ssd_heads/concat_reshape_softmax/mbox_loc_final,ssd_heads/concat_reshape_softmax/mbox_conf_final,ssd_heads/concat_reshape_softmax/mbox_priorbox \
    --input_model=./all_tf_models/object_detection/barrier/tf/0123/vehicle-license-plate-detection-barrier-0123.pb \
    --output_dir=tf_ov/vehicle-license-plate-detection-barrier-0123

# 3d-pose-baseline
python mo_tf.py \
    --input_shape=[1,32] \
    --input=inputs/enc_in \
    --output=linear_model/add_1 \
    --input_model=./all_tf_models/human_pose_estimation/3d-pose-baseline/tf/3d-pose-baseline.pb \
    --output_dir=tf_ov/3d-pose-baseline


# facenet-20180408-102900
python mo_tf.py \
    --freeze_placeholder_with_value "phase_train->False" \
    --reverse_input_channels \
    --input_shape=[1,160,160,3],[1] \
    --input=image_batch,phase_train \
    --mean_values=image_batch[127.5,127.5,127.5] \
    --scale_values=image_batch[128.0] \
    --output=embeddings \
    --input_model=./all_tf_models/face_recognition/facenet/CASIA-WebFace/tf/facenet-20180408-102900.pb \
    --output_dir=tf_ov/facenet-20180408-102900

# inceptionv2_ssd
python mo_tf.py \
    --input_shape=[1,600,600,3] \
    --input=image_tensor \
    --transformations_config=/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json \
    --tensorflow_object_detection_api_pipeline_config=./all_tf_models/object_detection/common/ssd_inceptionv2/tf/pipeline.config \
    --output=detection_classes,detection_scores,detection_boxes,num_detections \
    --input_model=./all_tf_models/object_detection/common/ssd_inceptionv2/tf/inceptionv2_ssd.pb \
    --output_dir=tf_ov/
