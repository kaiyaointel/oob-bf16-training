set -x

CONDA_PATH=`which activate`
DE_CONDA_PATH=`which deactivate`

CUR_PATH=`pwd`
workspace="$CUR_PATH/OOB_TF_Logs/"
TF_pretrain_path="/home2/tensorflow-broad-product/oob_tf_models"
precision="float32"
run_perf=1
collect_dnnl_verbose=0
use_TF_NativeFormat=0


for var in $@
    do
        case $var in
            --conda_name=*)
                conda_name=$(echo $var |cut -f2 -d=)
            ;;
            --workspace=*)
                workspace=$(echo $var |cut -f2 -d=)
            ;;
            --TF_pretrain_path=*)
                TF_pretrain_path=$(echo $var |cut -f2 -d=)
            ;;
            --framework=*)
                framework=$(echo $var |cut -f2 -d=)
            ;;
            --framework_version=*)
                framework_version=$(echo $var |cut -f2 -d=)
            ;;
            --precision=*)
                precision=$(echo $var |cut -f2 -d=)
            ;;
            --mode=*)
                mode=$(echo $var |cut -f2 -d=)
            ;;
            --batch_size=*)
                batch_size=($(echo $var |cut -f2 -d= |sed 's/,/ /g'))
            ;;
            --cores_per_instance=*)
                cores_per_instance=($(echo $var |cut -f2 -d= |sed 's/,/ /g'))
            ;;
            --checkpoint=*)
                checkpoint=$(echo $var |cut -f2 -d=)
            ;;
            --model_name=*)
                model_name=$(echo $var |cut -f2 -d=)
            ;;
            --run_perf=*)
                run_perf=$(echo $var |cut -f2 -d=)
            ;;
            --collect_dnnl_verbose=*)
                collect_dnnl_verbose=$(echo $var |cut -f2 -d=)
            ;;
            --use_TF_NativeFormat=*)
                use_TF_NativeFormat=$(echo $var |cut -f2 -d=)
            ;;
            *)
                echo "Error: No such parameter: ${var}"
                exit 1
            ;;
        esac
     done

source ${CONDA_PATH} ${conda_name}
which python

model_path=(
## Q3-2020
${TF_pretrain_path}/mlp/HugeCTR/HugeCTR.pb
${TF_pretrain_path}/ov/all_tf_models/action_recognition/i3d/flow/tf/i3d-flow.pb
${TF_pretrain_path}/ov/all_tf_models/action_recognition/i3d/rgb/tf/i3d-rgb.pb
${TF_pretrain_path}/ov/all_tf_models/classification/densenet/121/tf/densenet-121.pb
${TF_pretrain_path}/ov/all_tf_models/classification/densenet/161/tf/densenet-161.pb
${TF_pretrain_path}/ov/all_tf_models/classification/densenet/169/tf/densenet-169.pb
${TF_pretrain_path}/ov/all_tf_models/classification/googlenet/v1/tf/googlenet-v1.pb
${TF_pretrain_path}/ov/all_tf_models/classification/googlenet/v2/tf/googlenet-v2.pb
${TF_pretrain_path}/ov/all_tf_models/classification/googlenet/v3/tf/googlenet-v3.pb
${TF_pretrain_path}/ov/all_tf_models/classification/googlenet/v4/tf/googlenet-v4.pb
${TF_pretrain_path}/ov/all_tf_models/classification/image-retrieval-0001/image-retrieval-0001.pb
${TF_pretrain_path}/ov/all_tf_models/classification/inception-resnet/v2/tf/inception-resnet-v2.pb
${TF_pretrain_path}/ov/all_tf_models/classification/nasnet/large/tf/nasnet-a-large-331.pb
${TF_pretrain_path}/ov/all_tf_models/classification/nasnet/mobile/tf/nasnet-a-mobile-224.pb
${TF_pretrain_path}/ov/all_tf_models/classification/resnet/v1/101/tf/resnet-101.pb
${TF_pretrain_path}/ov/all_tf_models/classification/resnet/v1/152/tf/resnet-152.pb
${TF_pretrain_path}/ov/all_tf_models/classification/resnet/v1/50/tf/official/resnet-50.pb
${TF_pretrain_path}/ov/all_tf_models/classification/resnet/v2/101/tf/resnet-v2-101.pb
${TF_pretrain_path}/ov/all_tf_models/classification/resnet/v2/152/tf/resnet-v2-152.pb
${TF_pretrain_path}/ov/all_tf_models/classification/resnet/v2/50/tf/224x224/resnet-v2-50.pb
${TF_pretrain_path}/ov/all_tf_models/classification/squeezenet/1.1/tf/squeezenet1_1.pb
${TF_pretrain_path}/ov/all_tf_models/classification/vgg/16/tf/vgg16.pb
${TF_pretrain_path}/ov/all_tf_models/classification/vgg/19/tf/vgg19.pb
${TF_pretrain_path}/ov/all_tf_models/human_pose_estimation/3d-pose-baseline/tf/3d-pose-baseline.pb
${TF_pretrain_path}/ov/all_tf_models/human_pose_estimation/cpm/person/tf/cpm-person.pb
${TF_pretrain_path}/ov/all_tf_models/human_pose_estimation/openpose/pose/tf/openpose-pose.pb
${TF_pretrain_path}/ov/all_tf_models/instance_segmentation/mask_rcnn/mask_rcnn_inception_v2_coco/tf/mask_rcnn_inception_v2_coco.pb
${TF_pretrain_path}/ov/all_tf_models/IntelLabs/FastImageProcessing/NonlocalDehazing/intel-labs-nonlocal-dehazing.pb
${TF_pretrain_path}/ov/all_tf_models/IntelLabs/LearningToSeeInTheDark/Fuji/learning-to-see-in-the-dark-fuji.pb
${TF_pretrain_path}/ov/all_tf_models/IntelLabs/LearningToSeeInTheDark/Sony/learning-to-see-in-the-dark-sony.pb
${TF_pretrain_path}/ov/all_tf_models/language_representation/bert/base/uncased_L-12_H-768_A-12/tf/bert-base-uncased_L-12_H-768_A-12.pb
${TF_pretrain_path}/ov/all_tf_models/object_attributes/vehicle_attributes/tf/vehicle-attributes-barrier-0103.pb
${TF_pretrain_path}/ov/all_tf_models/object_detection/barrier/tf/0123/vehicle-license-plate-detection-barrier-0123.pb
${TF_pretrain_path}/ov/all_tf_models/object_detection/common/faster_rcnn/faster_rcnn_inception_v2_coco/tf/faster_rcnn_inception_v2_coco.pb
${TF_pretrain_path}/ov/all_tf_models/object_detection/common/faster_rcnn/faster_rcnn_resnet101_coco/tf/faster_rcnn_resnet101_coco.pb
${TF_pretrain_path}/ov/all_tf_models/object_detection/common/faster_rcnn/faster_rcnn_resnet50_coco/tf/faster_rcnn_resnet50_coco.pb
${TF_pretrain_path}/ov/all_tf_models/object_detection/common/faster_rcnn/faster_rcnn_resnet50_lowproposals_coco/tf/faster_rcnn_resnet50_lowproposals_coco.pb
${TF_pretrain_path}/ov/all_tf_models/object_detection/common/retinanet/tf/retinanet.pb
${TF_pretrain_path}/ov/all_tf_models/object_detection/common/rfcn/rfcn_resnet101_coco/tf/rfcn-resnet101-coco.pb
${TF_pretrain_path}/ov/all_tf_models/object_detection/common/ssd_inceptionv2/tf/inceptionv2_ssd.pb
${TF_pretrain_path}/ov/all_tf_models/object_detection/common/ssd_resnet50/ssd_resnet50_v1_fpn_coco/tf/ssd_resnet50_v1_fpn_coco.pb
${TF_pretrain_path}/ov/all_tf_models/object_detection/yolo/yolo_v2/tf/yolo-v2.pb
${TF_pretrain_path}/ov/all_tf_models/object_detection/yolo/yolo_v3/tf/yolo-v3.pb
${TF_pretrain_path}/ov/all_tf_models/PublicCompressed/detection/tinyYOLOv2/fp32_sparsity00/yolo-v2-tiny-ava-0001.pb
${TF_pretrain_path}/ov/all_tf_models/PublicCompressed/detection/tinyYOLOv2/fp32_sparsity30/yolo-v2-tiny-ava-sparse-30-0001.pb
${TF_pretrain_path}/ov/all_tf_models/PublicCompressed/detection/tinyYOLOv2/fp32_sparsity60/yolo-v2-tiny-ava-sparse-60-0001.pb
${TF_pretrain_path}/ov/all_tf_models/PublicCompressed/detection/YOLOv2/fp32_sparsity35/yolo-v2-ava-sparse-35-0001.pb
${TF_pretrain_path}/ov/all_tf_models/PublicCompressed/detection/YOLOv2/fp32_sparsity70/yolo-v2-ava-sparse-70-0001.pb
${TF_pretrain_path}/ov/all_tf_models/PublicCompressed/semantic_segmentation/icnet-camvid-tf-ws00/icnet-camvid-ava-0001.pb
${TF_pretrain_path}/ov/all_tf_models/PublicCompressed/semantic_segmentation/icnet-camvid-tf-ws30/icnet-camvid-ava-sparse-30-0001.pb
${TF_pretrain_path}/ov/all_tf_models/PublicCompressed/semantic_segmentation/icnet-camvid-tf-ws60/icnet-camvid-ava-sparse-60-0001.pb
${TF_pretrain_path}/ov/all_tf_models/PublicInHouse/classification/darknet19/darknet19.pb
${TF_pretrain_path}/ov/all_tf_models/PublicInHouse/classification/darknet53/darknet53.pb
${TF_pretrain_path}/ov/all_tf_models/PublicInHouse/object_detection/common/dssd/DSSD_12/tf/DSSD_12.pb
${TF_pretrain_path}/ov/all_tf_models/PublicInHouse/object_detection/common/yolo/v1_tiny/tf/tiny_yolo_v1.pb
${TF_pretrain_path}/ov/all_tf_models/PublicInHouse/object_detection/common/yolo/v2_tiny/tf/tiny_yolo_v2.pb
${TF_pretrain_path}/ov/all_tf_models/PublicInHouse/sequence_modelling/tcn/tf/TCN.pb
${TF_pretrain_path}/ov/all_tf_models/Retail/action_detection/pedestrian/rmnet_ssd/0028_tf/tf/rmnet_ssd.pb
${TF_pretrain_path}/ov/all_tf_models/Retail/object_attributes/emotions_recognition/0002/tf/icv-emotions-recognition-0002.pb
${TF_pretrain_path}/ov/all_tf_models/Security/feature_extraction/ava/tf/ava-face-recognition-3.0.0.pb
${TF_pretrain_path}/ov/all_tf_models/Security/object_detection/barrier/yolo/yolo-v2-tiny-vehicle-detection-0001/tf/yolo-v2-tiny-vehicle-detection-0001.pb
${TF_pretrain_path}/ov/all_tf_models/Security/object_detection/common/ava/stage2/tf/ava-person-vehicle-detection-stage2-2.0.0.pb
${TF_pretrain_path}/ov/all_tf_models/Security/object_detection/crossroad/1020/tf/person-vehicle-bike-detection-crossroad-yolov3-1020.pb
${TF_pretrain_path}/ov/all_tf_models/semantic_segmentation/deeplab/v3/deeplabv3.pb
${TF_pretrain_path}/ov/all_tf_models/text_detection/ctpn/tf/ctpn.pb
${TF_pretrain_path}/ckpt/faster_rcnn_resnet101_fgvc_2018_07_19/faster_rcnn_resnet101_fgvc_2018_07_19/faster_rcnn_resnet101_fgvc.pb
${TF_pretrain_path}/ckpt/faster_rcnn_resnet101_kitti_2018_01_28/faster_rcnn_resnet101_kitti_2018_01_28/faster_rcnn_resnet101_kitti.pb
${TF_pretrain_path}/ckpt/faster_rcnn_resnet101_lowproposals_coco_2018_01_28/faster_rcnn_resnet101_lowproposals_coco_2018_01_28/faster_rcnn_resnet101_lowproposals_coco.pb
${TF_pretrain_path}/ckpt/faster_rcnn_resnet50_fgvc_2018_07_19/faster_rcnn_resnet50_fgvc_2018_07_19/faster_rcnn_resnet50_fgvc.pb
${TF_pretrain_path}/ckpt/ssd_inception_v2_coco_2018_01_28/ssd_inception_v2_coco_2018_01_28/ssd_inception_v2_coco.pb
${TF_pretrain_path}/ckpt/ssd-resnet34_300x300/ssd_resnet34_300x300.pb
${TF_pretrain_path}/mlp/deeplabv3_mnv2_cityscapes_train/deeplab.pb
${TF_pretrain_path}/ckpt/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/SSD_ResNet50_V1_FPN_640x640_RetinaNet50.pb
${TF_pretrain_path}/dpg/DRAW/DRAW.pb
${TF_pretrain_path}/mlp/vnet/vnet.pb
${TF_pretrain_path}/mlp/GraphSage/GraphSage.pb
${TF_pretrain_path}/mlp/oob_gan_models/WGAN.pb
${TF_pretrain_path}/ov/all_tf_models/classification/efficientnet/b0/tf/efficientnet-b0.pb
${TF_pretrain_path}/ov/all_tf_models/classification/efficientnet/b0_auto_aug/tf/efficientnet-b0_auto_aug.pb
${TF_pretrain_path}/ov/all_tf_models/classification/efficientnet/b5/tf/efficientnet-b5.pb
${TF_pretrain_path}/ov/all_tf_models/classification/efficientnet/b7_auto_aug/tf/efficientnet-b7_auto_aug.pb
${TF_pretrain_path}/ov/all_tf_models/voice_recognition/vggvox/vggvox.pb
${TF_pretrain_path}/ov/all_tf_models/AIPG_trained/text_classification/vdcnn/agnews/tf/aipg-vdcnn.pb
${TF_pretrain_path}/ov/all_tf_models/human_pose_estimation/arttrack/coco/tf/arttrack-coco-multi.pb
${TF_pretrain_path}/ov/all_tf_models/human_pose_estimation/arttrack/mpii/tf/arttrack-mpii-single.pb
${TF_pretrain_path}/ov/all_tf_models/dna_sequencing/deepvariant/wgs/deepvariant_wgs.pb
${TF_pretrain_path}/ov/all_tf_models/text_detection/east/tf/east_resnet_v1_50.pb
${TF_pretrain_path}/ov/all_tf_models/face_recognition/facenet/CASIA-WebFace/tf/facenet-20180408-102900.pb
${TF_pretrain_path}/ov/all_tf_models/Retail/handwritten-score-recognition/0003/tf/handwritten-score-recognition-0003.pb
${TF_pretrain_path}/ov/all_tf_models/optical_character_recognition/license_plate_recognition/tf/license-plate-recognition-barrier-0007.pb
${TF_pretrain_path}/ov/all_tf_models/optical_character_recognition/text_recognition/tf/optical_character_recognition-text_recognition-tf.pb
${TF_pretrain_path}/ov/all_tf_models/face_reconstruction/PRNet/tf/PRNet.pb
${TF_pretrain_path}/ov/all_tf_models/Retail/text_recognition/bilstm_crnn_bilstm_decoder/0012/tf/text-recognition-0012.pb
${TF_pretrain_path}/dpg/Hierarchical/Hierarchical_LSTM.pb
${TF_pretrain_path}/ov/all_tf_models/object_detection/yolo/yolo_v3/yolo-v3-tiny/yolo-v3-tiny-tf/yolo-v3-tiny.pb
${TF_pretrain_path}/dpg/Resnet_v2_200/resnet_v2_200.pb
# Q4-2020
${TF_pretrain_path}/mlp/CBAM/CBAM.pb
${TF_pretrain_path}/dpg/ALBERT/ALBERT.pb
${TF_pretrain_path}/mlp/BERT_BASE/BERT_BASE.pb
${TF_pretrain_path}/mlp/BERT_LARGE/BERT_LARGE.pb
${TF_pretrain_path}/dpg/CharCNN/CharCNN.pb
${TF_pretrain_path}/ov/Dilation/dilation.pb
${TF_pretrain_path}/ov/all_tf_models/object_detection/common/faster_rcnn/faster_rcnn_inception_resnet_v2_atrous_coco/tf/faster_rcnn_inception_resnet_v2_atrous_coco.pb
${TF_pretrain_path}/ckpt/faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28/faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco.pb
${TF_pretrain_path}/ckpt/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_coco_2018_01_28.pb
${TF_pretrain_path}/ov/all_tf_models/object_detection/common/faster_rcnn/faster_rcnn_nas_lowproposals_coco/tf/faster_rcnn_nas_lowproposals_coco.pb
${TF_pretrain_path}/mlp/oob_gan_models/GAN.pb
${TF_pretrain_path}/ov/all_tf_models/image_inpainting/gmcnn/tf/gmcnn-places2.pb
${TF_pretrain_path}/mlp/oob_gan_models/LSGAN.pb
${TF_pretrain_path}/ov/all_tf_models/instance_segmentation/mask_rcnn/mask_rcnn_inception_resnet_v2_atrous_coco/tf/mask_rcnn_inception_resnet_v2_atrous_coco.pb
${TF_pretrain_path}/dpg/ncf_trained_movielens_1m/NCF-1B.pb
${TF_pretrain_path}/dpg/R-FCN/rfcn_resnet101_coco_2018_01_28/R-FCN.pb
${TF_pretrain_path}/mlp/transformer_lt_official_fp32_pretrained_model/graph/Transformer-LT.pb
${TF_pretrain_path}/oob/checkpoint_dynamic_memory_network/DynamicMemory.pb
${TF_pretrain_path}/oob/checkpoint_entity_network2/EntityNet.pb
${TF_pretrain_path}/oob/Seq2seqAttn/Seq2seqAttn.pb
${TF_pretrain_path}/mlp/SqueezeNet-tf/SqueezeNet.pb
${TF_pretrain_path}/mlp/ssd_resnet34_model/ssd_resnet34_1200x1200.pb
${TF_pretrain_path}/oob/TextRCNN/TextRCNN.pb
${TF_pretrain_path}/mlp/unet/U-Net.pb
${TF_pretrain_path}/dpg/wavenet/wavenet.pb
${TF_pretrain_path}/dpg/SphereFace/SphereFace.pb
${TF_pretrain_path}/oob/checkpoint_hier_atten_title/text_hier_atten_title_desc_checkpoint_MHA/HierAtteNet.pb
${TF_pretrain_path}/mlp/ResNet50_v1_5/model_dir/ResNet-50_v1.5.pb
${TF_pretrain_path}/dpg/ResNext_50/ResNext_50.pb
${TF_pretrain_path}/dpg/ResNext_101/ResNext_101.pb
${TF_pretrain_path}/oob/COVID-Net/COVID-Net.pb
${TF_pretrain_path}/mlp/CRNN/crnn.pb
${TF_pretrain_path}/mlp/CenterNet/CenterNet.pb
${TF_pretrain_path}/dpg/CapsuleNet/CapsuleNet.pb
${TF_pretrain_path}/ov/all_tf_models/speech_to_text/deepspeech/v1/tf/deepspeech.pb
${TF_pretrain_path}/oob/DIEN/DIEN.pb
${TF_pretrain_path}/ov/all_tf_models/semantic_segmentation/dense_vnet/tf/dense_vnet_abdominal_ct.pb
${TF_pretrain_path}/oob/TextCNN/TextCNN.pb
${TF_pretrain_path}/oob/TextRNN/TextRNN.pb
${TF_pretrain_path}/oob/show_and_tell/show_and_tell.pb
${TF_pretrain_path}/dpg/MiniGo/MiniGo.pb
## Q1-21
${TF_pretrain_path}/oob/oob_gan_models/ACGAN.pb
${TF_pretrain_path}/oob/oob_gan_models/BEGAN.pb
${TF_pretrain_path}/oob/oob_gan_models/CGAN.pb
${TF_pretrain_path}/oob/oob_gan_models/DRAGAN.pb
${TF_pretrain_path}/oob/oob_gan_models/EBGAN.pb
${TF_pretrain_path}/oob/oob_gan_models/infoGAN.pb
${TF_pretrain_path}/ov/all_tf_models/instance_segmentation/mask_rcnn/mask_rcnn_resnet101_atrous_coco/tf/mask_rcnn_resnet101_atrous_coco.pb
${TF_pretrain_path}/ov/all_tf_models/instance_segmentation/mask_rcnn/mask_rcnn_resnet50_atrous_coco/tf/mask_rcnn_resnet50_atrous_coco.pb
${TF_pretrain_path}/ov/all_tf_models/Security/object_detection/crossroad/1024/tf/person-vehicle-bike-detection-crossroad-yolov3-1024.pb
${TF_pretrain_path}/ov/all_tf_models/human_pose_estimation/pose-ae/multiperson/tf/pose-ae-multiperson.pb
${TF_pretrain_path}/ov/all_tf_models/human_pose_estimation/pose-ae/refinement/tf/pose-ae-refinement.pb
${TF_pretrain_path}/ov/all_tf_models/image_processing/srgan/tf/srgan.pb
${TF_pretrain_path}/dpg/SSD-ResNet34_1200x1200/ssd_resnet34_fp32_1200x1200_pretrained_model.pb
${TF_pretrain_path}/ov/all_tf_models/PublicInHouse/volumetric_segmentation/unet/3d/isensee_2017/tf/unet-3d-isensee_2017.pb
${TF_pretrain_path}/ov/all_tf_models/PublicInHouse/volumetric_segmentation/unet/3d/origin/tf/unet-3d-origin.pb
${TF_pretrain_path}/oob/oob_gan_models/WGAN_GP.pb
${TF_pretrain_path}/dpg/wide_deep/wide_deep.pb
${TF_pretrain_path}/dpg/EfficientDet/efficientdet-d0/EfficientDet-D0-512x512.pb
${TF_pretrain_path}/dpg/EfficientDet/efficientdet-d1/EfficientDet-D1-640x640.pb
${TF_pretrain_path}/dpg/EfficientDet/efficientdet-d2/EfficientDet-D2-768x768.pb
${TF_pretrain_path}/dpg/EfficientDet/efficientdet-d3/EfficientDet-D3-896x896.pb
${TF_pretrain_path}/dpg/EfficientDet/efficientdet-d4/EfficientDet-D4-1024x1024.pb
${TF_pretrain_path}/dpg/EfficientDet/efficientdet-d5/EfficientDet-D5-1280x1280.pb
${TF_pretrain_path}/dpg/EfficientDet/efficientdet-d6/EfficientDet-D6-1280x1280.pb
${TF_pretrain_path}/dpg/EfficientDet/efficientdet-d7/EfficientDet-D7-1536x1536.pb
${TF_pretrain_path}/ov/vehicle-license-plate-detection-barrier-0106/vehicle-license-plate-detection-barrier-0106.pb
${TF_pretrain_path}/ckpt/ssd_resnet101_v1_fpn/ssd_resnet_101_fpn_oidv4.pb
${TF_pretrain_path}/dpg/key-value-memory-networks/key-value-memory-networks.pb
${TF_pretrain_path}/dpg/NTM-One-Shot/model/NTM-One-Shot.pb
${TF_pretrain_path}/dpg/NetVLAD/NetVLAD.pb
${TF_pretrain_path}/mlp/NeuMF/NeuMF.pb
${TF_pretrain_path}/ckpt/context_rcnn_resnet101_snapshot_serengeti_2020_06_10/context_rcnn_resnet101_snapshot_serenget.pb
${TF_pretrain_path}/dpg/MANN/MANN.pb
${TF_pretrain_path}/mlp/dcgan/DCGAN.pb
${TF_pretrain_path}/dpg/simple_net/Evolution_ensemble.pb
${TF_pretrain_path}/mlp/keypoint/KeypointNet.pb
${TF_pretrain_path}/mlp/yolov4/YOLOv4.pb
${TF_pretrain_path}/mlp/3D-Unet/3DUNet.pb
${TF_pretrain_path}/dpg/adv_inception_v3/adv_inception_v3.pb
${TF_pretrain_path}/dpg/ens3_inception_v3/ens3_adv_inception_v3.pb
${TF_pretrain_path}/mlp/NCF/NCF.pb
${TF_pretrain_path}/mlp/faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid/faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid.pb
${TF_pretrain_path}/mlp/faster_rcnn_inception_resnet_v2_atrous_oid/faster_rcnn_inception_resnet_v2_atrous_oid.pb
${TF_pretrain_path}/mlp/faster_rcnn_resnet101_snapshot_serengeti/faster_rcnn_resnet101_snapshot_serengeti.pb
${TF_pretrain_path}/mlp/ssd_mobilenet_v1_0.75_depth_300x300_coco/ssd_mobilenet_v1_0.75_depth_300x300_coco.pb
${TF_pretrain_path}/mlp/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco.pb
)

ln -s ${TF_pretrain_path}/mlp/keypoint/cars_with_keypoints

for checkpoint in ${model_path[@]}
do
    ./launch_benchmark.sh --checkpoint=${checkpoint} --use_TF_NativeFormat=${use_TF_NativeFormat} --run_perf=${run_perf} --collect_dnnl_verbose=${collect_dnnl_verbose} --precision=${precision} --workspace=${workspace}
done

script_path=(
"${CUR_PATH}/workload/CBAM/"
"${CUR_PATH}/workload/DETR/"
"${CUR_PATH}/workload/Unet/"
"${CUR_PATH}/workload/time_series_LSTM/"
"${CUR_PATH}/workload/centernet_hg104/"
"${CUR_PATH}/workload/WD/"
)
for path in ${script_path[@]}
do
    cd ${path} 
    source ./auto_benchmark.sh --workspace=${workspace} \
                            --precision=${precision} \
                            --run_perf=${run_perf} \
                            --collect_dnnl_verbose=${collect_dnnl_verbose} \
                            --oob_home_path=${oob_home_path} \
                            --tf_pretrain_path=${TF_pretrain_path} \

done

source ${DE_CONDA_PATH}

## tensorflow-1.15.2
source ${CONDA_PATH} oob_tf_1.15
script_path=(
"${CUR_PATH}/workload/elmo/"
)
for path in ${script_path[@]}
do
    cd ${path} 
    source ./auto_benchmark.sh --workspace=${workspace} \
                            --precision=${precision} \
                            --run_perf=${run_perf} \
                            --collect_dnnl_verbose=${collect_dnnl_verbose} \
                            --oob_home_path=${oob_home_path} \
                            --tf_pretrain_path=${TF_pretrain_path} \

done
source ${DE_CONDA_PATH}
