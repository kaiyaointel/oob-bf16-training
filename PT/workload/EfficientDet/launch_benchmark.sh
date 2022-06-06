#!/bin/bash
set -xe
# import common funcs
source ../common.sh

# EfficientDet
function main {
    # prepare workload
    workload_dir=${PWD}
    source_code_dir="../../EfficientDet"
    source_code_name="EfficientDet"
    init_model ${source_code_dir} ${source_code_name}

    # setup
    pip install pandas
    pip install albumentations
    pip install pycocotools
    
    if [ "${bf16_train_cuda}" == "1" ];then
        git apply ${workload_dir}/effdet-bf16-cuda.patch
    else
        git apply ${workload_dir}/effdet-bf16-cpu.patch
    fi
    
    cp ${workload_dir}/train-bf16-cpu.py .
    cp ${workload_dir}/train-bf16-cuda.py .
    cp ${workload_dir}/train-fp32-cpu.py .

    # set common info
    init_params $@
    set_environment
    fetch_cpu_info
    logs_path_clean

    model_name="EfficientDet"
    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        #
        if [ "${bf16_train_cpu}" == "1" ];then
            echo "bf16_train_cpu=1, start to train bf16 on cpu"
            python train-bf16-cpu.py --dataset VOC --dataset_root /home2/pytorch-broad-models/ --network efficientdet-d0 --batch_size 32 || true
        elif [ "${bf16_train_cuda}" == "1" ];then
            echo "bf16_train_cuda=1, start to train bf16 on cuda"
            python train-bf16-cuda.py --gpu 0 --dataset VOC --dataset_root /home2/pytorch-broad-models/ --network efficientdet-d0 --batch_size 32 || true
        else
            echo "bf16_train_cpu/cuda=0, start to train fp32"
            python train-fp32-cpu.py --dataset VOC --dataset_root /home2/pytorch-broad-models/ --network efficientdet-d0 --batch_size 32 || true
        fi
        #
        #for batch_size in ${batch_size_list[@]}
        #do
            #generate_core
            #collect_perf_logs
        #done
    done
}

# run
function generate_core {
    log_file="${log_dir}/${model_name}.log"

    printf "numactl --localalloc --physcpubind ${cpu_array[i]} timeout 7200 \
        python train.py --config ../resources/train_config_ce.yaml
    > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
    
    echo -e "\n wait" >> ${excute_cmd_file}
    echo -e "\n\n\n\n Running..."
    source ${excute_cmd_file}
    echo -e "Finished.\n\n\n\n"

    throughput=$(grep 'throughput:' ${log_dir}/rcpi* |sed -e 's/.*throughput//;s/,.*//;s/[^0-9.]//g' |awk '
        BEGIN {
            sum = 0;
        }
        {
            sum = sum + $1;
        }
        END {
            printf("%.2f", sum);
        }
    ')
}

# Start
main "$@"
