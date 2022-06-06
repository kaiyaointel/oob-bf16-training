#!/bin/bash
set -xe
# import common funcs
source ../common.sh

# 3d-unet
function main {
    # prepare workload
    workload_dir=${PWD}
    source_code_dir="../../3d-unet"
    source_code_name="3d-unet"
    init_model ${source_code_dir} ${source_code_name}

    # setup
    pip install --no-deps torchvision
    pip install hdbscan
    pip install scikit-image
    pip install h5py
    pip install --no-deps Pillow
    pip install pyyaml
    pip install tensorboardX
    pip install tensorboard
    git apply ${workload_dir}/unet3d-training.patch
    python setup.py install
    cd resources && cp random3D.h5 random3D_copy.h5
    cd ../pytorch3dunet

    # set common info
    init_params $@
    set_environment
    fetch_cpu_info
    logs_path_clean

    model_name="3d-unet"
    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        #
        if [ "${bf16_train_cpu}" == "1" ];then
            echo "bf16_train_cpu=1, start to train bf16 on cpu"
            python train.py --config ../resources/train_config_ce.yaml --bf16-train-cpu || true
        elif [ "${bf16_train_cuda}" == "1" ];then
            echo "bf16_train_cuda=1, start to train bf16 on cuda"
            python train.py --config ../resources/train_config_ce.yaml --bf16-train-cuda || true
        else
            echo "bf16_train_cpu/cuda=0, start to train fp32"
            python train.py --config ../resources/train_config_ce.yaml || true
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
