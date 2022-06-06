#!/bin/bash
set -xe
# import common funcs
source ../common.sh

# SASRec.pytorch
function main {
    # prepare workload
    workload_dir=${PWD}
    source_code_dir="../../SASRec.pytorch"
    source_code_name="SASRec.pytorch"
    init_model ${source_code_dir} ${source_code_name}

    # setup
    pip install tqdm
    git apply ${workload_dir}/SASRec-training.patch

    # set common info
    init_params $@
    set_environment
    fetch_cpu_info
    logs_path_clean

    model_name="SASRec.pytorch"
    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        #
        if [ "${bf16_train_cpu}" == "1" ];then
            echo "bf16_train_cpu=1, start to train bf16 on cpu"
            python main.py --device=cpu --dataset=ml-1m --train_dir=default --bf16-train-cpu || true
        elif [ "${bf16_train_cuda}" == "1" ];then
            echo "bf16_train_cuda=1, start to train bf16 on cuda"
            python main.py --device=cuda --dataset=ml-1m --train_dir=default --bf16-train-cuda || true
        else
            echo "bf16_train_cpu/cuda=0, start to train fp32"
            python main.py --device=cpu --dataset=ml-1m --train_dir=default || true
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
