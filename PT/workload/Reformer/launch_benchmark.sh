#!/bin/bash
set -xe
# import common funcs
source ../common.sh

# Reformer
function main {
    # prepare workload
    workload_dir=${PWD}
    source_code_dir="../../Reformer"
    source_code_name="Reformer"
    init_model ${source_code_dir} ${source_code_name}

    init_params $@
    # setup
    if [ "${bf16_train_cuda}" == "1" ];then
        echo "apply bf16 patch for cuda..."
        git apply ${workload_dir}/Reformer-training-bf16-cuda.patch
    elif [ "${bf16_train_cpu}" == "1" ];then
        echo "apply bf16 patch for cpu..."
        git apply ${workload_dir}/Reformer-training-bf16-cpu.patch
    else
        echo "for Reformer fp32 training no patch is needed, pass this step"
    fi
    
    python setup.py install
    
    cd examples/enwik8_simple
    
    cp ${workload_dir}/train-bf16-cpu.py .
    cp ${workload_dir}/train-bf16-cuda.py .
    cp ${workload_dir}/train-fp32-cpu.py .
    
    
    # set common info
    set_environment
    fetch_cpu_info
    logs_path_clean

    model_name="Reformer"
    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        #
        if [ "${bf16_train_cpu}" == "1" ];then
            echo "bf16_train_cpu=1, start to train bf16 on cpu"
            python train-bf16-cpu.py || true
        elif [ "${bf16_train_cuda}" == "1" ];then
            echo "bf16_train_cuda=1, start to train bf16 on cuda"
            python train-bf16-cuda.py || true
        else
            echo "bf16_train_cpu/cuda=0, start to train fp32"
            python train-fp32-cpu.py || true
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
