#!/bin/bash
set -xe
# import common funcs
source ../common.sh

# ssd300
function main {
    # prepare workload
    workload_dir=${PWD}
    source_code_dir="../../ssd300"
    source_code_name="ssd300"
    init_model ${source_code_dir} ${source_code_name}

    # setup
    pip install mlperf_compliance
    cd single_stage_detector/ssd
    git apply ${workload_dir}/ssd300-training.patch

    # set common info
    init_params $@
    set_environment
    fetch_cpu_info
    logs_path_clean
    
    model_name="ssd300"
    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        #
        if [ "${bf16_train_cpu}" == "1" ];then
            echo "bf16_train_cpu=1, start to train bf16 on cpu"
            python train.py --bf16-train-cpu || true
        elif [ "${bf16_train_cuda}" == "1" ];then
            echo "bf16_train_cuda=1, start to train bf16 on cuda"
            python train.py --bf16-train-cuda || true
        else
            echo "bf16_train_cpu/cuda=0, start to train fp32"
            python train.py || true
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
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${cpu_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        printf "numactl --localalloc --physcpubind ${cpu_array[i]} timeout 7200 \
            python generate.py ./dataset/wmt14_en_de/ --path ./models/checkpoint_best.pt \
                --arch 'fconv' --beam 5 --remove-bpe \
                -i ${num_iter} --max-sentences ${batch_size} \
                --channels_last ${channels_last} \
                --precision ${precision} \
                ${addtion_options} \
        > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
    done
    echo -e "\n wait" >> ${excute_cmd_file}
    echo -e "\n\n\n\n Running..."
    source ${excute_cmd_file}
    echo -e "Finished.\n\n\n\n"

    # latency and throughput
    # latency=$(grep 'Throughput:' ${log_dir}/rcpi* |sed -e 's/.*Throughput//;s/,.*//;s/[^0-9.]//g' |awk -v bs=${batch_size} '
    #     BEGIN {
    #         sum = 0;
    #         i = 0;
    #     }
    #     {
    #         sum = sum + bs / $1 * 1000;
    #         i++;
    #     }
    #     END {
    #         sum = sum / i;
    #         printf("%.3f", sum);
    #     }
    # ')
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
