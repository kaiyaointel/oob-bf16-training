#!/bin/bash
set -xe
# import common funcs
source ../common.sh

# GNMT
function main {
    # prepare workload
    workload_dir="${PWD}"
    source_code_dir="../../maskrcnn/"
    init_model ${source_code_dir}

    patch -p1 < ${workload_dir}/gnmt.patch || true
    pip install -r ${workload_dir}/requirements.txt
    # pip install --no-deps torchvision -f https://download.pytorch.org/whl/torch_stable.html
    cd rnn_translator/pytorch/
    if [ ! -e dataset ];then
        ln -sf /home2/pytorch-broad-models/GNMT/dataset/ .
    fi

    # set common info
    init_params $@
    set_environment
    fetch_cpu_info
    logs_path_clean

    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        # cache weight
        python train.py --inference --dataset-dir ./dataset/data/ --val-batch-size 1 --test-batch-size 1 \
            --val-num-iters 3 --val-num-warmup 1  --print-freq 1 \
            --precision ${precision} --channels-last ${channels_last}
        #
        for batch_size in ${batch_size_list[@]}
        do
            generate_core
            collect_perf_logs
        done
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
            python train.py --inference --print-freq 10 --dataset-dir ./dataset/data/ \
                --val-batch-size ${batch_size} --test-batch-size ${batch_size} \
                --val-num-iters ${num_iter} --val-num-warmup ${num_warmup} \
                --precision ${precision} \
                --channels-last ${channels_last} \
                ${addtion_options} \
        > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
    done
    echo -e "\n wait" >> ${excute_cmd_file}
    echo -e "\n\n\n\n Running..."
    source ${excute_cmd_file}
    echo -e "Finished.\n\n\n\n"

    # latency and throughput
    # latency=$(grep 'Inference:' ${log_dir}/rcpi* |sed -e 's/.*Inference//;s/,.*//;s/[^0-9.]//g' |awk -v bs=${batch_size} '
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
    throughput=$(grep 'Inference:' ${log_dir}/rcpi* |sed -e 's/.*Inference//;s/,.*//;s/[^0-9.]//g' |awk '
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
