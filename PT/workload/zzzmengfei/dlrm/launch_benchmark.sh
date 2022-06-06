#!/bin/bash
set -xe
# import common funcs
source ../common.sh

# dlrm
function main {
    # prepare workload
    workload_dir="${PWD}"
    source_code_dir="../../dlrm/"
    init_model ${source_code_dir}

    patch -p1 < ${workload_dir}/dlrm.patch || true
    pip install -r ${workload_dir}/requirements.txt
    # pip install --no-deps torchvision -f https://download.pytorch.org/whl/torch_stable.html
    pip install --no-deps onnx
    pip install git+https://github.com/mlperf/logging.git
    if [ ! -e dlrm_kaggle ];then
        ln -sf /home2/pytorch-broad-models/dlrm_kaggle/ .
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
            python dlrm_s_pytorch.py --arch DLRM --inference-only --print-time --test-num-workers=${real_cores_per_instance} \
                --arch-sparse-feature-size=16 --arch-mlp-bot='13-512-256-64-16' --arch-mlp-top='512-256-1' \
                --data-generation=dataset --data-set=kaggle --raw-data-file=dlrm_kaggle/train.txt \
                --processed-data-file=dlrm_kaggle/kaggleAdDisplayChallenge_processed.npz \
                --loss-function=bce --round-targets=True --learning-rate=0.1 --print-freq=1024 --test-mini-batch-size=16384 \
                 --mini-batch-size=${batch_size} \
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
    # latency=$(grep 'Throughput' ${log_dir}/rcpi* |sed -e 's/.*Throughput//;s/,.*//;s/[^0-9.]//g' |awk -v bs=${batch_size} '
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
    throughput=$(grep 'Throughput' ${log_dir}/rcpi* |sed -e 's/.*Throughput//;s/,.*//;s/[^0-9.]//g' |awk '
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
