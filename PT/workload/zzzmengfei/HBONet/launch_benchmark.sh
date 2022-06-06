#!/bin/bash
set -xe
# import common funcs
source ../common.sh

# HBONet
function main {
    # prepare workload
    workload_dir="${PWD}"
    source_code_dir="../../HBONet/"
    init_model ${source_code_dir}

    patch -p1 < ${workload_dir}/HBONet.patch || true
    pip install -r ${workload_dir}/requirements.txt
    # pip install --no-deps torchvision -f https://download.pytorch.org/whl/torch_stable.html

    # set common info
    init_params $@
    set_environment
    fetch_cpu_info
    logs_path_clean

    # all HBONet models
    if [ "${model_name}" == "HBONet" ];then
        model_name="hbonet-0.25-224x224,hbonet-0.5-224x224,hbonet-1.0-224x224"
    fi
    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        # cache weight
        python imagenet.py --dummy -a hbonet -e --HBONet_name ${model_name} --warmup_iters 1 \
            --width-mult $(echo ${model_name} |cut -d '-' -f 2) --max_iters 2 -b 1 \
            --precision ${precision} --channels_last ${channels_last} --config_file ${workload_dir}/conf.yaml
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
            python imagenet.py --dummy -a hbonet -e \
                --HBONet_name ${model_name} --width-mult $(echo ${model_name} |cut -d '-' -f 2) \
                --warmup_iters ${num_warmup} --max_iters ${num_iter} \
                --config_file ${workload_dir}/conf.yaml \
                -b ${batch_size} \
                --precision ${precision} \
                --channels_last ${channels_last} \
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
    throughput=$(grep 'Throughput:' ${log_dir}/rcpi* |sed -e 's/.*Throughput//;s/,.*//;s/[^0-9.]//g' |awk '
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
