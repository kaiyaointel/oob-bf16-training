#!/bin/bash
set -xe
# import common funcs
source ../common.sh

# CRNN
function main {
    # prepare workload
    workload_dir="${PWD}"
    source_code_dir="../../conv-seq-seq/"
    init_model ${source_code_dir}

    patch -p1 < ${workload_dir}/convseq2seq.patch || true
    pip install -r ${workload_dir}/requirements.txt
    pip install --no-deps -e .
    if [ ! -e dataset ];then
        ln -sf /home2/pytorch-broad-models/Convseq2seq/dataset/ .
    fi
    if [ ! -e ./models/checkpoint_best.pt ];then
        mkdir -p ./models
        ln -sf /home2/pytorch-broad-models/Convseq2seq/models/checkpoint_best.pt ./models/checkpoint_best.pt
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
        python generate.py ./dataset/wmt14_en_de/ --path ./models/checkpoint_best.pt \
            --arch 'fconv' --beam 5 --remove-bpe -i 10 --max-sentences 1 \
            --channels_last ${channels_last} --precision ${precision} || true
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
