#!/bin/bash
set -xe
# import common funcs
source ../common.sh

# conv-seq-seq
function main {
    # prepare workload
    workload_dir=${PWD}
    source_code_dir="../../conv-seq-seq"
    source_code_name="conv-seq-seq"
    init_model ${source_code_dir} ${source_code_name}

    pip install --editable ./
    
    # init put ahead for bf16 distinguish
    init_params $@
    
    if [ "${bf16_train_cpu}" == "1" ];then
        git apply ${workload_dir}/conv-seq-seq-training-new.patch # for early stopping of train loop, added count
    elif [ "${bf16_train_cuda}" == "1" ];then
        git apply ${workload_dir}/conv-seq-seq-training-cuda.patch
    else
        git apply ${workload_dir}/conv-seq-seq-training-fp32.patch
    fi
    
    # set common info
    set_environment
    fetch_cpu_info
    logs_path_clean
    
    model_name="conv-seq-seq"
    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        #
        fairseq-train /home2/pytorch-broad-models/Convseq2seq/dataset/wmt14_en_de --batch-size 32 --arch fconv_wmt_en_de --cpu --lr 0.01 --max-epoch 1 || true
        #
        #for batch_size in ${batch_size_list[@]}
        #do
        #    generate_core
        #    collect_perf_logs
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
