#!/bin/bash
set -xe
# import common funcs
source ../common.sh

# huggingface_models
function main {
    # prepare workload
    workload_dir="${PWD}"
    source_code_dir="../../transformers/"
    init_model ${source_code_dir}
    #
    patch -p1 < ${workload_dir}/huggingface_models.patch || true
    pip install -r ${workload_dir}/requirements.txt
    # pip install --no-deps torchvision -f https://download.pytorch.org/whl/torch_stable.html
    python setup.py install
    if [ ! -d './glue_data/MRPC' ];then
        python ./utils/download_glue_data.py --tasks MRPC
    fi

    # set common info
    init_params $@
    set_environment
    fetch_cpu_info
    logs_path_clean

    #
    if [ "${model_name}" == 'huggingface_models' ];then
        model_name="albert,bart,roberta,xlm-roberta,xlnet"
    fi
    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        #
        if [ ${model_name} == "xlnet" ];then
            model_name_or_path='xlnet-base-cased'
        elif [ ${model_name} == "albert" ];then
            model_name_or_path='albert-base-v2'
        elif [ ${model_name} == "roberta" ];then
            model_name_or_path='roberta-base'
        elif [ ${model_name} == "distilbert" ];then
            model_name_or_path='distilbert-base-uncased'
        elif [ ${model_name} == "xlm-roberta" ];then
            model_name_or_path='xlm-roberta-base'
        elif [ ${model_name} == "bart" ];then
            model_name_or_path='bart-large'
        fi
        # cache weight
        python examples/run_glue.py --no-cuda --do_eval \
            --model_type ${model_name} --model_name_or_path ${model_name_or_path} \
            --task_name mrpc --output_dir ${model_name_or_path} \
            --data_dir ./glue_data/MRPC/ \
            --num_warmup_iters 1 --num_iters 2 \
            --per_gpu_eval_batch_size 1 \
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
            python examples/run_glue.py --no-cuda --do_eval \
                --model_type ${model_name} --model_name_or_path ${model_name_or_path} \
                --task_name mrpc --output_dir ${model_name_or_path} \
                --data_dir ./glue_data/MRPC/ \
                --num_warmup_iters ${num_warmup} --num_iters ${num_iter} \
                --per_gpu_eval_batch_size ${batch_size} \
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
