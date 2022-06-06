#!/bin/bash
set -xe
# import common funcs
source ../common.sh

# huggingface_models
function main {
    # prepare workload
    workload_dir=${PWD}
    source_code_dir="../../transformers"
    source_code_name="transformers"
    init_model ${source_code_dir} ${source_code_name}
    
    init_params $@
    # setup
    pip install --no-deps Pillow
    pip install tensorboard
    pip install tensorboardX
    pip install xlsxwriter
    if [ "${bf16_train_cuda}" == "1" ];then
        pip uninstall -y torchvision torch
        pip3 install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html -U
        pip install -r ./examples/requirements.txt
    fi
    python setup.py install
    cp ${workload_dir}/download_glue_dataset.py .
    cp ${workload_dir}/run_glue.py examples
    cp ${workload_dir}/run_glue_32.py examples
    cp ${workload_dir}/run_glue_cuda.py examples
    python download_glue_dataset.py --data_dir glue_data --tasks MRPC

    # set common info
    set_environment
    fetch_cpu_info
    logs_path_clean

    #
    if [ "${model_name}" == 'huggingface_models' ];then
        model_name="albert,bart,roberta,xlm-roberta,xlnet"
    fi
    model_name="albert,roberta,xlm-roberta,xlnet"
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
        echo ${model_name}
        #
        if [ "${bf16_train_cpu}" == "1" ];then
            echo "bf16_train_cpu=1, start to train bf16 on cpu"
            python examples/run_glue.py  --model_type ${model_name} \
            --model_name_or_path ${model_name_or_path} \
            --task_name mrpc \
            --do_train \
            --do_eval \
            --data_dir ./glue_data/MRPC/ \
            --max_seq_length 128 \
            --per_gpu_train_batch_size 32 \
            --learning_rate 2e-5 \
            --num_train_epochs 1 \
            --overwrite_output_dir \
            --output_dir ${model_name_or_path} \ || true
        elif [ "${bf16_train_cuda}" == "1" ];then
            echo "bf16_train_cuda=1, start to train bf16 on cuda"
            python examples/run_glue_cuda.py  --model_type ${model_name} \
            --model_name_or_path ${model_name_or_path} \
            --task_name mrpc \
            --do_train \
            --do_eval \
            --data_dir ./glue_data/MRPC/ \
            --max_seq_length 128 \
            --per_gpu_train_batch_size 32 \
            --learning_rate 2e-5 \
            --num_train_epochs 1 \
            --overwrite_output_dir \
            --output_dir ${model_name_or_path} \ || true
        else
            echo "bf16_train_cpu/cuda=0, start to train fp32"
            python examples/run_glue_32.py  --model_type ${model_name} \
            --model_name_or_path ${model_name_or_path} \
            --task_name mrpc \
            --do_train \
            --do_eval \
            --data_dir ./glue_data/MRPC/ \
            --max_seq_length 128 \
            --per_gpu_train_batch_size 32 \
            --learning_rate 2e-5 \
            --num_train_epochs 1 \
            --overwrite_output_dir \
            --output_dir ${model_name_or_path} \ || true
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
