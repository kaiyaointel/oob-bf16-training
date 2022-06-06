#!/bin/bash
set -xe
# import common funcs
source ../common.sh

# gpt-2
function main {
    # prepare workload
    workload_dir=${PWD}
    source_code_dir="../../gpt-2"
    source_code_name="gpt-2"
    init_model ${source_code_dir} ${source_code_name}

    init_params $@
    # setup
    if [ "${bf16_train_cpu}" == "1" ];then
        git apply ../workload/gpt-2/gpt-2-training.patch
    elif [ "${bf16_train_cuda}" == "1" ];then
        git apply ../workload/gpt-2/gpt-2-training-cuda.patch
    else
        git apply ../workload/gpt-2/gpt-2-training.patch
    fi
    pip install .
    pip install -r ./examples/requirements.txt
    cd examples && mkdir data
    cd data && mkdir wikitext-2-raw
    cd ../..
    cp ../workload/gpt-2/wiki.train.raw ./examples/data/wikitext-2-raw
    cp ../workload/gpt-2/wiki.test.raw ./examples/data/wikitext-2-raw
    export TRAIN_FILE=./data/wikitext-2-raw/wiki.train.raw
    export TEST_FILE=./data/wikitext-2-raw/wiki.test.raw
    cd examples
    
    # set common info
    set_environment
    fetch_cpu_info
    logs_path_clean

    model_name="gpt-2"
    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))
    
    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        #
        if [ "${bf16_train_cpu}" == "1" ];then
            echo "bf16_train_cpu=1, start to train bf16 on cpu"
            python run_language_modeling.py \
                    --output_dir=output \
                    --model_type=gpt2 \
                    --model_name_or_path=gpt2 \
                    --do_train \
                    --train_data_file=$TRAIN_FILE \
                    --do_eval \
                    --eval_data_file=$TEST_FILE \
                    --overwrite_output_dir \
                    --bf16-train-cpu || true
        elif [ "${bf16_train_cuda}" == "1" ];then
            echo "bf16_train_cuda=1, start to train bf16 on cuda"
            python run_language_modeling.py \
                    --output_dir=output \
                    --model_type=gpt2 \
                    --model_name_or_path=gpt2 \
                    --do_train \
                    --train_data_file=$TRAIN_FILE \
                    --do_eval \
                    --eval_data_file=$TEST_FILE \
                    --overwrite_output_dir \
                    --bf16-train-cuda || true
        else
            echo "bf16_train_cpu/cuda=0, start to train fp32"
            python run_language_modeling.py \
                    --output_dir=output \
                    --model_type=gpt2 \
                    --model_name_or_path=gpt2 \
                    --do_train \
                    --train_data_file=$TRAIN_FILE \
                    --do_eval \
                    --eval_data_file=$TEST_FILE \
                    --overwrite_output_dir || true
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
            python predict.py --config ../resources/test_config_dice.yaml \
                --num_iter ${num_iter} --num_warmup ${num_warmup} \
                --batch_size ${batch_size} \
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
