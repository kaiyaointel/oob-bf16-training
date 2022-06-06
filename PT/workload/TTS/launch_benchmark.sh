#!/bin/bash
set -xe
# import common funcs
source ../common.sh

# TTS
function main {
    # prepare workload
    workload_dir=${PWD}
    source_code_dir="../../TTS"
    source_code_name="TTS"
    init_model ${source_code_dir} ${source_code_name}

    # setup
    pip install numba==0.48
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libsndfile1-dev # no need to enter time-zone
    cp ../workload/TTS/TTS-setup.patch .
    git apply TTS-setup.patch
    python setup.py develop
    pip install -r requirements.txt

    cp -r /home2/pytorch-broad-models/TTS/datasets/ljspeech LJSpeech-1.1
    shuf LJSpeech-1.1/metadata_original.csv > LJSpeech-1.1/metadata_shuf.csv
    head -n 12000 LJSpeech-1.1/metadata_shuf.csv > LJSpeech-1.1/metadata_train.csv
    tail -n 1100 LJSpeech-1.1/metadata_shuf.csv > LJSpeech-1.1/metadata_val.csv

    cp ../workload/TTS/train_bf16.py .
    cp ../workload/TTS/config_bf16.json .

    # set common info
    init_params $@
    set_environment
    fetch_cpu_info
    logs_path_clean

    model_name="TTS"
    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        #
        if [ "${bf16_train_cpu}" == "1" ];then
            echo "bf16_train_cpu=1, start to train bf16 on cpu"
            python train_bf16.py --config_path config_bf16.json --bf16-train-cpu || true
        elif [ "${bf16_train_cuda}" == "1" ];then
            echo "bf16_train_cuda=1, start to train bf16 on cuda"
            python train_bf16.py --config_path config_bf16.json --bf16-train-cuda || true
        else
            echo "bf16_train_cpu/cuda=0, start to train fp32"
            python train_bf16.py --config_path config_bf16.json || true
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
