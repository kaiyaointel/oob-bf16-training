#!/bin/bash
set -xe
# import common funcs
source ../common.sh

# deepspeech
function main {
    # prepare workload
    workload_dir="${PWD}"
    source_code_dir="../../mlperf_training/"
    init_model ${source_code_dir}

    patch -p1 < ${workload_dir}/deepspeech.patch || true
    pip install -r ${workload_dir}/requirements.txt
    # pip install --no-deps torchvision -f https://download.pytorch.org/whl/torch_stable.html
    if [ ! -d audio ];then
        git clone https://github.com/pytorch/audio.git
        cd audio
        git submodule update --init --recursive
        pip install --no-deps -e .
        cd ../
    fi
    if [ ! -d warp-ctc ];then
        git clone https://github.com/SeanNaren/warp-ctc.git
        cd warp-ctc
        mkdir -p build
        cd build
        cmake .. && make
        cd ../pytorch_binding
        python setup.py install
        cd ../..
    fi
    cd 	speech_recognition/
    rsync -avz ${workload_dir}/*.csv .
    cd pytorch/
    if [ ! -e deepspeech_10.pth ];then
        ln -sf /home2/pytorch-broad-models/DeepSpeech2/models/deepspeech_10.pth .
    fi
    if [ ! -e dataset ];then
        ln -sf /home2/pytorch-broad-models/DeepSpeech2/LibriSpeech_dataset/ ./dataset
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
        # pre run
        python train.py --model_path ./deepspeech_10.pth --seed 1 --batch_size 1 --evaluate --eval_warmup 1 --eval_iter 2 \
            --precision ${precision} --channels_last ${channels_last} || true
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
            python train.py --evaluate --seed 1 --model_path ./deepspeech_10.pth \
                --eval_warmup ${num_warmup} --eval_iter ${num_iter} \
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
