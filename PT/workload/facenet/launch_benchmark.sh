#!/bin/bash
set -xe
# import common funcs
source ../common.sh

# facenet
function main {
    # prepare workload
    workload_dir=${PWD}
    source_code_dir="../../facenet"
    source_code_name="facenet"
    init_model ${source_code_dir} ${source_code_name}

    # setup
    ## facenet - have to source-install torchvision 0.9.0 instead of using 0.10.0
    export CUDA_HOME=/usr/local/cuda
    git clone https://github.com/pytorch/vision.git
    cd vision
    git checkout -t remotes/origin/release/0.9
    pip uninstall -y torchvision
    python setup.py install
    apt-get install ffmpeg libsm6 libxext6  -y
    pip install opencv-python
    cd ..
    
    # set common info
    init_params $@
    set_environment
    fetch_cpu_info
    logs_path_clean
    
    cp -r ${workload_dir}/testsmall ./data
    cp ${workload_dir}/train_example.py .
    if [ "${bf16_train_cpu}" == "1" ];then
        echo is train with bf16 cpu...
        cp ${workload_dir}/inception_resnet_v1_cpu.py ./models
        rm ./models/inception_resnet_v1.py
        mv ./models/inception_resnet_v1_cpu.py ./models/inception_resnet_v1.py
        cp ${workload_dir}/training_mod_for_bf16cpu.py ./models/utils/
        rm ./models/utils/training.py
        mv ./models/utils/training_mod_for_bf16cpu.py ./models/utils/training.py 
    elif [ "${bf16_train_cuda}" == "1" ];then
        echo is train with bf16 cuda...
        cp ${workload_dir}/inception_resnet_v1_cuda.py ./models
        rm ./models/inception_resnet_v1.py
        mv ./models/inception_resnet_v1_cuda.py ./models/inception_resnet_v1.py
    else
        echo "use fp32 to train, do nothing here"
    fi
    python setup.py install

    model_name="facenet"
    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        #
        python train_example.py || true
        #
        #for batch_size in ${batch_size_list[@]}
        #do
            #generate_core
            #collect_perf_logs
        #done
    done
    
    pip uninstall -y torchvision
    if [ "${bf16_train_cuda}" == "1" ];then
        pip uninstall -y torch
        pip3 install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html -U
    else
        pip install --no-deps torchvision # restore 0.10.0 torchvision
    fi
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
    throughput=$(grep 'throughput:' ${log_dir}/rcpi* |sed -e 's/.*throughput//;s/,.*//;s/[^0-9.]//g' |awk '
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
