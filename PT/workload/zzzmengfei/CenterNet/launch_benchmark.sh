#!/bin/bash
set -xe
# import common funcs
source ../common.sh

# CenterNet
function main {
    # prepare workload
    workload_dir="${PWD}"
    source_code_dir="../../CenterNet/"
    init_model ${source_code_dir}
    #
    patch -p1 < ${workload_dir}/CenterNet.patch || true
    pip install -U -r ${workload_dir}/requirements.txt
    # pip install --no-deps torchvision -f https://download.pytorch.org/whl/torch_stable.html
    cd models/py_utils/_cpools/ && python setup.py install
    cd ../../../external/ && make
    cd ../data/coco/PythonAPI/ && make
    cd ../../../
    mkdir -p data/coco/images/
    ln -sf /home2/pytorch-broad-models/COCO2014/annotations/ data/coco/
    ln -sf /home2/pytorch-broad-models/COCO2014/*2014 data/coco/images/
    mkdir -p cache/nnet/CenterNet-52/
    ln -sf /home2/pytorch-broad-models/CenterNet-52/CenterNet-52_480000.pkl cache/nnet/CenterNet-52/

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
        python test.py CenterNet-52 --arch CenterNet-52 --evaluate --testiter 480000 \
            --split validation --max_iters 2 --warmup 1 --batch_size 2 \
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
            python test.py CenterNet-52 --arch CenterNet-52 --evaluate --testiter 480000 \
                --split validation --max_iters ${num_iter} --warmup ${num_warmup} \
                --batch_size ${batch_size} \
                --precision ${precision} --channels_last ${channels_last} \
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
