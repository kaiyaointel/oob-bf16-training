#!/bin/bash
set -xe
# import common funcs
source ../common.sh

# deit
function main {
    # prepare workload
    workload_dir="${PWD}"
    source_code_dir="../../deit/"
    init_model ${source_code_dir}

    patch -p1 < ${workload_dir}/deit.patch || true
    # pip install -r ${workload_dir}/requirements.txt
    # pip install timm==0.3.2 --no-deps
    pip install --no-deps git+https://github.com/rwightman/pytorch-image-models.git
    # pip install --no-deps torchvision -f https://download.pytorch.org/whl/torch_stable.html
    if [ ! -e deit_base_patch16_224-b5f2ef4d.pth ];then
        ln -sf /home2/pytorch-broad-models/deitb/model/deit_base_patch16_224-b5f2ef4d.pth .
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
        python main.py --dummy --eval --arch Deit-B --resume deit_base_patch16_224-b5f2ef4d.pth -w 1 -i 2 \
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
            python main.py --dummy --eval --arch Deit-B \
                --resume deit_base_patch16_224-b5f2ef4d.pth \
                -w ${num_warmup} -i ${num_iter} \
                --batch-size ${batch_size} \
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
    # latency=$(grep 'Throughput is:' ${log_dir}/rcpi* |sed -e 's/.*Throughput//;s/,.*//;s/[^0-9.]//g' |awk -v bs=${batch_size} '
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
    throughput=$(grep 'Throughput is:' ${log_dir}/rcpi* |sed -e 's/.*Throughput//;s/,.*//;s/[^0-9.]//g' |awk '
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
