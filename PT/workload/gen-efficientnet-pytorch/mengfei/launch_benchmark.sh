#!/bin/bash
set -xe
# import common funcs
source ../common.sh

# gen-efficientnet-pytorch
function main {
    # prepare workload
    workload_dir="${PWD}"
    source_code_dir="../../gen-efficientnet-pytorch/"
    init_model ${source_code_dir}

    patch -p1 < ${workload_dir}/gen.patch || true
    cp ${workload_dir}/main.py ./
    pip install -r ${workload_dir}/requirements.txt
    # pip install --no-deps torchvision -f https://download.pytorch.org/whl/torch_stable.html

    # set common info
    init_params $@
    set_environment
    fetch_cpu_info
    logs_path_clean

    # all gen models
    if [ "${model_name}" == "gen-efficientnet-pytorch" ];then
        model_name="alexnet,resnet18,resnet34,resnet50,resnet101,resnet152,squeezenet1_0,"
        model_name+="squeezenet1_1,vgg11,vgg13,vgg16,vgg19,vgg11_bn,vgg13_bn,"
        model_name+="vgg16_bn,vgg19_bn,shufflenet_v2_x0_5,shufflenet_v2_x1_0,googlenet,"
        model_name+="resnext50_32x4d,resnext101_32x8d,wide_resnet50_2,wide_resnet101_2,"
        model_name+="inception_v3,efficientnet_b0,efficientnet_b1,efficientnet_b2,"
        model_name+="efficientnet_b3,efficientnet_b4,efficientnet_b5,efficientnet_b6,"
        model_name+="efficientnet_b7,efficientnet_b8,mnasnet1_0,mnasnet0_5,densenet121,"
        model_name+="densenet169,densenet201,densenet161,fbnetc_100,spnasnet_100,"
    fi
    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        # cache weight
        python ./main.py -e --performance --pretrained --dummy --no-cuda -j 1 -w 1 -i 2 \
            -a ${model_name} -b 1 --precision ${precision} --channels_last ${channels_last} --config_file ${workload_dir}/conf.yaml
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
            python ./main.py -e --performance --pretrained --dummy --no-cuda -j 1 \
                --config_file ${workload_dir}/conf.yaml \
                -w ${num_warmup} -i ${num_iter} \
                -a ${model_name} \
                -b ${batch_size} \
                --precision ${precision} \
                --channels_last ${channels_last} \
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
