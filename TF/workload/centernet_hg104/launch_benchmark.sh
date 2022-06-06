#!/bin/bash -x
# ./xxx.sh pb_dir0 pb_dir1 ...

function main {
    #
    fetch_cpu_info
    set_environment
    init_params $@
    
    if [ "$workspace" == "" ];then
        workspace="./OOB_TF_Logs/"
    fi

    # Launch benchmarking
    for bs in ${batch_size[@]}
    do
        #
        for cpi in ${cores_per_instance[@]}
        do
            # set env for TF NativeFormat
            if [ "$use_TF_NativeFormat" == "1" ];then
                export TF_ENABLE_MKL_NATIVE_FORMAT=1
            fi
            generate_core
            collect_logs
            if [ "$use_TF_NativeFormat" == "1" ];then
                unset TF_ENABLE_MKL_NATIVE_FORMAT
            fi
        done
    done

}

# cpu info
function fetch_cpu_info {
    #
    hostname
    cat /etc/os-release
    cat /proc/sys/kernel/numa_balancing
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
    lscpu
    free -h
    numactl -H
    sockets_num=$(lscpu |grep 'Socket(s):' |sed 's/[^0-9]//g')
    cores_per_socket=$(lscpu |grep 'Core(s) per socket:' |sed 's/[^0-9]//g')
    phsical_cores_num=$( echo "${sockets_num} * ${cores_per_socket}" |bc )
    numa_nodes_num=$(lscpu |grep 'NUMA node(s):' |sed 's/[^0-9]//g')
    cores_per_node=$( echo "${phsical_cores_num} / ${numa_nodes_num}" |bc )
    numa_nodes_use=1
    cores_per_instance=${cores_per_node}
    
    #
    gcc -v
    python -V
    git remote -v
    git branch
    git show |head -5

}

# environment
function set_environment {
    #
    export KMP_BLOCKTIME=1
    export KMP_AFFINITY=granularity=fine,verbose,compact,1,0

}

# init params
function init_params {

    period="$(date +%Y)WW$(date +%W)"
    mode="realtime"
    mode_="Inference (real time)"
    framework='Intel-TensorFlow'
    precision="FP32"
    unit='images/s'
    submit_date="$(date +%m/%d/%Y)"
    category="$(date +%s)"
    framework_version="https://gitlab.devtools.intel.com/TensorFlow/Direct-Optimization/private-tensorflow.git a090dde5d843a65af33764b47dcec5eca2ef0c32"
    dnnl="oneDNN v1.x"
    #platform='CLX_8280'
    #system_os="2S28C-CLX 8280 192GB DDR4 2933 (16GB*6*2) Ubuntu 18.04.3 LTS"
    dataset="Dummy"
    dummy_or_not='Dummy'
    model_repo="$(git remote -v |head -1 |awk '{print $2}') $(git rev-parse HEAD)"
    
    batch_size=(1)
    cores_per_instance=(4)
    checkpoint=""
    run_perf=1
    collect_dnnl_verbose=0
    use_TF_NativeFormat=1
    
    for var in $@
    do
        case $var in
            --workspace=*)
                workspace=$(echo $var |cut -f2 -d=)
            ;;
            --framework=*)
                framework=$(echo $var |cut -f2 -d=)
            ;;
            --framework_version=*)
                framework_version=$(echo $var |cut -f2 -d=)
            ;;
            --precision=*)
                precision=$(echo $var |cut -f2 -d=)
            ;;
            --mode=*)
                mode=$(echo $var |cut -f2 -d=)
            ;;
	        --model_name=*)
                model_name=$(echo $var |cut -f2 -d=)
            ;;
            --batch_size=*)
                batch_size=($(echo $var |cut -f2 -d= |sed 's/,/ /g'))
            ;;
            --cores_per_instance=*)
                cores_per_instance=($(echo $var |cut -f2 -d= |sed 's/,/ /g'))
            ;;
            --checkpoint=*)
                checkpoint=$(echo $var |cut -f2 -d=)
            ;;
            --run_perf=*)
                run_perf=$(echo $var |cut -f2 -d=)
            ;;
            --collect_dnnl_verbose=*)
                collect_dnnl_verbose=$(echo $var |cut -f2 -d=)
            ;;
            --use_TF_NativeFormat=*)
                use_TF_NativeFormat=$(echo $var |cut -f2 -d=)
            ;;
            *)
                echo "Error: No such parameter: ${var}"
                exit 1
            ;;
        esac
    done
    echo "###################### run_perf is : ${run_perf}"
}

# run
function generate_core {
    # cpu array
    cpu_array=($(numactl -H |grep "node [0-9]* cpus:" |sed "s/.*node [0-9]* cpus: *//" |\
    head -${numa_nodes_use} |cut -f1-${cores_per_node} -d' ' |sed 's/$/ /' |tr -d '\n' |awk -v cpi=${cpi} -v cpn=${cores_per_node} '{
        for( i=1; i<=NF; i++ ) {
            if(i % cpi == 0 || i % cpn == 0) {
                print $i","
            }else {
                printf $i","
            }
        }
    }' |sed "s/,$//"))
    instance=${#cpu_array[@]}

    if [ "$run_perf" == "1" ];then
        # set run command
        log_dir="${workspace}/${framework}-${model_name}-${mode}-${precision}-bs${bs}-cpi${cpi}-ins${instance}-nnu${numa_nodes_use}-$(date +'%s')"
        mkdir -p ${log_dir}
        excute_cmd_file="${log_dir}/${framework}-run-$(date +'%s').sh"
        rm -f ${excute_cmd_file}
    fi

    if [ "$collect_dnnl_verbose" == "1" ];then
        # set for collect dnnl verbose
        verbose_log_dir="${workspace}/${framework}-${model_name}-${mode}-${precision}-bs${bs}-cpi${cpi}-ins${instance}-nnu${numa_nodes_use}-$(date +'%s')/verbose_logs"
        mkdir -p ${verbose_log_dir}
        with_dnnl_verbose_excute_cmd_file="${verbose_log_dir}/${framework}-run-with-dnnl-verbose-$(date +'%s').sh"
        rm -f ${with_dnnl_verbose_excute_cmd_file}
    fi

   
    echo "******************************************** ${model_name}"

    ############### RUN PERFORMANCE ###############
    if [ "$run_perf" == "1" ];then
      
		for(( i=0; i<instance; i++ ))
		do
			real_cores_per_instance=$(echo ${cpu_array[i]} |awk -F, '{print NF}')
			log_file="${log_dir}/rcpi${real_cores_per_instance}_ins${i}.log"

			printf "numactl --localalloc --physcpubind ${cpu_array[i]} \
                timeout 7200 python3 run_object_detection_saved_model.py --num_iter 400 --num_warmup 50 \
            > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
		done

        echo -e "\n wait" >> ${excute_cmd_file}
        echo -e "\n\n\n bs: $bs, cores_per_instance: $cpi, instance: ${instance} is Running"

        sleep 3
        source ${excute_cmd_file}
    fi
    ############### RUN PERFORMANCE DONE ################


    ############### RUN DNNL VERBOSE ###############
    if [ "$collect_dnnl_verbose" == "1" ];then
        export DNNL_VERBOSE=1
        export MKLDNN_VERBOSE=1
       
		for(( i=0; i<instance; i++ ))
		do
			real_cores_per_instance=$(echo ${cpu_array[i]} |awk -F, '{print NF}')
			verbose_log_file="${verbose_log_dir}/rcpi${real_cores_per_instance}_ins${i}_dnnl_verbose.log"

			printf "numactl --localalloc --physcpubind ${cpu_array[i]} \
                timeout 7200  python3 run_object_detection_saved_model.py --num_iter 20 --num_warmup 10 \
			> ${verbose_log_file} 2>&1 &  \n" |tee -a ${with_dnnl_verbose_excute_cmd_file}
		done

        echo -e "\n wait" >> ${with_dnnl_verbose_excute_cmd_file}
        echo -e "\n\n\n bs: $bs, cores_per_instance: $cpi, instance: ${instance} is Running"

        sleep 3
        source ${with_dnnl_verbose_excute_cmd_file}

        parse_verbose_file="${verbose_log_dir}/rcpi4_ins1_dnnl_verbose.log"
        python parsednn.py -f ${parse_verbose_file} 2>&1 > "$verbose_log_dir/verbose_parse.txt"

        unset MKLDNN_VERBOSE
        unset DNNL_VERBOSE
    fi
    ############### RUN DNNL VERBOSE DONE ###############
}

# collect logs
function collect_perf_logs {
    #
    latency=$(grep 'Throughput:' ${log_dir}/rcpi* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk -v bs=$bs '
    BEGIN {
        sum = 0;
        i = 0;
    }
    {
        sum = sum + bs / $1 * 1000;
        i++;
    }
    END {
        sum = sum / i;
        printf("%.3f", sum);
    }')

    throughput=$(grep 'Throughput:' ${log_dir}/rcpi* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk '
    BEGIN {
        sum = 0;
    }
    {
        sum = sum + $1;
    }
    END {
        printf("%.2f", sum);
    }')
    #
    # echo logs
    input_shape=$(grep 'Find input node:' ${log_dir}/rcpi*ins0.log |head -1 |sed 's/.*node: *//')
    key_params="$(head -n 1 ${excute_cmd_file})"
}

function collect_verbose_logs {
    compute_op_time=$(grep 'compute-op-time' ${verbose_log_dir}/verbose_parse.txt |sed -e 's/.*compute-op-time//;s/[^0-9.]//g')
}

function collect_logs {
    # initial results
    throughput=0
    compute_op_time=0
    input_shape=""
    key_params=""

    # collect logs
    if [ "${run_perf}" == "1" ];then
        collect_perf_logs
    fi

    if [ "${collect_dnnl_verbose}" == "1" ];then
        collect_verbose_logs
    fi

    #echo "${type};${period};${mode_};${model_name};${platform};${framework};${precision};${numa_nodes_use};${instance}x${cores_per_instance};\
    #${bs};${goal};${throughput};${unit};${notes};${submit_date};${category};${framework_version};${latency};${dnnl};${system_os};${log};\
    #${input_shape};${dataset};${dummy_or_not};${key_params};${model_repo}" |tee -a ${workspace}/summary.log
    echo "${period}; ${model_name}; ${precision}; ${throughput}; ${compute_op_time}; ${bs}; ${instance}; ${cores_per_instance}; ${framework}; ${framework_version}; " | tee -a ${workspace}/summary.log
}


# Start
main "$@"

