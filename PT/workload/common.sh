#!/bin/bash

# init model
function init_model {
    rm -rf ${1}
    git checkout ${1}
    cd ../../.. #"You need to run this command from the toplevel of the working tree"
    git submodule update --init PT/${2}
    cd PT/${2}
}

# parameters
function init_params {
    if [ "${WORKSPACE}" == "" ];then
        WORKSPACE=${PWD}/logs
    fi
    framework='pytorch'
    model_name=''
    batch_size=8
    numa_nodes_use=1
    dnnl_verbose=1
    bf16_train_cpu=0
    bf16_train_cuda=1
    precision='bfloat16'
    #
    for var in $@
    do
        case ${var} in
            --workspace=*|-ws=*)
                WORKSPACE=$(echo $var |cut -f2 -d=)
            ;;
            --numa_nodes_use=*|--numa=*)
                numa_nodes_use=$(echo $var |cut -f2 -d=)
            ;;
            --cores_per_instance=*)
                cores_per_instance=$(echo $var |cut -f2 -d=)
            ;;
            --profile=*)
                profile=$(echo $var |cut -f2 -d=)
            ;;
            --dnnl_verbose=*)
                dnnl_verbose=$(echo $var |cut -f2 -d=)
            ;;
            --channels_last=*)
                channels_last=$(echo $var |cut -f2 -d=)
            ;;
            --framework=*)
                framework=$(echo $var |cut -f2 -d=)
            ;;
            --model_name=*|--model=*|-m=*)
                model_name=$(echo $var |cut -f2 -d=)
            ;;
            --mode_name=*|--mode=*)
                mode_name=$(echo $var |cut -f2 -d=)
            ;;
            --precision=*|--mode=*)
                precision=$(echo $var |cut -f2 -d=)
            ;;
            --batch_size=*|-bs=*|-b=*)
                batch_size=$(echo $var |cut -f2 -d=)
            ;;
            --num_warmup=*|--warmup=*|-w=*)
                num_warmup=$(echo $var |cut -f2 -d=)
            ;;
            --num_iter=*|--iter=*|-i=*)
                num_iter=$(echo $var |cut -f2 -d=)
            ;;
            --bf16_train_cpu=*)
                bf16_train_cpu=$(echo $var |cut -f2 -d=)
            ;;
            --bf16_train_cuda=*)
                bf16_train_cuda=$(echo $var |cut -f2 -d=)
            ;;
            *)
                echo "ERROR: No such param: ${var}"
                exit 1
            ;;
        esac
    done
}

# environment
function set_environment {
    #
    export KMP_BLOCKTIME=1
    export KMP_AFFINITY=granularity=fine,verbose,compact,1,0

    # DNN Verbose
    if [ "${dnnl_verbose}" == "1" ];then
        export DNNL_VERBOSE=1
        export MKLDNN_VERBOSE=1
    else
        unset DNNL_VERBOSE MKLDNN_VERBOSE
    fi
    
    # AMX
    if [ "${precision}" == "bfloat16" ];then
        export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
    else
        unset DNNL_MAX_CPU_ISA
    fi
    
    # Profile
    addtion_options=" ${OOB_ADDITION_PARAMS} "
    if [ "${profile}" == "1" ];then
        addtion_options+=" --profile "
    fi
}

# cpu info
function fetch_cpu_info {
    # hardware
    hostname
    cat /etc/os-release
    cat /proc/sys/kernel/numa_balancing
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
    lscpu
    #echo q | htop | aha --line-fix | html2text -width 1920 | grep -v -E "F1Help|xml version=|agent.jar" || true
    uname -a
    free -h
    numactl -H
    sockets_num=$(lscpu |grep 'Socket(s):' |sed 's/[^0-9]//g')
    cores_per_socket=$(lscpu |grep 'Core(s) per socket:' |sed 's/[^0-9]//g')
    phsical_cores_num=$( echo "${sockets_num} * ${cores_per_socket}" |bc )
    numa_nodes_num=$(lscpu |grep 'NUMA node(s):' |sed 's/[^0-9]//g')
    cores_per_node=$( echo "${phsical_cores_num} / ${numa_nodes_num}" |bc )
    if [ "${numa_nodes_use}" == "all" ];then
        numa_nodes_use=$(lscpu |grep 'NUMA node(s):' |awk '{print $NF}')
    fi
    cpu_model="$(lscpu |grep 'Model name:' |sed 's/.*: *//')"
    if [[ "${cpu_model}" == *"8180"* ]];then
        device_type="SKX"
    elif [[ "${cpu_model}" == *"8280"* ]];then
        device_type="CLX"
    elif [[ "${cpu_model}" == *"8380H"* ]];then
        device_type="CPX"
    elif [[ "${cpu_model}" == *"8380"* ]];then
        device_type="ICX"
    else
        device_type="SPR"
    fi
    # cpu array
    cpu_array=($(numactl -H |grep "node [0-9]* cpus:" |sed "s/.*node [0-9]* cpus: *//" |\
    head -${numa_nodes_use} |cut -f1-${cores_per_node} -d' ' |sed 's/$/ /' |tr -d '\n' |awk -v cpi=${cores_per_instance} -v cpn=${cores_per_node} '{
        for( i=1; i<=NF; i++ ) {
            if(i % cpi == 0 || i % cpn == 0) {
                print $i","
            }else {
                printf $i","
            }
        }
    }' |sed "s/,$//"))
    instance=${#cpu_array[@]}

    # environment
    gcc -v
    python -V
    pip list
    git remote -v
    git branch
    #git show -s
    fremework_version="$(pip list |& grep -E "^torch[[:space:]]|^pytorch[[:space:]]" |awk '{printf("%s",$2)}')"
}

function logs_path_clean {
    # logs saved
    log_dir="${WORKSPACE}/${framework}-${model_name}-${mode_name}-${precision}-bs${batch_size}-"
    log_dir+="cpi${cores_per_instance}-ins${instance}-nnu${numa_nodes_use}-$(date +'%s')"
    mkdir -p ${log_dir}
    if [ ! -e ${WORKSPACE}/summary.log ];then
        printf "framework, model_name, mode_name, precision, batch_size, " | tee ${WORKSPACE}/summary.log
        printf "cores_per_instance, instance, throughput, comp_time, mem_time, link, \n" | tee -a ${WORKSPACE}/summary.log
    fi
    # exec cmd
    excute_cmd_file="${log_dir}/${framework}-run-$(date +'%s').sh"
    rm -f ${excute_cmd_file}
}

function collect_perf_logs {
    comp_time=0
    mem_time=0
    # performance
    if [ $[ ${dnnl_verbose} + ${profile} ] -eq 0 ];then
        # mlpc dashboard json
        generate_json_details no_input ${log_dir}/mlpc_perf.json benchmark
    fi
    # dnnl verbose
    if [ "${dnnl_verbose}" == "1" ];then
        for i_file in $(find ${log_dir}/ -type f -name "rcpi*.log" |sort)
        do
            python ${workload_dir}/../../dnnl_parser.py -f ${i_file} >> ${log_dir}/dnnlverbose.log 2>&1 || true
            break
        done
        # mlpc dashboard json
        generate_json_details ${log_dir}/dnnlverbose.log ${log_dir}/mlpc_dnnl.json
    fi
    # profiling
    if [ "${profile}" == "1" ];then
        mv ./timeline ${log_dir}
        for i_file in $(find ${log_dir}/timeline -type f -name "timeline*.json" |sort)
        do
            python ${workload_dir}/../../profile_parser.py -f ${i_file} >> ${log_dir}/profiling.log 2>&1 || true
            break
        done
        # mlpc dashboard json
        generate_json_details ${log_dir}/profiling.log ${log_dir}/mlpc_prof.json
    fi
    # summary
    artifact_url="${BUILD_URL}artifact/$(basename ${log_dir})"
    printf "${framework}, ${model_name}, ${mode_name}, ${precision}, " |tee ${log_dir}/result.txt |tee -a ${WORKSPACE}/summary.log
    printf "${batch_size}, ${cores_per_instance}, ${instance}, ${throughput}, " |tee -a ${log_dir}/result.txt |tee -a ${WORKSPACE}/summary.log
    printf "${comp_time}, ${mem_time}, ${artifact_url}, \n" |tee -a ${log_dir}/result.txt |tee -a ${WORKSPACE}/summary.log
}

function generate_json_details {
    input_log="$1"
    json_file="$2"
    if [ "$3" == "benchmark" ];then
        echo -e """
            {
                'framework' : 'PyTorch',
                'category' : 'OOB_performance_v2',
                'device' : '${device_type}',
                'period' : '$(date +%Y)WW$(date -d 'next week'  +%U)',
                'quarter': 'Q$(date +%q),,,$(date +%y)',
                'datatype' : '${precision}',
                'results' : [
                    {
                        'bs' : '${batch_size}',
                        'framework_version' : '${fremework_version}',
                        'core/instance' : '${cores_per_instance}',
                        'comp_op' : '',
                        'cast_op' : '',
                        'model_name' : '${model_name}',
                        'perf' : '${throughput}',
                        'instance' : '${instance}'
                    }
                ]
            }
        """ |sed "s/'/\"/g;s/,,,/'/" > ${json_file}
    else
        # comp/mem time
        op_time=($(
            grep "[0-9]$" ${input_log} |awk 'BEGIN {
                comp_time = 0;
                mem_time = 0;
            } {
                if($1 ~/matmul|conv/) {
                    comp_time += $2;
                }else {
                    mem_time += $2;
                }
            }END {
                printf("%.3f  %.3f", comp_time, mem_time);
            }'
        ))
        comp_time=${op_time[0]}
        mem_time=${op_time[1]}
        # json
        echo -e """
            {
                'framework' : 'PyTorch',
                'category' : 'OOB_timeline',
                'device' : '${device_type}',
                'period' : '$(date +%Y)WW$(date -d 'next week'  +%U)',
                'quarter': 'Q$(date +%q),,,$(date +%y)',
                'datatype' : '${precision}',
                'results' : [
                    {
                        'model_name' : '${model_name}',
                        'cast_op': '',
                        'comp_op':'',
                        'bs' : '${batch_size}',
                        'framework_version' : '${fremework_version}',
                        'perf' : '${throughput}',
                        'instance' : '${instance}',
                        'core/instance' : '${cores_per_instance}',
                        'details': [
            $(
                grep "[0-9]$" ${input_log} |sort |awk 'BEGIN {
                    op_name = "tmp_name";
                    op_time = 0;
                    op_calls = 0;
                }{
                    if(op_name ~/.*matmul.*|.*conv.*/) {
                        type = "cpu";
                    }else {
                        type = "mem";
                    }
                    if($1 != op_name) {
                        if(op_name != "tmp_name") {
                            printf("{\"calls\":\"%d\", \"type\":\"%s\", \"primitive\":\"%s\", \"time_ms\":\"%.3f\"},\n", op_calls, type, op_name, op_time);
                        }
                        op_name = $1;
                        op_time = $2;
                        op_calls = $3;
                    }else {
                        op_time += $2;
                        op_calls += $3;
                    }
                }END {
                    printf("{\"calls\":\"%d\", \"type\":\"%s\", \"primitive\":\"%s\", \"time_ms\":\"%.3f\"}\n", op_calls, type, op_name, op_time);
                }'
            )
                        ]
                    }
                ]
            }
        """ |sed "s/'/\"/g;s/,,,/'/" > ${json_file}
    fi
    # post
    if [ "${OOB_MLPC_DASHBOARD}" == "1" ];then
        post_mlpc
    fi
}

function post_mlpc {
    # post data to mlpc dashboard
    mlpc_api="http://mlpc.intel.com/api/store_oob"
    curl -X POST -H "Content-Type: application/json" -d @${json_file} ${mlpc_api}
}
