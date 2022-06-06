@NonCPS
def jsonParse(def json) {
    new groovy.json.JsonSlurperClassic().parseText(json)
}

SUB_NODE_LABEL = ''
if ('SUB_NODE_LABEL' in params) {
    echo "SUB_NODE_LABEL in params"
    if (params.SUB_NODE_LABEL != '') {
        SUB_NODE_LABEL = params.SUB_NODE_LABEL
    }
}
echo "SUB_NODE_LABEL: $SUB_NODE_LABEL"

CONDA_PATH = ''
if ('CONDA_PATH' in params) {
    echo "CONDA_PATH in params"
    if (params.CONDA_PATH != '') {
        CONDA_PATH = params.CONDA_PATH
    }
}
echo "CONDA_PATH: $CONDA_PATH"

VIRTUAL_ENV = 'oob'
if ('VIRTUAL_ENV' in params) {
    echo "VIRTUAL_ENV in params"
    if (params.VIRTUAL_ENV != '') {
        VIRTUAL_ENV = params.VIRTUAL_ENV
    }
}
echo "VIRTUAL_ENV: $VIRTUAL_ENV"

TF_pretrain_path = ''
if ('TF_pretrain_path' in params) {
    echo "TF_pretrain_path in params"
    if (params.TF_pretrain_path != '') {
        TF_pretrain_path = params.TF_pretrain_path
    }
}
echo "TF_pretrain_path: $TF_pretrain_path"

PRECISION = 'float32'
if ('PRECISION' in params) {
    echo "PRECISION in params"
    if (params.PRECISION != '') {
        PRECISION = params.PRECISION
    }
}
echo "PRECISION: $PRECISION"

RUN_PERF = '1'
if ('RUN_PERF' in params) {
    echo "RUN_PERF in params"
    if (params.RUN_PERF != '') {
        RUN_PERF = params.RUN_PERF
    }
}
echo "RUN_PERF: $RUN_PERF"

USE_TF_NATIVEFORMAT = '1'
if ('USE_TF_NATIVEFORMAT' in params) {
    echo "USE_TF_NATIVEFORMAT in params"
    if (params.USE_TF_NATIVEFORMAT != '') {
        USE_TF_NATIVEFORMAT = params.USE_TF_NATIVEFORMAT
    }
}
echo "USE_TF_NATIVEFORMAT: $USE_TF_NATIVEFORMAT"

COLLECT_DNNL_VERBOSE = '0'
if ('COLLECT_DNNL_VERBOSE' in params) {
    echo "COLLECT_DNNL_VERBOSE in params"
    if (params.COLLECT_DNNL_VERBOSE != '') {
        COLLECT_DNNL_VERBOSE = params.COLLECT_DNNL_VERBOSE
    }
}
echo "COLLECT_DNNL_VERBOSE: $COLLECT_DNNL_VERBOSE"

BATCH_SIZE = ''
if ('BATCH_SIZE' in params) {
    echo "BATCH_SIZE in params"
    if (params.BATCH_SIZE != '') {
        BATCH_SIZE = params.BATCH_SIZE
    }
}

echo "BATCH_SIZE: $BATCH_SIZE"

CORES_PER_INSTANCE = ''
if ('CORES_PER_INSTANCE' in params) {
    echo "CORES_PER_INSTANCE in params"
    if (params.CORES_PER_INSTANCE != '') {
        CORES_PER_INSTANCE = params.CORES_PER_INSTANCE
    }
}

echo "CORES_PER_INSTANCE: $CORES_PER_INSTANCE"

FRAMEWORK = ''
if ('FRAMEWORK' in params) {
    echo "FRAMEWORK in params"
    if (params.FRAMEWORK != '') {
        FRAMEWORK = params.FRAMEWORK
    }
}

echo "FRAMEWORK: $FRAMEWORK"

FRAMEWORK_VERSION = ''
if ('FRAMEWORK_VERSION' in params) {
    echo "FRAMEWORK_VERSION in params"
    if (params.FRAMEWORK_VERSION != '') {
        FRAMEWORK_VERSION = params.FRAMEWORK_VERSION
    }
}
echo "FRAMEWORK_VERSION: $FRAMEWORK_VERSION"

MODEL_NAME = ''
if ('MODEL_NAME' in params) {
    echo "MODEL_NAME in params"
    if (params.MODEL_NAME != '') {
        MODEL_NAME = params.MODEL_NAME
    }
}
echo "MODEL_NAME: $MODEL_NAME"

WHL_URL = ''
if ('WHL_URL' in params) {
    echo "WHL_URL in params"
    if (params.WHL_URL != '') {
        WHL_URL = params.WHL_URL
    }
}
echo "WHL_URL: $WHL_URL"

NUM_WARMUP = ''
if ('NUM_WARMUP' in params) {
    echo "NUM_WARMUP in params"
    if (params.NUM_WARMUP != '') {
        NUM_WARMUP = params.NUM_WARMUP
    }
}
echo "NUM_WARMUP: $NUM_WARMUP"

NUM_ITER = ''
if ('NUM_ITER' in params) {
    echo "NUM_ITER in params"
    if (params.NUM_ITER != '') {
        NUM_ITER = params.NUM_ITER
    }
}
echo "NUM_ITER: $NUM_ITER"

def cleanup(){
    try {
        sh '''#!/bin/bash 
        set -x
        cd $WORKSPACE
        rm -rf *
        lscpu
        '''
    } catch(e) {
        echo "==============================================="
        echo "ERROR: Exception caught in cleanup()           "
        echo "ERROR: ${e}"
        echo "==============================================="
        echo "Error while doing cleanup"
    }
}

def Configuration_Environment(WHL_URL,CONDA_PATH,VIRTUAL_ENV){
    withEnv(["WHL_URL=${WHL_URL}","CONDA_PATH=$CONDA_PATH","conda_env=$VIRTUAL_ENV"]){
        sh'''
        #!/usr/bin/env bash
        set -x
        . ${CONDA_PATH}
        
        #if [ $(conda info -e | grep ${conda_env} | wc -l) != 0 ]; then
        #    conda remove --name ${conda_env} --all -y
        #fi
    
        #conda_dir=$(dirname $(dirname $(which conda)))
        #if [ -d ${conda_dir}/envs/${conda_env} ]; then
        #    rm -rf ${conda_dir}/envs/${conda_env}
        #fi
    
        #conda create python=3.6 -y -n ${conda_env}
        
        conda activate ${conda_env}
        which python
        if [ "${WHL_URL}" == "" ];then
            pip install tensorflow
        else
            mkdir tf_whl && cd tf_whl
            wget ${WHL_URL}
            pip install *.whl
        fi
    '''
    }
}

node(SUB_NODE_LABEL){

    cleanup()
    dir("oob_perf") {
        checkout scm
    }

    try{
        stage("BUILD ENV") {
            Configuration_Environment(WHL_URL, CONDA_PATH, VIRTUAL_ENV)
        }

        def inputJson = jsonParse(readFile("$WORKSPACE/oob_perf/TF/model.json"))
        model_path = inputJson."${model_name}"."model_path"
        stage("RUN MODEL"){
            withEnv(["model_name=$MODEL_NAME", "model_path=${model_path}","CONDA_PATH=$CONDA_PATH","conda_env=$VIRTUAL_ENV", \
            "USE_TF_NATIVEFORMAT=${USE_TF_NATIVEFORMAT}", "RUN_PERF=${RUN_PERF}", "COLLECT_DNNL_VERBOSE=${COLLECT_DNNL_VERBOSE}", \
            "PRECISION1=${PRECISION}", "BATCH_SIZE=${BATCH_SIZE}","CORES_PER_INSTANCE=${CORES_PER_INSTANCE}","FRAMEWORK=${FRAMEWORK}", \
            "FRAMEWORK_VERSION=${FRAMEWORK_VERSION}","TF_pretrain_path=${TF_pretrain_path}","NUM_WARMUP=${NUM_WARMUP}","NUM_ITER=${NUM_ITER}"])
             {

                 sh '''#!/usr/bin/env bash
                    #sudo ln -fs /bin/bash /bin/sh
                    set -xe
                    . ${CONDA_PATH}
                    conda activate ${conda_env}
                    which python
                    cd oob_perf/TF
                    CUR_PATH=`pwd`
                    echo "CUR_PATH = $CUR_PATH"
                    workspace="$WORKSPACE/OOB_TF_Logs/"
                    if [ ! -d ${workspace} ];then
                        mkdir -p ${workspace}
                    fi
                    
                    if  [ -d "${CUR_PATH}/${model_path}" ];then
                        cd ${model_path}
                        source ./auto_benchmark.sh
                    else
                        ./launch_benchmark.sh --checkpoint=${TF_pretrain_path}${model_path} --use_TF_NativeFormat=${USE_TF_NATIVEFORMAT} \
                            --run_perf=${RUN_PERF} --collect_dnnl_verbose=${COLLECT_DNNL_VERBOSE} --precision=${PRECISION1} --workspace=${workspace} \
                            --batch_size=${BATCH_SIZE} --cores_per_instance=${CORES_PER_INSTANCE} --framework=${FRAMEWORK} --framework_version=${FRAMEWORK_VERSION} \
                            --num_iter=${NUM_ITER} --num_warmup=${NUM_WARMUP}
                    fi
                 '''
             }
        }

    }catch(e){
        currentBuild.result = "FAILURE"
        throw e
    }finally{
        // save log files
        stage("Archive Artifacts") {
            archiveArtifacts artifacts: "**/OOB_TF_Logs/**", excludes: null
            fingerprint: true
        }
    }
}
