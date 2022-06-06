import groovy.json.*
NODE_LABEL = 'master'
if ('NODE_LABEL' in params) {
    echo "NODE_LABEL in params"
    if (params.NODE_LABEL != '') {
        NODE_LABEL = params.NODE_LABEL
    }
}
echo "NODE_LABEL: $NODE_LABEL"

SUB_NODE_LABEL = 'tfoob'
if ('SUB_NODE_LABEL' in params) {
    echo "SUB_NODE_LABEL in params"
    if (params.SUB_NODE_LABEL != '') {
        SUB_NODE_LABEL = params.SUB_NODE_LABEL
    }
}
echo "SUB_NODE_LABEL: $SUB_NODE_LABEL"

// first support single sub_node, should support multi sub_node through SUB_NODE_LABEL
SUB_NODE_HOSTNAME = ''
if ('SUB_NODE_HOSTNAME' in params) {
    echo "SUB_NODE_HOSTNAME in params"
    if (params.SUB_NODE_HOSTNAME != '') {
        SUB_NODE_HOSTNAME = params.SUB_NODE_HOSTNAME
    }
}
echo "SUB_NODE_HOSTNAME: $SUB_NODE_HOSTNAME"

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

OOB_Q4_20 = ''
if ('OOB_Q4_20' in params) {
    echo "OOB_Q4_20 in params"
    if (params.OOB_Q4_20 != '') {
        OOB_Q4_20 = params.OOB_Q4_20
    }
}
echo "OOB_Q4_20: $OOB_Q4_20"

OOB_Q3_20 = ''
if ('OOB_Q3_20' in params) {
    echo "OOB_Q3_20 in params"
    if (params.OOB_Q3_20 != '') {
        OOB_Q3_20 = params.OOB_Q3_20
    }
}
echo "OOB_Q3_20: $OOB_Q3_20"

OOB_Q1_21 = ''
if ('OOB_Q1_21' in params) {
    echo "OOB_Q1_21 in params"
    if (params.OOB_Q4_20_MODELS != 'Ture') {
        OOB_Q1_21 = params.OOB_Q1_21
    }
}
echo "OOB_Q1_21: $OOB_Q1_21"

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

// param to count completed models
run_model_counts = 0
run_model_completed_counts = 0

def Run_Models_Jobs()
{
    def jobs = [:]

    OOB_Q4_20 = OOB_Q4_20.split(',')
    OOB_Q3_20 = OOB_Q3_20.split(',')
    OOB_Q1_21 = OOB_Q1_21.split(',')
    def MODEL_LIST = OOB_Q4_20 + OOB_Q3_20 + OOB_Q1_21
    run_model_counts = MODEL_LIST.size()
    println("run_model_counts = " + run_model_counts)
    MODEL_LIST.each { case_name ->
        if (case_name != ""){
            List model_params = [
                    string(name: "SUB_NODE_LABEL", value: SUB_NODE_LABEL),
                    string(name: "CONDA_PATH", value: CONDA_PATH),
                    string(name: "VIRTUAL_ENV", value: VIRTUAL_ENV),
                    string(name: "TF_pretrain_path", value: TF_pretrain_path),
                    string(name: "PRECISION", value: PRECISION),
                    string(name: "RUN_PERF", value: RUN_PERF),
                    string(name: "USE_TF_NATIVEFORMAT", value: USE_TF_NATIVEFORMAT),
                    string(name: "COLLECT_DNNL_VERBOSE", value: COLLECT_DNNL_VERBOSE),
                    string(name: "BATCH_SIZE", value: BATCH_SIZE),
                    string(name: "CORES_PER_INSTANCE", value: CORES_PER_INSTANCE),
                    string(name: "FRAMEWORK", value: FRAMEWORK),
                    string(name: "FRAMEWORK_VERSION", value: FRAMEWORK_VERSION),
                    string(name: "MODEL_NAME", value: case_name),
                    string(name: "WHL_URL", value: WHL_URL),
                    string(name: "NUM_WARMUP", value: NUM_WARMUP),
                    string(name: "NUM_ITER", value: NUM_ITER)
            ]

            jobs["${FRAMEWORK}_${case_name}"] = {
                println("---------${FRAMEWORK}_${case_name}_precision----------")
                sub_jenkins_job = "test_suyue_TF_OOB"
                downstreamJob = build job: sub_jenkins_job, propagate: false, parameters: model_params

                catchError {
                    copyArtifacts(
                            projectName: sub_jenkins_job,
                            selector: specific("${downstreamJob.getNumber()}"),
                            filter: 'OOB_TF_Logs/**',
                            fingerprintArtifacts: true,
                            target: "${case_name}",
                            optional: true)

                    // Archive in Jenkins
                    archiveArtifacts artifacts: "${case_name}/**", allowEmptyArchive: true
                    sh """#!/bin/bash
                        if [ -r ${case_name}/OOB_TF_Logs/summary.log ]; then
                            cat ${case_name}/OOB_TF_Logs/summary.log >> ${WORKSPACE}/summary.log
                        else
                            echo "${case_name},failed" >> ${WORKSPACE}/summary.log
                        fi
                    """
                }

                def downstreamJobStatus = downstreamJob.result
                run_model_completed_counts += 1

                if (downstreamJobStatus != 'SUCCESS') {
                    currentBuild.result = "FAILURE"
                }
            }
        }
    }

    return jobs
}

def Status_Check_Job(){
    println("status check job")
    status_check_round = 0
    SUB_NODE_HOSTNAME = SUB_NODE_HOSTNAME.split(',')
    while(run_model_completed_counts < run_model_counts){
        println("run_model_completed_counts = " + run_model_completed_counts)
        println("run_model_counts = " + run_model_counts)
        SUB_NODE_HOSTNAME.each { case_name ->
            println(case_name)
            withEnv(["sub_host_name=$case_name"]){
                sh'''#!/bin/bash
                    status=`timeout 3 ssh pengxiny@${sub_host_name} echo 1`
                    if [ $? != 0 ]; then 
                        echo "reboot"
                    fi
                '''
            }
            status_check_round += 1
            println("status_check_round：" + status_check_round)
            sleep(10)
    }
}
}

node(NODE_LABEL){
    try {
        deleteDir()
        dir("oob_perf") {
            checkout scm
        }

        stage("RUN MODELS") {

            def job_list = [:]
            def model_jobs = Run_Models_Jobs()

            if (model_jobs.size() > 0){
                job_list['Status Check'] = {
                    Status_Check_Job()
                }
                job_list = job_list + model_jobs
            }

            parallel job_list
        }

    }catch(Exception e) {
        currentBuild.result = "FAILURE"
        error(e.toString())
    } finally {
        dir(WORKSPACE){
            sh'''#!/bin/bash
                if [ -f summary.log ]; then
                    cp summary.log results.csv
                fi
            '''
        }
        archiveArtifacts '*.log, *.csv'
    }
}



