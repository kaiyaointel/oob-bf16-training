set -x

# init params
function init_params {
    for var in $@
    do
        case $var in
            --workspace=*)
                WORKSPACE=$(echo $var |cut -f2 -d=)
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
            --run_perf=*)
                run_perf=$(echo $var |cut -f2 -d=)
            ;;
            --collect_dnnl_verbose=*)
                collect_dnnl_verbose=$(echo $var |cut -f2 -d=)
            ;;
            --oob_home_path=*)
                OOB_HOME_PATH=$(echo $var |cut -f2 -d=)
            ;;
            --tf_pretrain_path=*)
                tf_pretrain_path=$(echo $var |cut -f2 -d=)
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

    if [ "$workspace" == "" ];then
        cur_path=`pwd`
        workspace="${cur_path}/oob_tf_logs/"
    fi
    if [ "$framework" == "" ];then
        framework="Intel-tf"
    fi
    if [ "$tf_pretrain_path" == "" ];then
        tf_pretrain_path="/home2/tensorflow-broad-product/oob_tf_models/"
    fi
    if [ "$OOB_HOME_PATH" == "" ];then
        OOB_HOME_PATH="~/extended-broad-product"
    fi
    if [ "$precision" == "" ];then
        precision="float32"
    fi
    if [ "$run_perf" == "" ];then
        run_perf=1
    fi
    if [ "$collect_dnnl_verbose" == "" ];then
        collect_dnnl_verbose=0
    fi
    if [ "$use_TF_NativeFormat" == "" ];then
        use_TF_NativeFormat=0
    fi

    model="GPT2"
    PATCH_DIR=`pwd`
}

init_params $@

if [ ! -d "gpt-2" ]; then
  git clone https://github.com/openai/gpt-2.git
fi

cd gpt-2/

git reset a74da5d --hard
pip3 install -r requirements.txt
echo "**** ${PATCH_DIR}"
cp ${PATCH_DIR}/gpt-2.patch . 
git apply gpt-2.patch

# dataset
# python download_model.py 124M
# if [ ! -d 'models'];then
#  mkdir -p models
# fi
rm -rf models
ln -s /home2/tensorflow-broad-product/oob_tf_models/dpg/GPT-2/gpt-2/models/ models

cp ${PATCH_DIR}/launch_benchmark.sh .
cp ${PATCH_DIR}/hparams.py ./src/
bash ./launch_benchmark.sh --checkpoint=${model} --precision=${precision} --collect_dnnl_verbose=${collect_dnnl_verbose} --workspace=${workspace}
