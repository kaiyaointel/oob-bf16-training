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

    model_name="Attention_OCR"
    PATCH_DIR=`pwd`
}
init_params $@

if [ ! -d "models" ]; then
  git clone https://github.com/tensorflow/models.git
fi

cd models/research/attention_ocr/python/
git reset 32671b --hard

cp ../../../../attention_ocr.patch . && git apply attention_ocr.patch

# dataset
ln -s /home2/tensorflow-broad-product/oob_tf_models/mlp/attention_ocr/ attention_ocr
cp -r datasets/testdata/ datasets/data/
MYDIR=`pwd`
echo $MYDIR
cp ../../../../launch_benchmark.sh .
./launch_benchmark.sh --checkpoint="" \
                           --model_name=${model_name} \
                           --precision=${precision} \
                           --run_perf=${run_perf} \
                           --collect_dnnl_verbose=${collect_dnnl_verbose} \
                           --workspace=${workspace} \
                           --framework_version=${framework_version} \
                           --framework=${framework}
