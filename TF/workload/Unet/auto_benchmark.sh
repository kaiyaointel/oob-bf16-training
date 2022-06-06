set -x

# init params
function init_params {
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
    if [ "$OOB_HOEM_PATH" == "" ];then
        OOB_HOEM_PATH="~/extended-broad-product"
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

    model_name="Unet"
    PATCH_DIR=`pwd`
}

init_params $@


git clone https://github.com/divamgupta/image-segmentation-keras.git
cd image-segmentation-keras && git reset --hard d2581da52bcd4b0782b9610996f025c6caaea8dc
python setup.py install

ln -s ${tf_pretrain_path}/oob/Unet/dataset dataset
ln -s ${tf_pretrain_path}/oob/Unet/checkpoint checkpoint

cp ${PATCH_DIR}/run_benchmark.py .
cp ${PATCH_DIR}/launch_benchmark.sh .
cp ${OOB_HOME_PATH}/parsednn.py .

bash ./launch_benchmark.sh --checkpoint="" \
                           --model_name=${model_name} \
                           --precision=${precision} \
                           --run_perf=${run_perf} \
                           --collect_dnnl_verbose=${collect_dnnl_verbose} \
                           --workspace=${workspace} \
                           --framework_version=${framework_version} \
                           --framework=${framework}

