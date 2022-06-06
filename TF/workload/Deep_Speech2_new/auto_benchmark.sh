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

    model_name="DeepSpeech2"
    PATCH_DIR=`pwd`
}
init_params $@

PATCH_DIR=`pwd`
echo ${PATCH_DIR}
if [ ! -d "models" ]; then
    git clone https://github.com/tensorflow/models.git
fi

cd models/research/deep_speech/
git reset 5b9feb6 --hard
pip install -r requirements.txt
pip install tf-models-official

cp ../../../ds2.patch .
git apply ds2.patch


#./run_deep_speech.sh

##### prepare dataset
cp -r /home2/tensorflow-broad-product/oob_tf_models/dpg/Deep_Speech2/dataset dataset
ln -s /home2/tensorflow-broad-product/oob_tf_models/dpg/Deep_Speech2/model/ model

CUR_PATH=`pwd`
dev_clean="${CUR_PATH}/dataset/LibriSpeech/LibriSpeech/dev-clean.csv"
dev_other="${CUR_PATH}/dataset/LibriSpeech/LibriSpeech/dev-other.csv"
echo ${dev_clean}
echo ${dev_other}
wait

# Step 2: generate train dataset and evaluation dataset
echo "Data preprocessing..."
eval_file="${CUR_PATH}/dataset/librispeech_data/eval_dataset.csv"

head -1 $dev_clean > $eval_file
for filename in $dev_clean $dev_other
do
    sed 1d $filename >> $eval_file
done

# Step 3: filter out the audio files that exceed max time duration.
#final_train_file="dataset/librispeech_data/final_train_dataset.csv"
final_eval_file="${CUR_PATH}/dataset/librispeech_data/final_eval_dataset.csv"

MAX_AUDIO_LEN=27.0
awk -v maxlen="$MAX_AUDIO_LEN" 'BEGIN{FS="\t";} NR==1{print $0} NR>1{cmd="soxi -D "$1""; cmd|getline x; if(x<=maxlen) {print $0}; close(cmd);}' $eval_file > $final_eval_file

##### Prepare Dataset done


## prepare weight
ln -s ${Tensorflow_PRETRAIN_DIR}/oob_tf_models/dpg/Deep_Speech2/model/
## prepare weight done

MYDIR=`pwd`
echo $MYDIR

cp ${PATCH_DIR}/launch_benchmark.sh .
cp ${OOB_HOEM_PATH}/parsednn.py .


## FP32
bash ./launch_benchmark.sh --checkpoint=${model_name} --precision=float32 --collect_dnnl_verbose=0 --workspace=${workspace}

## BF16
# bash ./launch_benchmark.sh --checkpoint=${model} --precision=bfloat16 --collect_dnnl_verbose=1 --run_perf=1 --workspace=${LOG_PATH}/oob_pytorch/

