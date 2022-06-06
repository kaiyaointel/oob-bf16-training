# Huggingface Group (enabled 4 workloads)  
## Xeon    
```
cd ../../transformers
python setup.py install
```
Download GLUE data by (sometimes connection might crash and raise ConnectionReserError, just try again later)
```
cp ../workload/huggingface_models/download_glue_dataset.py .
python download_glue_dataset.py --data_dir glue_data --tasks MRPC
```
(if you run original download_glue_dataset.py you must add these 3 lines below otherwise you can't download, but you can omit this now as download_glue_dataset.py in this folder is already the modified version)  
```
import io  
URLLIB = urllib.request  
'MRPC':'https://raw.githubusercontent.com/MegEngine/Models/master/official/nlp/bert/glue_data/MRPC/dev_ids.tsv' inside the TASK2PATH  
```
Set env var
```
export GLUE_DIR=/home2/kyao/extended-broad-product/PT/transformers/glue_data/  
export TASK_NAME=MRPC  
```  
Appy patch by
```
git apply ../workload/huggingface_models/huggingface-training.patch
```
### albert  
```
python examples/run_glue.py  --model_type albert \
  --model_name_or_path albert-base-v1 \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --overwrite_output_dir \
  --output_dir /tmp/$TASK_NAME/
```
### xlm-roberta  
```
python examples/run_glue.py \
  --model_type xlm-roberta \
  --model_name_or_path xlm-roberta-base \
  --task_name $TASK_NAME \
  --do_train \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --overwrite_output_dir \
  --output_dir /tmp/$TASK_NAME/
```
### roberta  
```
python examples/run_glue.py \
  --model_type roberta \
  --model_name_or_path roberta-base \
  --task_name $TASK_NAME \
  --do_train \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --overwrite_output_dir \
  --output_dir /tmp/$TASK_NAME/
```
### xlnet  
```
python examples/run_glue.py \
  --model_type xlnet \
  --model_name_or_path xlnet-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --overwrite_output_dir \
  --output_dir /tmp/$TASK_NAME/
```  
## CUDA   
```
cd ../../transformers
python setup.py install
```
Download GLUE data by (sometimes connection might crash and raise ConnectionReserError, just try again later)
```
cp ../workload/huggingface_models/download_glue_dataset.py .
python download_glue_dataset.py --data_dir glue_data --tasks MRPC
```
(if you run original download_glue_dataset.py you must add these 3 lines below otherwise you can't download, but you can omit this now as download_glue_dataset.py in this folder is already the modified version)  
```
import io  
URLLIB = urllib.request  
'MRPC':'https://raw.githubusercontent.com/MegEngine/Models/master/official/nlp/bert/glue_data/MRPC/dev_ids.tsv' inside the TASK2PATH  
```
Set env var
```
export GLUE_DIR=/home2/kyao/extended-broad-product/PT/transformers/glue_data/  
export TASK_NAME=MRPC  
```  
Appy patch by
```
git apply ../workload/huggingface_models/huggingface-training-cuda.patch
```
### albert  
```
python examples/run_glue.py  --model_type albert \
  --model_name_or_path albert-base-v1 \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --overwrite_output_dir \
  --output_dir /tmp/$TASK_NAME/
```
### xlm-roberta  
```
python examples/run_glue.py \
  --model_type xlm-roberta \
  --model_name_or_path xlm-roberta-base \
  --task_name $TASK_NAME \
  --do_train \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --overwrite_output_dir \
  --output_dir /tmp/$TASK_NAME/
```
### roberta  
```
python examples/run_glue.py \
  --model_type roberta \
  --model_name_or_path roberta-base \
  --task_name $TASK_NAME \
  --do_train \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --overwrite_output_dir \
  --output_dir /tmp/$TASK_NAME/
```
### xlnet  
```
python examples/run_glue.py \
  --model_type xlnet \
  --model_name_or_path xlnet-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --overwrite_output_dir \
  --output_dir /tmp/$TASK_NAME/
```