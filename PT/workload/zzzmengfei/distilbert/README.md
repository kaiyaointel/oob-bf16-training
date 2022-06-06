# BERT_LARGE_MRPC

## dependency
```bash
git clone https://github.com/huggingface/transformers.git 
cd transformers
git reset --hard 03ec02a667d5ed3075ea65b9f89ef7135e97f6b4
git apply ../workload/distilbert/distilbert.patch
python setup.py install
cd examples
pip install -r requirements.txt
```

## dataset

```bash
python ./utils/download_glue_data.py --data_dir ${dataset_path} --tasks MRPC
```

## run
 

```python
cd examples/text-classification/
python run_glue.py   --model_name_or_path distilbert-base-uncased \
                     --task_name MRPC \
                     --do_eval \
                     --data_dir ./MRPC/  \
                     --max_seq_length 128 \
                     --per_device_eval_batch_size 1 \
                     --output_dir ./mrpc_output/ \
                     --mkldnn \
                     --jit  (failed)


"""
will get output like:

 time cost 8.719177484512329
 total samples 397 
 inference latency: 0.021962663688947934s
 inference Throughput: 45.53181773225534 images/s

"""

```