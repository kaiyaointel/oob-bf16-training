# bert_large is merged in huggingface_models now. This repo is a bak and will be deleted soon.

# BERT_LARGE_MRPC

## dependency
```bash
git clone https://github.com/huggingface/transformers.git 
checkout to 03ec02a667d5ed3075ea65b9f89ef7135e97f6b4
cd transformer/ && git apply bert_large.patch
python setup.py install
```

## dataset

```bash
python ./utils/download_glue_data.py --data_dir ${dataset_path} --tasks MRPC
```

## run
 

```python
cd examples/text-classification/
python run_glue.py   --model_name_or_path bert-large-cased \
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

 time cost 95.9666497707367
 total samples 397
 inference latency: 0.24172959639984054s
 inference Throughput: 4.136853802320168 images/s

"""

```