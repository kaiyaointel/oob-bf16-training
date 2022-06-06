# CBAM

> Note: this AttRec neet to run with intel-tensorlow==1.15.2

## prepare

### dependency
```bash
pip install intel-tensorflow==1.15.2
pip install tflearn
```

### patch
```bash
git clone https://github.com/kobiso/CBAM-tensorflow.git
cd CBAM-tensorflow && git checkout 808f53
cp CBAM.patch . && git apply CBAM.patch
```
### dataset
> dataset will download automatic when you do benchmark with script.

## run

```bash
python -m pdb ResNeXt.py --attention_module cbam_block
```


## help info
```bash
python ResNeXt.py --help

usage: ResNeXt.py [-h] [--model_name MODEL_NAME]
                  [--attention_module ATTENTION_MODULE]
                  [--weight_decay WEIGHT_DECAY] [--momentum MOMENTUM]
                  [--learning_rate LEARNING_RATE]
                  [--reduction_ratio REDUCTION_RATIO]
                  [--batch_size BATCH_SIZE] [--iteration ITERATION]
                  [--test_iteration TEST_ITERATION]
                  [--total_epochs TOTAL_EPOCHS] [--num_warmup NUM_WARMUP]
                  [--num_eval_iter NUM_EVAL_ITER]

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        model name
  --attention_module ATTENTION_MODULE
                        attention module name you want to use
  --weight_decay WEIGHT_DECAY
                        weight_decay
  --momentum MOMENTUM   momentum
  --learning_rate LEARNING_RATE
                        learning_rate
  --reduction_ratio REDUCTION_RATIO
                        reduction_ratio
  --batch_size BATCH_SIZE
                        batch_size of train and eval
  --iteration ITERATION
                        training iteration
  --test_iteration TEST_ITERATION
                        test iteration
  --total_epochs TOTAL_EPOCHS
                        total_epochs
  --num_warmup NUM_WARMUP
                        num of eval warmup
  --num_eval_iter NUM_EVAL_ITER

```
