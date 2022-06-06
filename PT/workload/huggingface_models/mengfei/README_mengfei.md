# Prepare

## dependency
```
python setup.py install
```

## dataset
Before running anyone of these GLUE tasks you should download the
[GLUE data](https://gluebenchmark.com/tasks) by running
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
and unpack it to some directory `$GLUE_DIR`.

```bash
python download_glue_dataset.py --data_dir glue_data --tasks MRPC
```

## patch

```bash
cd xlent/
git apply ../workload/huggingface_models/xlnet.patch
python setup.py install
```


# Benchmaring
```python
python -u examples/run_glue.py --model_type xlnet \
                               --data_dir ./glue_data/MRPC/ \
                               --model_name_or_path xlnet-base-cased \ 
                               --task_name mrpc \
                               --output_dir xlnet-base-cased \
                               --per_gpu_eval_batch_size 1\
                               --num_warmup_iters 10
                               --no-cuda \
                               --do_eval \
                               --mkldnn \
                               --jit (failed with jit)
'''
# will output like:
 time cost 24.806803226470947
 inference latency: 0.06080098830017389s
 inference Throughput: 16.447101074459674 samples/s
'''
```

## help info
```python
python examples/run_glue.py -h
usage: run_glue.py [-h] --model_name_or_path MODEL_NAME_OR_PATH --model_type
                   MODEL_TYPE [--config_name CONFIG_NAME]
                   [--tokenizer_name TOKENIZER_NAME] [--cache_dir CACHE_DIR]
                   --task_name TASK_NAME --data_dir DATA_DIR
                   [--max_seq_length MAX_SEQ_LENGTH] [--overwrite_cache]
                   --output_dir OUTPUT_DIR [--overwrite_output_dir]
                   [--do_train] [--do_eval] [--evaluate_during_training]
                   [--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE]
                   [--per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE]
                   [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                   [--learning_rate LEARNING_RATE]
                   [--weight_decay WEIGHT_DECAY] [--adam_epsilon ADAM_EPSILON]
                   [--max_grad_norm MAX_GRAD_NORM]
                   [--num_train_epochs NUM_TRAIN_EPOCHS]
                   [--max_steps MAX_STEPS] [--warmup_steps WARMUP_STEPS]
                   [--logging_steps LOGGING_STEPS] [--save_steps SAVE_STEPS]
                   [--save_total_limit SAVE_TOTAL_LIMIT]
                   [--eval_all_checkpoints] [--no_cuda] [--seed SEED] [--fp16]
                   [--fp16_opt_level FP16_OPT_LEVEL] [--local_rank LOCAL_RANK]
                   [--num_warmup_iters NUM_WARMUP_ITERS] [--profiling]
                   [--mkldnn] [--jit]

optional arguments:
  -h, --help            show this help message and exit
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to pre-trained model or shortcut name selected in
                        the list: distilbert-base-uncased, distilbert-base-
                        uncased-distilled-squad, distilbert-base-cased,
                        distilbert-base-cased-distilled-squad, distilbert-
                        base-german-cased, distilbert-base-multilingual-cased,
                        distilbert-base-uncased-finetuned-sst-2-english,
                        albert-base-v1, albert-large-v1, albert-xlarge-v1,
                        albert-xxlarge-v1, albert-base-v2, albert-large-v2,
                        albert-xlarge-v2, albert-xxlarge-v2, camembert-base,
                        umberto-commoncrawl-cased-v1, umberto-wikipedia-
                        uncased-v1, xlm-roberta-base, xlm-roberta-large, xlm-
                        roberta-large-finetuned-conll02-dutch, xlm-roberta-
                        large-finetuned-conll02-spanish, xlm-roberta-large-
                        finetuned-conll03-english, xlm-roberta-large-
                        finetuned-conll03-german, bart-large, bart-large-mnli,
                        bart-large-cnn, bart-large-xsum, mbart-large-en-ro,
                        roberta-base, roberta-large, roberta-large-mnli,
                        distilroberta-base, roberta-base-openai-detector,
                        roberta-large-openai-detector, bert-base-uncased,
                        bert-large-uncased, bert-base-cased, bert-large-cased,
                        bert-base-multilingual-uncased, bert-base-
                        multilingual-cased, bert-base-chinese, bert-base-
                        german-cased, bert-large-uncased-whole-word-masking,
                        bert-large-cased-whole-word-masking, bert-large-
                        uncased-whole-word-masking-finetuned-squad, bert-
                        large-cased-whole-word-masking-finetuned-squad, bert-
                        base-cased-finetuned-mrpc, bert-base-german-dbmdz-
                        cased, bert-base-german-dbmdz-uncased, bert-base-
                        japanese, bert-base-japanese-whole-word-masking, bert-
                        base-japanese-char, bert-base-japanese-char-whole-
                        word-masking, bert-base-finnish-cased-v1, bert-base-
                        finnish-uncased-v1, bert-base-dutch-cased, xlnet-base-
                        cased, xlnet-large-cased, flaubert-small-cased,
                        flaubert-base-uncased, flaubert-base-cased, flaubert-
                        large-cased, xlm-mlm-en-2048, xlm-mlm-ende-1024, xlm-
                        mlm-enfr-1024, xlm-mlm-enro-1024, xlm-mlm-tlm-
                        xnli15-1024, xlm-mlm-xnli15-1024, xlm-clm-enfr-1024,
                        xlm-clm-ende-1024, xlm-mlm-17-1280, xlm-mlm-100-1280
  --model_type MODEL_TYPE
                        Model type selected in the list: distilbert, albert,
                        camembert, xlm-roberta, bart, roberta, bert, xlnet,
                        flaubert, xlm
  --config_name CONFIG_NAME
                        Pretrained config name or path if not the same as
                        model_name
  --tokenizer_name TOKENIZER_NAME
                        Pretrained tokenizer name or path if not the same as
                        model_name
  --cache_dir CACHE_DIR
                        Where do you want to store the pre-trained models
                        downloaded from s3
  --task_name TASK_NAME
                        The name of the task to train selected in the list:
                        cola, mnli, mnli-mm, mrpc, sst-2, sts-b, qqp, qnli,
                        rte, wnli
  --data_dir DATA_DIR   The input data dir. Should contain the .tsv files (or
                        other data files) for the task.
  --max_seq_length MAX_SEQ_LENGTH
                        The maximum total input sequence length after
                        tokenization. Sequences longer than this will be
                        truncated, sequences shorter will be padded.
  --overwrite_cache     Overwrite the cached training and evaluation sets
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and
                        checkpoints will be written.
  --overwrite_output_dir
                        Overwrite the content of the output directory
  --do_train            Whether to run training.
  --do_eval             Whether to run eval on the dev set.
  --evaluate_during_training
                        Run evaluation during training at each logging step.
  --per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE
                        Batch size per GPU/CPU for training.
  --per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE
                        Batch size per GPU/CPU for evaluation.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before
                        performing a backward/update pass.
  --learning_rate LEARNING_RATE
                        The initial learning rate for Adam.
  --weight_decay WEIGHT_DECAY
                        Weight decay if we apply some.
  --adam_epsilon ADAM_EPSILON
                        Epsilon for Adam optimizer.
  --max_grad_norm MAX_GRAD_NORM
                        Max gradient norm.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform.
  --max_steps MAX_STEPS
                        If > 0: set total number of training steps to perform.
                        Override num_train_epochs.
  --warmup_steps WARMUP_STEPS
                        Linear warmup over warmup_steps.
  --logging_steps LOGGING_STEPS
                        Log every X updates steps.
  --save_steps SAVE_STEPS
                        Save checkpoint every X updates steps.
  --save_total_limit SAVE_TOTAL_LIMIT
                        Limit the total amount of checkpoints, delete the
                        older checkpoints in the output_dir, does not delete
                        by default
  --eval_all_checkpoints
                        Evaluate all checkpoints starting with the same prefix
                        as model_name ending and ending with step number
  --no_cuda             Avoid using CUDA even if it is available
  --seed SEED           random seed for initialization
  --fp16                Whether to use 16-bit (mixed) precision (through
                        NVIDIA apex) instead of 32-bit
  --fp16_opt_level FP16_OPT_LEVEL
                        For fp16: Apex AMP optimization level selected in
                        ['O0', 'O1', 'O2', and 'O3'].See details at
                        https://nvidia.github.io/apex/amp.html
  --local_rank LOCAL_RANK
                        For distributed training: local_rank
  --num_warmup_iters NUM_WARMUP_ITERS
                        Warmup steps for evaluation benchmarking.
  --profiling           Doing profiling on cpu.
  --mkldnn              Use Intel IPEX.
  --jit                 Use jit optimize to do optimization.

```
