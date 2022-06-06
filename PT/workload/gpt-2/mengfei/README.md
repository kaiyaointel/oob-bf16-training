# GPT-2

# Prepare

## Install dependency packages
```
pip -r requirements.txt
```

## Apply patch
```
cd GPT-2/
git apply ../workload/GPT-2/gpt-2.patch
```

## dataset & model
Will download model automaticly, no need dataset.


# Benchmark
```bash
cd ../../GPT-2/examples
python run_generation.py --no_cuda \
                         --model_type=gpt2 \
                         --model_name_or_path=gpt2 \
                         --num_return_sequences=100 \
                         --prompt="hello world" \
                         --num_warmup_iter 10 \
                         --iter 100 \
                         --mkldnn \
                         --jit (failed now)

###
 time cost 9.847846508026123
 inference Latency: 0.09847846508026123 s
 inference Throughput: 10.154504329296634 samples/s
###
```

## help info

```bash
python run_generation.py -h
usage: run_generation.py [-h] --model_type MODEL_TYPE --model_name_or_path
                         MODEL_NAME_OR_PATH [--prompt PROMPT]
                         [--length LENGTH] [--stop_token STOP_TOKEN]
                         [--temperature TEMPERATURE]
                         [--repetition_penalty REPETITION_PENALTY] [--k K]
                         [--p P] [--padding_text PADDING_TEXT]
                         [--xlm_language XLM_LANGUAGE] [--seed SEED]
                         [--no_cuda]
                         [--num_return_sequences NUM_RETURN_SEQUENCES]
                         [--num_warmup_iter NUM_WARMUP_ITER]
                         [--benchmark_iter BENCHMARK_ITER] [--mkldnn] [--jit]

optional arguments:
  -h, --help            show this help message and exit
  --model_type MODEL_TYPE
                        Model type selected in the list: gpt2, ctrl, openai-
                        gpt, xlnet, transfo-xl, xlm
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to pre-trained model or shortcut name selected in
                        the list: gpt2, ctrl, openai-gpt, xlnet, transfo-xl,
                        xlm
  --prompt PROMPT
  --length LENGTH
  --stop_token STOP_TOKEN
                        Token at which text generation is stopped
  --temperature TEMPERATURE
                        temperature of 1.0 has no effect, lower tend toward
                        greedy sampling
  --repetition_penalty REPETITION_PENALTY
                        primarily useful for CTRL model; in that case, use 1.2
  --k K
  --p P
  --padding_text PADDING_TEXT
                        Padding text for Transfo-XL and XLNet.
  --xlm_language XLM_LANGUAGE
                        Optional language when used with the XLM model.
  --seed SEED           random seed for initialization
  --no_cuda             Avoid using CUDA when available
  --num_return_sequences NUM_RETURN_SEQUENCES
                        The number of samples to generate.
  --num_warmup_iter NUM_WARMUP_ITER
                        The number warmup, default is 50.
  --benchmark_iter BENCHMARK_ITER
                        The number iters of benchmark, default is 500.
  --mkldnn              Use Intel IPEX to optimize.
  --jit                 Use jit optimize to do optimization.

```