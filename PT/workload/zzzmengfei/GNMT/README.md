# install dependency package

1. apply patch

```
cd PT/maskrcnn
git apply ../workload/GNMT/gnmt.patch
```

2. install requirements

```
pip install mlperf-compliance==0.0.10
```

# run real time inference

```bash
cd rnn_translator/pytorch
python train.py --inference \
				--ipex \
				--dataset-dir /home2/pytorch-broad-models/GNMT/dataset/data \
				--precision bfloat16 \ # optional
				--jit # optional, jit will decreace the perf

'''
Inference: 960 Tok/s
'''

```
