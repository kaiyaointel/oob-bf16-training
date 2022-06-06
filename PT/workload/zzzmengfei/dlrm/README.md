# install dependency package

1. apply patch

```
cd PT/dlrm
git apply ../workload/DLRM/dlrm.patch
```

2. install requirements

```
pip install -r requirements.txt
```

# run real time inference

```bash
./bench/dlrm_s_criteo_kaggle.sh \
								--precision=bfloat16 \ # optional
								--jit # optional, jit will decreace the perf

'''
Throughput is: 6504.813896 its/s
'''

```
