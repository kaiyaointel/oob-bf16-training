# install dependency package

1. apply patch

```
cd PT/PyTorch-GAN/implementations/began
git apply ../../../workload/began/began.patch
```


# run real time inference

```bash
python began.py --inference \
				--ipex \
				--precision bfloat16 \ # optional
				--jit # optional, jit will decreace the perf

'''
Throughput is: 8250.50 images/sec
'''

```
