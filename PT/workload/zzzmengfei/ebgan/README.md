# install dependency package

1. apply patch

```
cd PT/PyTorch-GAN/implementations/ebgan
git apply ../../../workload/ebgan/ebgan.patch
```


# run real time inference

```bash
python ebgan.py --inference \
				--ipex \
				--precision bfloat16 \ # optional
				--jit # optional, jit will decreace the perf

'''
Throughput is: 9611.61 images/sec
'''

```
