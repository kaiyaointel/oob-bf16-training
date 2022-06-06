# install dependency package

1. apply patch

```
cd PT/PyTorch-GAN/implementations/cgan
git apply ../../../workload/cgan/cgan.patch
```


# run real time inference

```bash
python cgan.py --inference \
				--ipex \
				--precision bfloat16 \ # optional
				--jit # optional, jit will decreace the perf

'''
Throughput is: 10243.90 images/sec
'''

```
