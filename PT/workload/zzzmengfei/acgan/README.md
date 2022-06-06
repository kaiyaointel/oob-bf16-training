# install dependency package

1. apply patch

```
cd PT/PyTorch-GAN/implementations/acgan
git apply ../../../workload/acgan/acgan.patch
```


# run real time inference

```bash
python acgan.py --inference \
				--ipex \
				--precision bfloat16 \ # optional
				--jit # optional, jit will decreace the perf

'''
Throughput is: 870.05 images/sec
'''

```
