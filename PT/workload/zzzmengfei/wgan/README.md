# install dependency package

1. apply patch

```
cd PT/PyTorch-GAN/implementations/wgan
git apply ../../../workload/wgan/wgan.patch
```


# run real time inference

```bash
python wgan.py --inference \
				--ipex \
				--precision bfloat16 \ # optional
				--jit # optional, jit will decreace the perf

'''
Throughput is: 23403.32 images/sec
'''

```
