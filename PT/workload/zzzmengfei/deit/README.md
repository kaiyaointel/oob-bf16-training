# install dependency package

1. apply patch

```
cd PT/deit
git apply ../workload/deit/deit.patch
```

2. install requirements

```
pip install timm==0.3.2 --no-deps
```


# pre-trained model

mkdir model
ln -s $/home2/pytorch-broad-models/deitb/model/deit_base_patch16_224-b5f2ef4d.pth ./model/deit_base_patch16_224-b5f2ef4d.pth

# run real time inference

```bash
python -u main.py --dummy \
				  --eval \
				  --resume ./model/deit_base_patch16_224-b5f2ef4d.pth
				  --ipex \
				  --precision bfloat16 \ # optional
				  --jit     # optional, jit will decreace the perf

'''
Throughput is: 45.519887 imgs/s
'''
```
