# install dependency package

1. apply patch

```
cd PT/transformer-xl
git apply ../workload/transformer-xl/transformer-xl.patch
```


# pre-trained model && dataset

```
ln -s /home2/pytorch-broad-models/transfo-xl/model/model.pt ./pytorch/model/model.pt

./getdata.sh 
```

# run real time inference

```bash
cd pytorch
bash run_wt103_base.sh eval \
					  --work_dir ./model/ \
					  --ipex \
					  --precision bfloat16 \ # optional

'''
Throughput is: 1.59 segments/s
'''

```
