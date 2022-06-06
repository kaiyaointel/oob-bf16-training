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
python main.py --env-name "PongDeterministic-v4" \
			   --ipex \
			   --jit # optional, jit will decreace the perf

'''
Throughput is: 341.910741 segments/s
'''

```
