# Prepare

Capture root directory for path and model_dir
```
root=`pwd`
```

Git clone this repository, and `cd` into directory for remaining commands
```
git clone https://github.com/ltgoslo/simple_elmo.git && cd simple_elmo/simple_elmo
```

Install required packages:
```
pip install tensorflow==1.15.2
pip --no-cache-dir install pandas scikit-learn h5py numpy smart_open>1.8.1
```

Patch
```
cp /root/0001-mlpc_elmo.patch .
git apply 0001-mlpc_elmo.patch
```

Test Data:
```
/home2/tensorflow-broad-product/oob_tf_models/dpg/elmo/data/mrpc_500.tsv.gz
```

Pre-trained-model:
```
/home2/tensorflow-broad-product/oob_tf_models/dpg/elmo/model
```

# Running

To run inference using pretrained model:
```
cp -p simple_elmo/examples/get_elmo_vectors.py .
python3 get_elmo_vectors.py  -i /workspace/test_data/mrpc_500.tsv.gz -e /workspace/model/ --num_iter 500 --num_warmup 10 --batch 1
```

Iterations and warmup can be passed by following (Default: num_iter=500, num_warmup=10):
```
--num_iter 10 --num_warmup 5
```

Default batch size = 128 in code
Throughput is measured in sentences per second
