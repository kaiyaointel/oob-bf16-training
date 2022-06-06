# Prepare

Capture root directory for path and model_dir
```
root=`pwd`
```

Git clone this repository, and `cd` into directory for remaining commands
```
git clone https://github.com/bfs18/nsynth_wavenet/tree/master/wavenet.git && cd wavenet
```

Install required packages:
```
pip install tensorflow==1.15.2
pip3 --no-cache-dir install --upgrade -r /root/requirements.txt
apt-get install -y libsndfile1
```

Patch
```
cp /root/parallel_wavenet.patch .
git apply parallel_wavenet.patch
```

Pre trained model for parallel wavenet with contrastive loss:
```
/home2/tensorflow-broad-product/oob_tf_models/dpg/Parallel_WaveNet
```

# Running

To run inference using pretrained model: parallel wavenet model with contrastive loss
```
python3 eval_parallel_wavenet.py --ckpt_dir /home2/tensorflow-broad-product/oob_tf_models/dpg/Parallel_WaveNet --source_path tests/test_data/ --save_path tests/pred_data
```
Iterations and warmup can be passed by adding following to the above run command:
```
--num_iter 10 --num_warmup 5
```

Throughput is measured in samples per second