# Prepare

Capture root directory for path and model_dir
```
root=`pwd`
```

Git clone this repository, and `cd` into directory for remaining commands
```
git clone https://github.com/jiegzhan/time-series-forecasting-rnn-tensorflow.git && cd time-series-forecasting-rnn-tensorflow
```

Install required packages:
```
pip install tensorflow
pip --no-cache-dir install keras pandas matplotlib
```

Patch
```
cp /root/time_series_LSTM.patch .
git apply time_series_LSTM.patch
```

# Running

To run inference using pretrained model:
```
python3 train_predict.py --train_file ./data/daily-minimum-temperatures-in-me.csv --parameter_file ./training_config.json
```

Iterations and warmup can be passed by following (Default: num_iter=500, num_warmup=10):
```
--num_iter 10 --num_warmup 5
```

Default batch size = 1
Throughput is measured in samples per second
