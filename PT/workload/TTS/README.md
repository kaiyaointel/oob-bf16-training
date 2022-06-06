## TTS  
```
cd ../../TTS
```

Apply setup patch for numpy version  
```
cp ../workload/TTS/TTS-setup.patch .
git apply TTS-setup.patch
```

Env setup  
```
python setup.py develop
pip install -r requirements.txt
```

Dataset setup guide refer to: https://github.com/mozilla/TTS/tree/fab74dd5be681d5fb51080515f60e1b20c6b8d40 and https://gist.github.com/erogol/97516ad65b44dbddb8cd694953187c5b

```
wget http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xjf LJSpeech-1.1.tar.bz2
shuf LJSpeech-1.1/metadata.csv > LJSpeech-1.1/metadata_shuf.csv
head -n 12000 LJSpeech-1.1/metadata_shuf.csv > LJSpeech-1.1/metadata_train.csv
tail -n 1100 LJSpeech-1.1/metadata_shuf.csv > LJSpeech-1.1/metadata_val.csv
```
Alternatively if you have access to /home2/pytorch-broad-models/TTS/datasets/ljspeech you can just do
```
cp -r /home2/pytorch-broad-models/TTS/datasets/ljspeech LJSpeech-1.1
shuf LJSpeech-1.1/metadata_original.csv > LJSpeech-1.1/metadata_shuf.csv
head -n 12000 LJSpeech-1.1/metadata_shuf.csv > LJSpeech-1.1/metadata_train.csv
tail -n 1100 LJSpeech-1.1/metadata_shuf.csv > LJSpeech-1.1/metadata_val.csv
```

Copy train and config file and run train
```
cp ../workload/TTS/train_bf16.py .
cp ../workload/TTS/config_bf16.json .
python train_bf16.py --config_path config_bf16.json
```