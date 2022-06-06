# Prepare

Git clone this repository, and `cd` into directory for remaining commands
```
git clone https://github.com/openai/gpt-2.git && cd gpt-2
```

Install python packages:
```
pip install tensorflow==1.15.2
pip3 install -r requirements.txt
```

Download the model data
```
python3 download_model.py 124M
```

Patch
```
cp ../gpt-2.patch .
git apply gpt-2.patch
```

# Running

To generate unconditional samples from the small model:
```
python3 src/generate_unconditional_samples.py 
```

we use words/s as the throughput.
