### gpt-2
Apply patch  
```
cd ../../gpt-2
git apply ../workload/gpt-2/gpt-2-training.patch
```
(PS: the old patch does not modify /src/transformers/modeling_gpt2.py)  
Env setup
```
pip install .
pip install -r ./examples/requirements.txt
```
Set up training dataset
```
cd examples && mkdir data
cd data && mkdir wikitext-2-raw
cd ../..
cp ../workload/gpt-2/wiki.train.raw ./examples/data/wikitext-2-raw
cp ../workload/gpt-2/wiki.test.raw ./examples/data/wikitext-2-raw
```
Set up env var
```
export TRAIN_FILE=./data/wikitext-2-raw/wiki.train.raw
export TEST_FILE=./data/wikitext-2-raw/wiki.test.raw
```
Go to work folder
```
cd examples
```
Run training
```
python run_language_modeling.py \
    --output_dir=output \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --overwrite_output_dir
```