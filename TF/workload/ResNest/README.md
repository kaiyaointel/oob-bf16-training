# ResNest
this is for ***ResNest50***, ***ResNest101***, ***ResNest50-3D***

## Prepare

Git clone this repository, and `cd` into directory for remaining commands
```bash
git clone https://github.com/QiaoranC/tf_ResNeSt_RegNet_model.git && cd tf_ResNeSt_RegNet_model
```


### Download the model data
you can run training to get a trained model, will saved at ${model_path}
```bash
python simpel_test.py --model_path ResNest50/ --model_name resnest50 --train
python simpel_test.py --model_path ResNest101/ --model_name resnest101 --train
python simpel_test.py --model_path ResNest50-3d/ --model_name resnest50_3d --train
```
or you can find trained model here: `ace:/home2/tensorflow-broad-product/oob_tf_models/oob/ResNest/`

Patch
```bash
git apply resnest.patch
```

## Running

To generate unconditional samples from the small model:
```bash
python simpel_test.py --model_path ResNest50/ --model_name resnest50 
python simpel_test.py --model_path ResNest101/ --model_name resnest101
python simpel_test.py --model_path ResNest50-3d/ --model_name resnest50_3d
```

### help info

```bash
python simpel_test.py -h
usage: simpel_test.py [-h] [--model_name MODEL_NAME] [--model_path MODEL_PATH]
                    [--train] [--num_warmup NUM_WARMUP] [--num_iter NUM_ITER]
                    [--batch_size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        name of model, can be [resnest50, resnest50_3d,
                        resnest101]
  --model_path MODEL_PATH
                        save path if train, load path if eval
  --train               weather training, default is false
  --num_warmup NUM_WARMUP
                        numbers of warmup iteration, default is 10
  --num_iter NUM_ITER   numbers of eval iteration, default is 500
  --batch_size BATCH_SIZE
                        eval batch size, default is 1


```

