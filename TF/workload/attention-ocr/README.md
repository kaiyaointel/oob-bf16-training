# Attention_OCR

## Prepare

Git clone this repository, and `cd` into directory for remaining commands
```bash
git clone https://github.com/tensorflow/models.git && cd models/research/attention_ocr/
git reset 32671b --hard
```

Install python packages:
```bash
pip install intel-tensorflow==1.15.2
```

Download the model data:
```bash
cd python/
scp -r shareduser@mlt-ace.sh.intel.com:/home2/tensorflow-broad-product/oob_tf_models/mlp/attention_ocr/ .
```

Patch:
```bash
git apply attention_ocr.patch
```

## Running

run with below code:
```bash
python demo_inference.py --num_iter 500 --num_warmup 50 --checkpoint ./attention_ocr/model.ckpt-399731
```

### help info
```bash
python demo_inference.py --help

flags:

demo_inference.py:
  --eval_batch_size: Num of warmup.
    (default: '1')
    (an integer)
  --image_path_pattern: A file pattern with a placeholder for the image index.
    (default: '')
  --num_iter: Num of total benchmark samples.
    (default: '500')
    (an integer)
  --num_warmup: Num of warmup.
    (default: '50')
    (an integer)

Try --helpfull to get a list of all flags.

```

