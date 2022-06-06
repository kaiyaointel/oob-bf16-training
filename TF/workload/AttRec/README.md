# AttRec
    TensorFlow implemenation of AttRec model in paper "Next Item Recommendation with Self-Attention"

> Note: this AttRec neet to run with intel-tensorlow==1.15.2


## run
```bash
python main.py --batch_size=1 --eval=True
```


## help info
```bash
python main.py --help

       USAGE: main.py [flags]
flags:

main.py:
  --batch_size: batch size
    (default: '256')
    (an integer)
  --eval: evaluation the model
    (default: 'True')
  --file_path: training data dir
    (default: 'input/u.data')
  --mode: train or test
    (default: 'train')
  --num_eval_samples: number of total eval samples
    (default: '500')
    (an integer)
  --num_warmup: number of warmup
    (default: '50')
    (an integer)
  --save_path: the whole path to save the model
    (default: 'save_path/model1.ckpt')
  --test_path: testing data dir
    (default: 'input/test.csv')
  --train: train the model
    (default: 'False')
  --train_path: training data dir
    (default: 'input/train.csv')

Try --helpfull to get a list of all flags.

```
