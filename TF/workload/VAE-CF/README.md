# BKC of VAE-CF

## requirements

```bash
pip install -r requirements.txt
```
> Need tensorflow==1.15, tf2.x will meet issue.

## run benchmark

```bash
git clone https://github.com/mkfilipiuk/VAE-CF.git 
cp patch.diff VAE-CF
git apply patch.diff


python run.py --train

```
Then you will get the throughput
