# dcgan
## CUDA
```
pip uninstall torch
pip3 install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html -U
cd ../../dcgan/dcgan
git apply ../../workload/dcgan/dcgan-training-cuda.patch
python main.py --dataset fake --batchSize 64 --bf16Train --cuda
```
## Xeon
```
cd ../../dcgan/dcgan
git apply ../../workload/dcgan/dcgan-training.patch
python main.py --dataset fake --batchSize 64 --bf16Train
```