```
cd ../../wide_deep
pip install -e .
cd pytorch_widedeep
ln -s /home2/pytorch-broad-models/widedeep/data .
ln -s /home2/pytorch-broad-models/widedeep/model .
cp ../workload/wide_deep/inference_arch1_binary.py .
git apply ../workload/wide_deep/wide_deep_training.patch
(git apply ../workload/wide_deep/wide_deep_training_cuda.patch)
python -u inference_arch1_binary.py --inf --batch_size 1 (this command is for train, --eval means inf only)
```