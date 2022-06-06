# DCGAN

# prepared

```bash
cd PT/dcgan/dcgan
git apply ../../workload/dcgan/enable_jit.patch
```

# run

```bash
python main.py --inference --dataset fake --jit --batchSize 1 
```
