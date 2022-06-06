# install dependency package

1. apply patch

```
cd PT/PyTorch-GAN/
git apply ../workload/srgan/srgan.patch
cd implementations/srgan
```

# run real time inference

```bash
python srgan.py --evaluate\
	        --ipex \
		--precision bfloat16 \ # optional
                --batch_size 128 \
                --num-iterations 200 \
	        --jit

'''
Throughput is: 8341.21 images/sec
'''

```

# Help Info

```
optional arguments:
  -h, --help            show this help message and exit
  --epoch EPOCH         epoch to start training from
  --n_epochs N_EPOCHS   number of epochs of training
  --dataset_name DATASET_NAME
                        name of the dataset
  --batch_size BATCH_SIZE
                        size of the batches
  --lr LR               adam: learning rate
  --b1 B1               adam: decay of first order momentum of gradient
  --b2 B2               adam: decay of first order momentum of gradient
  --decay_epoch DECAY_EPOCH
                        epoch from which to start lr decay
  --n_cpu N_CPU         number of cpu threads to use during batch generation
  --hr_height HR_HEIGHT
                        high res. image height
  --hr_width HR_WIDTH   high res. image width
  --channels CHANNELS   number of image channels
  --sample_interval SAMPLE_INTERVAL
                        interval between saving image samples
  --checkpoint_interval CHECKPOINT_INTERVAL
                        interval between model checkpoints
  --evaluate            evaluate only
  --num-iterations NUM_ITERATIONS
                        max iterations to run
  --warmup WARMUP       iterations to warmup
  --cuda                Use CUDA
  --ipex                Use IPEX
  --precision PRECISION
                        Precision, "float32" or "bfloat16"
  --jit                 Use jit script model
```
