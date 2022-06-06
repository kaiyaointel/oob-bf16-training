import argparse
import time
from keras_segmentation.models.unet import unet

model = unet(n_classes = 51 , input_height = 416, input_width = 608)

#model.train(
#    train_images =  "dataset1/images_prepped_train/",
#    train_annotations = "dataset1/annotations_prepped_train/",
#    checkpoints_path = "chckpoint/unet_1" , epochs=5
#)

# Please Note: Number of iterations is multiples of 100 because the test data has 100 images

parser = argparse.ArgumentParser(description='Tensorflow Unet')
parser.add_argument('-n', '--num_iter', default=5, type=int,
                   help='numbers of inference iteration in multiples of 100 (default: 5)')
parser.add_argument('--num_warmup', default=10, type=int,
                   help='numbers of warmup iteration, default is 10')
parser.add_argument('--checkpoints_path', default='./checkpoint/unet_1', type=str,
                   help='path to checkpoints, default is ./checkpoint/unet_1')
parser.add_argument('--inp', default='./dataset/images_prepped_test/', type=str,
                   help='path to inp, default is ./dataset/images_prepped_test/')

args = parser.parse_args()
num_iter = args.num_iter
num_warmup = args.num_warmup

for i in range (num_warmup):
    out = model.predict_segmentation(
        checkpoints_path = args.checkpoints_path,
        inp = args.inp + "/0016E5_07959.png",
        out_fname = "./out_dir/out.png"
    )

tic = time.time()
for i in range(num_iter):
    out = model.predict_multiple(
        checkpoints_path =  args.checkpoints_path,
        inp_dir = args.inp,
        out_dir = "./out_dir/"
    )
toc = time.time()

print("Throughput: {}".format(num_iter * 100 / (toc - tic)))
#import matplotlib.pyplot as plt
#plt.imshow(out)

# evaluating the model
#print(model.evaluate_segmentation( checkpoints_path = "/workspace/image-segmentation-keras/chckpoint/unet_1", inp_images_dir = "dataset1/images_prepped_test_100/"  , annotations_dir = "dataset1/annotations_prepped_test/" ) )