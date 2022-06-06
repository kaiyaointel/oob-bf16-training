import argparse
import numpy as np
import time
import tensorflow as tf
from models.model_factory import get_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("--model_name", type=str, 
                        help="name of model, can be [resnest50, resnest50_3d, resnest101]")
    parser.add_argument("--model_path", type=str, 
                        help="save path if train, load path if eval")
    parser.add_argument("--train", action='store_true',
                        help="weather training, default is false")
    parser.add_argument("--num_warmup", type=int, default=10,
                        help="numbers of warmup iteration, default is 10")
    parser.add_argument("--num_iter", type=int, default=500,
                        help="numbers of eval iteration, default is 500")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="eval batch size, default is 1")
    args = parser.parse_args()

    # model_names = ['ResNest50','ResNest101','ResNest200','ResNest269']
    # model_names = ['resnest50']
    # model_names = ['resnest50_3d','resnest101_3d']
    # model_names = ['GENet_light','GENet_normal','GENet_large']
    
    # model_names = ['RegNetX400','RegNetX1.6','RegNetY400','RegNetY1.6']
    # input_shape = [224,224,3]

    model_names = ['resnest50', 'resnest50_3d', 'resnest101']
    assert args.model_name in model_names, 'model name not valid!'

    args.num_iter = 10 if args.train else args.num_iter
    input_img = tf.random.normal([args.num_iter, 256, 256, 3])
    label=tf.constant([2,1,3,4,5,6,4,3,5,6] * (args.num_iter//10))

    if args.train:
        print('model_name', args.model_name)
        print('-'*10)
        model = tf.keras.models.load_model('ResNest50/')
        model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy())
        model.summary()
        model.fit(input_img, label, batch_size=args.batch_size, epochs=1)
        model.save(args.model_path)

    else:
        print('model_name', args.model_name)
        print('-'*10)
        model = tf.keras.models.load_model(args.model_path)
        model.summary()
        warmup_input = tf.random.normal([10, 256, 256, 3])
        warmup_label = tf.constant([2,1,3,4,5,6,4,3,5,6])
        # warmup
        model.evaluate(warmup_input, warmup_label)
        tic = time.time()
        model.evaluate(input_img, label, batch_size=args.batch_size)
        total_time = time.time() - tic
        print(" Total time: {}\n Total sample: {}\n Throughput: {:.2f} samples/s".format(
            total_time, args.num_iter, args.num_iter / total_time
        ))