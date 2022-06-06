from tensorflow.data import Dataset
from dlrm import DLRM, get_dlrm
from tensorflow.keras import optimizers
from tqdm import tqdm
import tensorflow as tf
import dataloader
import time
import argparse
import numpy as np

DATA_DIR = './dataset/'
dim_embed = 4
bottom_mlp_size = [8, 4]
top_mlp_size = [128, 64, 1]
total_iter = int(1e5)


def train(save_path='./save_dlrm', data_path='./dataset/', batch_size=1024, epochs=10):

    raw_data = dataloader.load_criteo(data_path)
    # Sample 1000 batches for training
    train_dataset = Dataset.from_tensor_slices({
                    'dense_features': raw_data['X_int_train'][:batch_size*1000],
                    'sparse_features': raw_data['X_cat_train'][:batch_size*1000],
                    'label': raw_data['y_train'][:batch_size*1000]
                }).batch(batch_size).prefetch(1).shuffle(5*batch_size)

    # Sample 100 batches for validation
    val_dataset = Dataset.from_tensor_slices({
                    'dense_features': raw_data['X_int_val'][:batch_size*100],
                    'sparse_features': raw_data['X_cat_val'][:batch_size*100],
                    'label': raw_data['y_val'][:batch_size*100]
             }).batch(batch_size)

    model = get_dlrm(m_spa=dim_embed,
                     ln_emb=raw_data['counts'],
                     ln_bot=bottom_mlp_size,
                     ln_top=top_mlp_size)

    dense_features=raw_data['X_int_train'][:batch_size*100],
    sparse_features= raw_data['X_cat_train'][:batch_size*100],
    label= raw_data['y_train'][:batch_size*100]

    model.compile(optimizer='adam', loss='mse')
    model.fit([dense_features, sparse_features], label, epochs=epochs, batch_size=batch_size)
    model.save(save_path)

def eval(model_path, data_path, batch_size, num_warmup, num_iter):
    dlrm_model = tf.keras.models.load_model(model_path)
    dlrm_model.summary()
    auc = tf.keras.metrics.AUC()

    dense_features = []
    sparse_features = []
    label = []
    raw_data = dataloader.load_criteo(data_path)
    # Sample 100 batches for validation
    val_dataset = Dataset.from_tensor_slices({
                    'dense_features': raw_data['X_int_val'][:batch_size*1000],
                    'sparse_features': raw_data['X_cat_val'][:batch_size*1000],
                    'label': raw_data['y_val'][:batch_size*1000]
             }).batch(batch_size)

    for batch_data in val_dataset:
        dense_features.append(batch_data['dense_features'])
        sparse_features.append(batch_data['sparse_features'])
        label.append(batch_data['label'])

    for index, data in enumerate(zip(dense_features, sparse_features, label)):
        d, s, l = data
        pred = dlrm_model.evaluate([d, s], l, verbose=0)
        # auc.update_state(y_true=l, y_pred=pred)
        if index == num_warmup:
            tic = time.time()
        if index == num_iter:
            break

    toc = time.time()
    throughput = (num_iter - num_warmup) * batch_size / (toc - tic)

    print("Throughput: {:.2f} fps".format(throughput))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="path of model", default='./save_dlrm')
    parser.add_argument("--data_path", help="path of dataset", default='./dataset')
    parser.add_argument("-n", "--num_iter", type=int, default=500,
                        help="numbers of inference iteration, default is 500")
    parser.add_argument("--num_warmup", type=int, default=10,
                        help="numbers of warmup iteration, default is 10")
    parser.add_argument("--disable_optimize", action='store_false',
                        help="use this to disable optimize_for_inference")
    parser.add_argument("-b", "--batch_size", type=int, default=1,
                        help="numbers of inference iteration, default is 1 for realtime inferen.")
    parser.add_argument("--train",  action='store_true',
                        help="Train DLRM with kaggle dataset.")
    
    args = parser.parse_args()

    model_path = args.model_path
    data_path = args.data_path
    batch_size = args.batch_size
    num_warmup = args.num_warmup
    model_path = args.model_path
    num_iter = args.num_iter
    is_train = args.train

    if is_train:
        train(model_path)
    else:
        eval(model_path, data_path, batch_size, num_warmup, num_iter)


