#install StellarGraph if running on Google Colab
import sys
# if 'google.colab' in sys.modules:
#     %pip install -q stellargraph[demos]==1.3.0b
import stellargraph as sg

# try:
#     sg.utils.validate_notebook_version("1.3.0b")
# except AttributeError:
#     raise ValueError(
#         f"This notebook requires StellarGraph version 1.3.0b, but a different version {sg.__version__} is installed.  Please see <https://github.com/stellargraph/stellargraph/issues/1172>."
#     ) from None
import pandas as pd
import os

import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN

from tensorflow.keras import layers, optimizers, losses, metrics, Model
import tensorflow as tf
from sklearn import preprocessing, model_selection
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import keras
# %matplotlib inline

dataset = sg.datasets.Cora()

G, node_subjects = dataset.load()


# node_subjects.value_counts().to_frame()

train_subjects, test_subjects = model_selection.train_test_split(node_subjects, train_size=140, test_size=None, stratify=node_subjects)
val_subjects, test_subjects = model_selection.train_test_split(test_subjects, train_size=500, test_size=None, stratify=test_subjects)

target_encoding = preprocessing.LabelBinarizer()

train_targets = target_encoding.fit_transform(train_subjects)
val_targets = target_encoding.transform(val_subjects)
test_targets = target_encoding.transform(test_subjects)
generator = FullBatchNodeGenerator(G, method="gcn")

train_gen = generator.flow(train_subjects.index, train_targets)

gcn = GCN(layer_sizes=[16, 16], activations=["relu", "relu"], generator=generator, dropout=0.5)
x_inp, x_out = gcn.in_out_tensors()

predictions = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

# def get_model():
model = Model(inputs=x_inp, outputs=predictions)
model.compile(
    optimizer=optimizers.Adam(lr=0.01),
    loss=losses.categorical_crossentropy,
    metrics=["acc"],
)

# model = get_model()

val_gen = generator.flow(val_subjects.index, val_targets)

from tensorflow.keras.callbacks import EarlyStopping

es_callback = EarlyStopping(monitor="val_acc", patience=50, restore_best_weights=True)

# callbacks = [
#     keras.callbacks.ModelCheckpoint('./models/gcn.h5', save_weights_only=False, save_best_only=True, mode='min'),
#     keras.callbacks.ReduceLROnPlateau(),
# ]

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="my_models",
        save_best_only=True,
        monitor="val_loss",
        verbose=1,
    )
]

history = model.fit(
    train_gen,
    epochs=200,
    validation_data=val_gen,
    verbose=2,
    shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
    # callbacks=[es_callback],
    callbacks=callbacks,
)
# model.save("my_model")

# model.save('/my_model')
tf.keras.models.save_model(model, 'my_model')
# tf.saved_model.simple_save(model,"./models", inputs, outputs)
# tf.saved_model.save(pretrain,path)

test_gen = generator.flow(test_subjects.index, test_targets)

test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))


all_nodes = node_subjects.index
all_gen = generator.flow(all_nodes)
all_predictions = model.predict(all_gen)

node_predictions = target_encoding.inverse_transform(all_predictions.squeeze())
pd.DataFrame({"Predicted": node_predictions, "True": node_subjects})
