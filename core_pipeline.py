import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import glob

from data_loader import createDataset as createDataset
from data_visualization import Visualization as Visualization

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import (
    Input, 
    Flatten, 
    GlobalAveragePooling2D, 
    Dropout, 
    Dense)

def set_seed(SEED):
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    tf.random.set_seed(SEED)
    tf.keras.backend.clear_session()
    print('SEED SET!')


class DefaultConfig:
    seed: int = 42
    img_size: int = 224
    labels: list = ['cat', 'dog']
    batch_size: int = 16

set_seed(DefaultConfig.seed)


paths = glob.glob("./kagglecatsanddogs_3367a/**/*.jpg", recursive=True)
np.random.shuffle(paths)

train_paths, test_paths = train_test_split(
    paths, test_size=0.25, random_state=DefaultConfig.seed)

len(train_paths)
len(test_paths)

createdataset = createDataset(train_paths,
            DefaultConfig.batch_size,
            DefaultConfig.img_size,
            training=False)

data = createdataset.load_data()

visualizer = Visualization()

for batch in data.take(1):
    visualizer.visualize(batch, False)

baseModel = MobileNetV2(
 include_top = False,
 weights='imagenet',
)

baseModel.trainable = False

in_ = Input(shape=(
    DefaultConfig.img_size, 
    DefaultConfig.img_size, 
    3))
base_ = baseModel(in_)
global_ = GlobalAveragePooling2D()(base_)
flatten = Flatten()(global_)
dense_1 = Dense(64, activation='relu')(flatten)
drop_1 = Dropout(0.3)(dense_1)
dense_2 = Dense(32, activation='relu')(drop_1)
drop_2 = Dropout(0.2)(dense_2)
dense_3 = Dense(10, activation='relu')(drop_2)
out_ = Dense(2)(dense_3)

model = tf.keras.Model(inputs=[in_], outputs=[out_])

model.summary()


# Custom model training
optimizer = tf.keras.optimizers.SGD(
    learning_rate=1e-3
    )
loss_fn = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True
    )

train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

import time 
epochs = 100

for epoch in range(epochs):
  start_time = time.time()
  print(f"Start of epoch {epoch}")
  for step, (x_batch_train, y_batch_train) in enumerate(data):
    with tf.GradientTape() as tape:
      logits = model(x_batch_train, training=True)
      loss_value = loss_fn(y_batch_train, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y_batch_train, logits)
    if step % 500 == 0: 
      print("Training loss at step %d : %.4f" 
      %(step, float(loss_value)))
  train_acc = train_acc_metric.result()
  print("Training acc over epoch %d: %f"
   %(epoch, float(train_acc)))
  train_acc_metric.reset_states()
  for x_batch_val, y_batch_val in data:
    val_logits = model(x_batch_val, training=False)
    val_acc_metric.update_state(y_batch_val, val_logits)
  val_acc = val_acc_metric.result()
  val_acc_metric.reset_states()
  print(f"Validation acc: {val_acc}")
  print(f"Time taken: {time.time() - start_time}")