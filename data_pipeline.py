import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot  as  plt
import numpy as np

gpus = tf.config.list_physical_devices('GPU')

for gpu in gpus:

    tf.config.experimental.set_memory_growth(gpu, True)

ds_train=tfds.load('fashion_mnist',split='train')
train_iter=ds_train.as_numpy_iterator()

def scale_images(data):

  image=data['image']

  image = tf.cast(image, tf.float32)

  return image/255.0

ds_train=tfds.load('fashion_mnist',split='train')
ds_train=ds_train.map(scale_images)
ds_train=ds_train.cache()
ds_train=ds_train.shuffle(60000)
ds_train=ds_train.batch(128)
ds_train=ds_train.prefetch(64)