import tensorflow as tf 
import tensorflow_addons as tfa 
from tensorflow import keras
from tensorflow.keras import layers

import re
import numpy as np
import matplotlib.pyplot as plt 


monet_path = 'gs://kds-00b8801b5282f824427cc2a2ed904cc6fcd501afbdf21b509c7f4126/monet_tfrec/'
photo_path = 'gs://kds-00b8801b5282f824427cc2a2ed904cc6fcd501afbdf21b509c7f4126/photo_tfrec/'

monet_files = tf.io.gfile.glob(monet_path + '*.tfrec')
photo_files = tf.io.gfile.glob(photo_path + '*.tfrec')
num_parallel = tf.data.experimental.AUTOTUNE
image_size = [256, 256, 3]

def decode_image(image, image_size):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.cast(image, tf.float32) / 127.5 - 1
  image = tf.reshape(image, image_size)
  return image

def read_tfrecord(example):
  tfrecord_format = {
    'image_name': tf.io.FixedLenFeature([], tf.string),
    'image': tf.io.FixedLenFeature([], tf.string), 
    'target': tf.io.FixedLenFeature([], tf.string)
  }

  example = tf.io.parse_single_example(example, tfrecord_format)
  image = decode_image(example['image'], image_size)
  return image

def load_dataset(filenames, ordered=False):
  ignore_order = tf.data.Options()
  if not ordered:
    ignore_order.experimental_deterministic = False
    
  dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=num_parallel)
  dataset = dataset.with_options(ignore_order)
  dataset = dataset.map(read_tfrecord, num_parallel_calls=num_parallel)
  return dataset

def get_dataset(filenames, augment=None, repeat=True, shuffle=True, batch_size=1):
  dataset = load_dataset(filenames)
  
  if augment:
    dataset = dataset.map(augment, num_parallel_calls=num_parallel)
  if repeat:
    dataset = dataset.repeat()
  if shuffle:
    dataset = dataset.shuffle(2048)

  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.cache()
  dataset = dataset.prefetch(num_parallel)

  return dataset

def get_monet_dataset(augment=None, repeat=True, shuffle=True, batch_size=1):
  return get_dataset(monet_files, augment, repeat, shuffle, batch_size)

def get_photo_dataset(augment=None, repeat=True, shuffle=True, batch_size=1):
  return get_dataset(photo_files, augment, repeat, shuffle, batch_size)

def get_length():
  n_monet = np.sum([int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in monet_files])
  n_photo = np.sum([int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in photo_files])
  return max(n_monet, n_photo)

def info():
  n_monet = np.sum([int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in monet_files])
  n_photo = np.sum([int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in photo_files])
  
  print(f'Dataset get from https://www.kaggle.com/c/gan-getting-started')
  print(f'Monet TFRecord Files: {len(monet_files)}')
  print(f'Photo TFRecord Files: {len(photo_files)}')
  print(f'Monet image file: {n_monet}')
  print(f'Photo image file: {n_photo}')