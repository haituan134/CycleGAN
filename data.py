import tensorflow as tf 
import tensorflow_addons as tfa 
from tensorflow import keras
from tensorflow.keras import layers

import re
import numpy as np
import matplotlib.pyplot as plt 

class Data():
  def __init__(self, num_parallel=tf.data.experimental.AUTOTUNE):
    super().__init__()
    monet_path = 'gs://kds-00b8801b5282f824427cc2a2ed904cc6fcd501afbdf21b509c7f4126/monet_tfrec/'
    photo_path = 'gs://kds-00b8801b5282f824427cc2a2ed904cc6fcd501afbdf21b509c7f4126/photo_tfrec/'

    self.monet_files = tf.io.gfile.glob(monet_path + '*.tfrec')
    self.photo_files = tf.io.gfile.glob(photo_path + '*.tfrec')
    self.num_parallel = num_parallel
    self.image_size = [256, 256, 3]

  def decode_image(self, image, image_size):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 127.5 - 1
    image = tf.reshape(image, image_size)
    return image

  def read_tfrecord(self, example):
    tfrecord_format = {
      'image_name': tf.io.FixedLenFeature([], tf.string),
      'image': tf.io.FixedLenFeature([], tf.string), 
      'target': tf.io.FixedLenFeature([], tf.string)
    }

    example = tf.io.parse_single_example(example, tfrecord_format)
    image = self.decode_image(example['image'], self.image_size)
    return image

  def load_dataset(self, filenames, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
      ignore_order.experimental_deterministic = False
      
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=self.num_parallel)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(self.read_tfrecord, num_parallel_calls=self.num_parallel)
    return dataset

  def get_dataset(self, filenames, augment=None, repeat=True, shuffle=True, batch_size=1):
    dataset = self.load_dataset(filenames)
    
    if augment:
      dataset = dataset.map(augment, num_parallel_calls=self.num_parallel)
    if repeat:
      dataset = dataset.repeat()
    if shuffle:
      dataset = dataset.shuffle(2048)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.cache()
    dataset = dataset.prefetch(self.num_parallel)

    return dataset

  def get_monet_dataset(self, augment=False, repeat=True, shuffle=True, batch_size=1):
    return self.get_dataset(self.monet_files)

  def get_photo_dataset(self, augment=False, repeat=True, shuffle=True, batch_size=1):
    return self.get_dataset(self.photo_files)
  
  def info(self):
    n_monet = np.sum([int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in self.monet_files])
    n_photo = np.sum([int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in self.photo_files])
    
    print(f'Monet TFRecord Files: {len(self.monet_files)}')
    print(f'Photo TFRecord Files: {len(self.photo_files)}')
    print(f'Monet image file: {n_monet}')
    print(f'Photo image file: {n_photo}')