import tensorflow as tf 
import tensorflow_addons as tfa 
from tensorflow import keras
from tensorflow.keras import layers

import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output

def downsample(filters, size, apply_instancenorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))
    if apply_instancenorm:
        result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))
    result.add(layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)
  gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

  result = keras.Sequential()
  result.add(layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))
  result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))
  if apply_dropout:
      result.add(layers.Dropout(0.5))
  result.add(layers.ReLU())

  return result

def Generator(OUTPUT_CHANNELS):
  inputs = layers.Input(shape=[None, None, 3])
  
  down_stack = [
                downsample(64, 4, apply_instancenorm=False),
                downsample(128, 4),
                downsample(256, 4),
                downsample(512, 4),
                downsample(512, 4),
                downsample(512, 4),
                downsample(512, 4),
                downsample(512, 4),              
  ]

  up_stack = [
              upsample(512, 4, apply_dropout=True),
              upsample(512, 4, apply_dropout=True),
              upsample(512, 4, apply_dropout=True),
              upsample(512, 4),
              upsample(256, 4), 
              upsample(128, 4),
              upsample(64, 4)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')

  x = inputs
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = layers.Concatenate()([x, skip])

  x = last(x)

  return keras.Model(inputs=inputs, outputs=x)

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)
  gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

  inp = layers.Input(shape=[None, None, 3], name='input_image')

  x = inp

  down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
  down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
  down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

  zero_pad1 = layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
  conv = layers.Conv2D(512, 4, strides=1,
                        kernel_initializer=initializer,
                        use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  norm1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)

  leaky_relu = layers.LeakyReLU()(norm1)

  zero_pad2 = layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = layers.Conv2D(1, 4, strides=1,
                        kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

  return tf.keras.Model(inputs=inp, outputs=last)

class CycleGAN():
  def __init__(self):
    self.lambda_value = 10

    self.generator_g = Generator(OUTPUT_CHANNELS=3)
    self.generator_f = Generator(OUTPUT_CHANNELS=3)

    self.discriminator_x = Discriminator()
    self.discriminator_y = Discriminator()

    self.generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    self.generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    self.discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    self.discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    self.checkpoint = tf.train.Checkpoint(
      generator_g = self.generator_g,
      generator_f = self.generator_f,
      discriminator_x = self.discriminator_x,
      discriminator_y = self.discriminator_y,
      generator_g_optimizer = self.generator_g_optimizer,
      generator_f_optimizer = self.generator_f_optimizer,
      discriminator_x_optimizer = self.discriminator_x,
      discriminator_y_optimizer = self.discriminator_y
    )

    self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

  def discriminator_loss(self, real, generated):
    real_loss = self.loss_obj(tf.ones_like(real), real)
    generated_loss = self.loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5

  def generator_loss(self, generated):
    return self.loss_obj(tf.ones_like(generated), generated)

  def calc_cycle_loss(self, real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return self.lambda_value * loss1

  def identity_loss(self, real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return self.lambda_value * 0.5 * loss

  @tf.function
  def train_step(self, real_x, real_y):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
      # Generator G translates X -> Y
      # Generator F translates Y -> X.
      
      fake_y = self.generator_g(real_x, training=True)
      cycled_x = self.generator_f(fake_y, training=True)

      fake_x = self.generator_f(real_y, training=True)
      cycled_y = self.generator_g(fake_x, training=True)

      # same_x and same_y are used for identity loss.
      same_x = self.generator_f(real_x, training=True)
      same_y = self.generator_g(real_y, training=True)

      disc_real_x = self.discriminator_x(real_x, training=True)
      disc_real_y = self.discriminator_y(real_y, training=True)

      disc_fake_x = self.discriminator_x(fake_x, training=True)
      disc_fake_y = self.discriminator_y(fake_y, training=True)

      # calculate the loss
      gen_g_loss = self.generator_loss(disc_fake_y)
      gen_f_loss = self.generator_loss(disc_fake_x)
      
      total_cycle_loss = self.calc_cycle_loss(real_x, cycled_x) + self.calc_cycle_loss(real_y, cycled_y)
      
      # Total generator loss = adversarial loss + cycle loss
      total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
      total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

      disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
      disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)
    
    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                          self.generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                          self.generator_f.trainable_variables)
    
    discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                              self.discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                              self.discriminator_y.trainable_variables)
    
    # Apply the gradients to the optimizer
    self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                              self.generator_g.trainable_variables))

    self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                              self.generator_f.trainable_variables))
    
    self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                  self.discriminator_x.trainable_variables))
    
    self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                  self.discriminator_y.trainable_variables))
    
    return total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss
  
  def generate_images(self, test_input):
    prediction = self.generator_g(test_input)
    plt.figure(figsize=(12, 12))
    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
      plt.subplot(1, 2, i+1)
      plt.title(title[i])
      # getting the pixel values between [0, 1] to plot it.
      plt.imshow(display_list[i] * 0.5 + 0.5)
      plt.axis('off')
    plt.show()

  def train(self, dataset, EPOCHS, checkpoint_pr):
    for epoch in tqdm(range(EPOCHS)):

      for image_x, image_y in dataset:
        total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss = self.train_step(image_x, image_y)

      clear_output(wait=True)
      # Using a consistent image (sample_horse) so that the progress of the model
      # is clearly visible.
      generate_images(generator_g, image_x[0])

      if (epoch + 1) % 5 == 0:
        save_path = self.checkpoint.save(file_prefix=checkpoint_pr)
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, save_path))

      template = 'Epoch {}, Generator g loss {}, Generator f loss {}, Discriminator g Loss {}, Discriminator f Loss {}'
      print (template.format(epoch, total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss))