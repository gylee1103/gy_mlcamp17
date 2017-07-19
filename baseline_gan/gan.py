
import tensorflow as tf
from model_block import *

def build_model(input_X, input_Y, input_Y_noise, is_training=True, learning_rate=0.0002):
  batch_size, target_size, _, target_channel = input_X.get_shape().as_list()

  # X is  Sketch, Y is Pen

  num_block = 4

  Y_from_X = generator(input_X, is_training, num_block, "generatorG", reuse=False)
  Y_from_Y = generator(input_Y_noise, is_training, num_block, "generatorG", reuse=True)

  predictions = {'Y_from_X': Y_from_X, 'Y_from_Y': Y_from_Y}

  if is_training:

    real_DY = discriminator(input_Y, is_training, "discriminatorD", reuse=False)
    fake_DY = discriminator(Y_from_X, is_training, "discriminatorD", reuse=True)

    loss_real_D = tf.reduce_mean(tf.squared_difference(real_DY, tf.ones_like(real_DY)))
    loss_fake_D = tf.reduce_mean(tf.square(fake_DY))
    loss_D = (loss_real_D + loss_fake_D) / 2

    loss_recon = tf.reduce_mean(tf.squared_difference(input_Y, Y_from_Y))
    loss_G_gan = tf.reduce_mean(tf.squared_difference(fake_DY, tf.ones_like(fake_DY)))
    loss_G = loss_G_gan + 10 * loss_recon

    losses = {'loss_G': loss_G, 'loss_D': loss_D}

    t_vars = tf.trainable_variables()

    G_vars = [var for var in t_vars if "generatorG" in var.name]
    D_vars = [var for var in t_vars if "discriminatorD" in var.name]


    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      G_optimizer = optimizer.minimize(loss_G, var_list=G_vars)
      D_optimizer = optimizer.minimize(loss_D, var_list=D_vars)

    with tf.control_dependencies([G_optimizer, D_optimizer]):
      train_op = tf.no_op(name='train_op')

  else:
    train_op = None
    losses = None

  return train_op, losses, predictions

