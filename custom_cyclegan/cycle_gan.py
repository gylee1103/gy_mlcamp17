
import tensorflow as tf
from model_block import *

def build_model(input_X, input_Y, cycle_lambda=10, is_training=True, learning_rate=0.0002):
  batch_size, target_size, _, target_channel = input_X.get_shape().as_list()

  # X is  Sketch, Y is Pen

  num_block = 4

  Y_from_X = generator_local(input_X, is_training, num_block, "generatorG", reuse=False, gen_sketch=False)
  X_from_Y = generator_local(input_Y, is_training, num_block, "generatorF", reuse=False)

  X_cycled = generator_local(Y_from_X, is_training, num_block, "generatorF", reuse=True)
  Y_cycled = generator_local(X_from_Y, is_training, num_block, "generatorG", reuse=True, gen_sketch=False)

  predictions = {'Y_from_X': Y_from_X, 'X_from_Y': X_from_Y,
      'X_cycled': X_cycled, 'Y_cycled': Y_cycled}

  if is_training:


    real_DX = discriminator(input_X, is_training, "discriminatorDX", reuse=False)
    fake_DX0 = discriminator(X_from_Y, is_training, "discriminatorDX", reuse=True)

    real_DY = discriminator(input_Y, is_training, "discriminatorDY", reuse=False)
    fake_DY0 = discriminator(Y_from_X, is_training, "discriminatorDY", reuse=True)

    loss_real_DX = tf.reduce_mean(tf.squared_difference(real_DX, tf.ones_like(real_DX)))
    loss_fake_DX = tf.reduce_mean(tf.square(fake_DX0))
    loss_DX = (loss_real_DX + loss_fake_DX) / 2

    loss_real_DY = tf.reduce_mean(tf.squared_difference(real_DY, tf.ones_like(real_DY)))
    loss_fake_DY = tf.reduce_mean(tf.square(fake_DY0))# + tf.reduce_mean(tf.square(fake_DY1)) 
    loss_DY = (loss_real_DY + loss_fake_DY) / 2

    cycle_loss_X = tf.reduce_mean(tf.abs(X_cycled - input_X))
    cycle_loss_Y = tf.reduce_mean(tf.abs(Y_cycled - input_Y))
    cycle_loss = cycle_loss_X + cycle_loss_Y

    loss_GAN_F = tf.reduce_mean(tf.squared_difference(fake_DX0, tf.ones_like(fake_DX0)))

    loss_GAN_G = tf.reduce_mean(tf.squared_difference(fake_DY0, tf.ones_like(fake_DY0)))

    loss_F = loss_GAN_F + cycle_lambda * cycle_loss
    loss_G = loss_GAN_G + cycle_lambda * cycle_loss

    losses = {'loss_G': loss_G, 'loss_F': loss_F, 'loss_DX': loss_DX,
        'loss_DY': loss_DY, 'cycle_loss': cycle_loss, 'loss_GAN_G': loss_GAN_G}

    t_vars = tf.trainable_variables()

    G_vars = [var for var in t_vars if "generatorG" in var.name]
    F_vars = [var for var in t_vars if "generatorF" in var.name]
    DX_vars = [var for var in t_vars if "discriminatorDX" in var.name]
    DY_vars = [var for var in t_vars if "discriminatorDY" in var.name]


    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      G_optimizer = optimizer.minimize(loss_G, var_list=G_vars)
      F_optimizer = optimizer.minimize(loss_F, var_list=F_vars)
      DX_optimizer = optimizer.minimize(loss_DX, var_list=DX_vars)
      DY_optimizer = optimizer.minimize(loss_DY, var_list=DY_vars)

    with tf.control_dependencies([G_optimizer, DY_optimizer, F_optimizer, DX_optimizer]):
      train_op = tf.no_op(name='train_op')

  else:
    train_op = None
    losses = None

  return train_op, losses, predictions
