
import tensorflow as tf
from model_block import *

def build_model(input_X, input_Y, cycle_lambda=10, is_training=True, learning_rate=0.0002):
  batch_size, target_size, _, target_channel = input_X.get_shape().as_list()

  num_block = 4


  # AutoEncoded using Pen Data(Y)
  Y_after_AE = generator(input_Y, is_training, num_block, "AE", reuse=False)

  # AutoEncoded using Sketch Data(X) - but not used for train
  X_after_AE = generator(input_X, is_training, num_block, "AE", reuse=True)

  predictions = {'Y_after_AE': Y_after_AE, 'X_after_AE': X_after_AE}

  if is_training:

    loss_AE_Y = tf.reduce_mean(tf.abs(input_Y - Y_after_AE))
    loss_AE_X = tf.reduce_mean(tf.abs(input_X - X_after_AE))

    losses = {'loss_AE_Y': loss_AE_Y, 'loss_AE_X':loss_AE_X}

    t_vars = tf.trainable_variables()

    AE_vars = [var for var in t_vars if "AE" in var.name]


    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    AE_optimizer = optimizer.minimize(loss_AE_Y, var_list=AE_vars)

    with tf.control_dependencies(update_ops + [AE_optimizer]):
      train_op = tf.no_op(name='train_op')

  else:
    train_op = None
    losses = None

  return train_op, losses, predictions

