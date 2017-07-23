import tensorflow as tf
from model_block import generator, discriminator, add_noise

def build_model(input_X, input_Y, is_training=True, learning_rate=0.0002):
  batch_size, target_size, _, target_channel = input_X.get_shape().as_list()

  noisy_input_Y = add_noise(input_Y)

  # AutoEncoded using Pen Data(Y)
  Y_after_FCN = generator(noisy_input_Y, "FCN", reuse=False)

  # AutoEncoded using Sketch Data(X) - but not used for train
  X_after_FCN = generator(input_X, "FCN", reuse=True)

  predictions = {'Y_from_X': X_after_FCN, 'Y_after_FCN': Y_after_FCN}

  if is_training:

    loss_FCN_Y = tf.reduce_mean(tf.abs(input_Y - Y_after_FCN))
    loss_FCN_X = tf.reduce_mean(tf.abs(input_X - X_after_FCN))

    losses = {'loss': loss_FCN_Y, 'loss_FCN_X':loss_FCN_X}

    t_vars = tf.trainable_variables()

    FCN_vars = [var for var in t_vars if "FCN" in var.name]

    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    FCN_optimizer = optimizer.minimize(loss_FCN_Y, var_list=FCN_vars)

    with tf.control_dependencies(update_ops + [FCN_optimizer]):
      train_op = tf.no_op(name='train_op')

  else:
    train_op = None
    losses = None

  return train_op, losses, predictions

