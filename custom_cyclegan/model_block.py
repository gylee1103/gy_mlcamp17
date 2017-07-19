import tensorflow as tf

def residual_block(x, is_training, kernel_size=3, activation=tf.nn.relu):
  x = tf.layers.conv2d(x, x.get_shape()[3], kernel_size=kernel_size, strides=1,
      padding='SAME', activation=activation)
  x = tf.layers.batch_normalization(x, training=is_training)

  r = tf.layers.conv2d(x, x.get_shape()[3], kernel_size=kernel_size, strides=1,
      padding='SAME', activation=None)
  r = tf.layers.batch_normalization(r, training=is_training)
  x = x + r
  return x


def generator(x, is_training, num_block, scope_name, reuse, is_gray=True, gen_sketch=True):
  if is_gray:
    image_channel = 1
  else:
    image_channel = 3
  input = x

  with tf.variable_scope(scope_name, reuse=reuse) as vscope:
    x = tf.layers.conv2d(x, 64, kernel_size=3, strides=1, padding='SAME',
        activation=tf.nn.relu) # H, W
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.layers.conv2d(x, 64, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/2, W/2
    x = tf.layers.batch_normalization(x, training=is_training)
    skip1 = x

    x = tf.layers.conv2d(x, 128, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/4, W/4
    x = tf.layers.batch_normalization(x, training=is_training)

    for ridx in range(num_block):
      x = residual_block(x, is_training, kernel_size=3)

    x = tf.layers.conv2d_transpose(x, 128, kernel_size=3, strides=2, 
        padding='SAME', activation=tf.nn.relu) # H/2, W/2
    x = tf.layers.batch_normalization(x, training=is_training)

    x = tf.concat([x, skip1], 3)

    x = tf.layers.conv2d_transpose(x, 64, kernel_size=3, strides=2, 
        padding='SAME', activation=tf.nn.relu) # H, W
    x = tf.layers.batch_normalization(x, training=is_training)

    output = tf.layers.conv2d_transpose(x, image_channel, kernel_size=3, strides=1, 
        padding='SAME', activation=tf.nn.tanh) # H, W


    return output

def discriminator(x, is_training, scope_name, reuse):
  with tf.variable_scope(scope_name, reuse=reuse) as vscope:
    x = tf.layers.conv2d(x, 32, kernel_size=8, strides=8, padding='SAME',
        activation=tf.nn.relu) # H/2, W/2
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.layers.conv2d(x, 64, kernel_size=4, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/4, W/4
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.layers.conv2d(x, 64, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/8, W/8
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.layers.conv2d(x, 32, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/16, W/16
    output = tf.layers.conv2d(x, 1, kernel_size=1, strides=1, padding='SAME',
        activation=tf.nn.sigmoid) # H/16, W/16

    return output

