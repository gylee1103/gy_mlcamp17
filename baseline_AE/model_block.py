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


def generator(x, is_training, num_block, scope_name, reuse, is_gray=True):
  if is_gray:
    image_channel = 1
  else:
    image_channel = 3
  with tf.variable_scope(scope_name, reuse=reuse) as vscope:
    x = tf.layers.conv2d(x, 10, kernel_size=3, strides=1, padding='SAME',
        activation=tf.nn.relu) # H, W
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.layers.conv2d(x, 10, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/2, W/2
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.layers.conv2d(x, 10, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/4, W/4
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.layers.conv2d(x, 10, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/8, W/8
    x = tf.layers.batch_normalization(x, training=is_training)

    #for ridx in range(num_block):
    #  x = residual_block(x, is_training, kernel_size=3)
    x = tf.layers.conv2d_transpose(x, 10, kernel_size=3, strides=2, 
        padding='SAME', activation=tf.nn.relu) # H/4, W/4
    x = tf.layers.batch_normalization(x, training=is_training)

    x = tf.layers.conv2d_transpose(x, 10, kernel_size=3, strides=2, 
        padding='SAME', activation=tf.nn.relu) # H/2, W/2
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.layers.conv2d_transpose(x, 10, kernel_size=3, strides=2, 
        padding='SAME', activation=tf.nn.relu) # H, W
    output = tf.layers.conv2d_transpose(x, image_channel, kernel_size=3, strides=1, 
        padding='SAME', activation=None) # H, W

    return output

def discriminator(x, is_training, scope_name, reuse):
  with tf.variable_scope(scope_name, reuse=reuse) as vscope:
    x = tf.layers.conv2d(x, 32, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/2, W/2
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.layers.conv2d(x, 64, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/4, W/4
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.layers.conv2d(x, 128, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/8, W/8
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.layers.conv2d(x, 256, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/16, W/16
    x = tf.layers.dense(x, 512, activation=tf.nn.relu)
    output = tf.layers.dense(x, 1, activation=tf.nn.sigmoid)

    return output

