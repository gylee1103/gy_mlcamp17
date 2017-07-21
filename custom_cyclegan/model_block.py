import tensorflow as tf

def instance_normalization(x, index):
  with tf.variable_scope("instance_norm"):
    depth = x.get_shape()[3]
    scale = tf.get_variable("scale" + str(index), [depth], initializer=tf.random_normal_initializer(
      mean=1.0, stddev=0.02, dtype=tf.float32))
    offset = tf.get_variable("offset" + str(index), [depth], initializer=tf.random_normal_initializer(
      mean=0.0, stddev=0.02, dtype=tf.float32))
    mean, variance = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (x - mean) * inv
    return scale*normalized + offset


def residual_block(x, index, is_training, kernel_size=3, activation=tf.nn.relu):
  x = tf.layers.conv2d(x, x.get_shape()[3], kernel_size=kernel_size, strides=1,
      padding='SAME', activation=activation)
  #x = tf.layers.batch_normalization(x, training=is_training)
  x = instance_normalization(x, "rd0_" + str(index))

  r = tf.layers.conv2d(x, x.get_shape()[3], kernel_size=kernel_size, strides=1,
      padding='SAME', activation=None)
  #r = tf.layers.batch_normalization(r, training=is_training)
  r = instance_normalization(r, "rd1_" + str(index))

  x = x + r
  return x


def generator(x, is_training, num_block, scope_name, reuse, is_gray=True, gen_output2=False):
  if is_gray:
    image_channel = 1
  else:
    image_channel = 3
  input = x

  with tf.variable_scope(scope_name, reuse=reuse) as vscope:
    x = tf.layers.conv2d(x, 64, kernel_size=3, strides=1, padding='SAME',
        activation=tf.nn.relu) # H, W
    #x = tf.layers.batch_normalization(x, training=is_training)
    x = instance_normalization(x, 0)

    x = tf.layers.conv2d(x, 64, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/2, W/2
    #x = tf.layers.batch_normalization(x, training=is_training)
    x = instance_normalization(x, 1)

    x = tf.layers.conv2d(x, 64, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/4, W/4
    #x = tf.layers.batch_normalization(x, training=is_training)
    x = instance_normalization(x, 2)

    for ridx in range(num_block):
      x = residual_block(x, ridx, is_training, kernel_size=3)

    x = tf.layers.conv2d_transpose(x, 128, kernel_size=3, strides=2, 
        padding='SAME', activation=tf.nn.relu) # H/2, W/2
    #x = tf.layers.batch_normalization(x, training=is_training)
    x = instance_normalization(x, 3)

    x = tf.layers.conv2d_transpose(x, 32, kernel_size=3, strides=2, 
        padding='SAME', activation=tf.nn.relu) # H, W
    #x = tf.layers.batch_normalization(x, training=is_training)
    x = instance_normalization(x, 4)

    output = tf.layers.conv2d_transpose(x, image_channel, kernel_size=3, strides=1, 
        padding='SAME', activation=tf.nn.tanh) # H, W

    if gen_output2:
      output2 = tf.layers.conv2d_transpose(x, image_channel, kernel_size=3, strides=1, 
          padding='SAME', activation=tf.nn.tanh) # H, W
      return output, output2


    return output

def discriminator(x, is_training, scope_name, reuse):
  with tf.variable_scope(scope_name, reuse=reuse) as vscope:
    x = tf.layers.conv2d(x, 32, kernel_size=8, strides=8, padding='SAME',
        activation=tf.nn.relu) # H/2, W/2
    #x = tf.layers.batch_normalization(x, training=is_training)
    x = instance_normalization(x, 0)
    x = tf.layers.conv2d(x, 64, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/4, W/4
    #x = tf.layers.batch_normalization(x, training=is_training)
    x = instance_normalization(x, 1)
    x = tf.layers.conv2d(x, 64, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/8, W/8
    #x = tf.layers.batch_normalization(x, training=is_training)
    x = instance_normalization(x, 2)
    x = tf.layers.conv2d(x, 32, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/16, W/16
    output = tf.layers.conv2d(x, 1, kernel_size=1, strides=1, padding='SAME',
        activation=tf.nn.sigmoid) # H/16, W/16

    return output

