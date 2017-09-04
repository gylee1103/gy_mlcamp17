import tensorflow as tf

def instance_normalization(x, index):
  with tf.variable_scope("instance_norm"):
    depth = x.get_shape()[3]
    scale = tf.get_variable("scale" + str(index), [depth],
        initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02, dtype=tf.float32))
    offset = tf.get_variable("offset" + str(index), [depth], 
        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02, dtype=tf.float32))
    mean, variance = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (x - mean) * inv
    return scale*normalized + offset

def residual_block(x, index, kernel_size=3, activation=tf.nn.relu):
  x = tf.layers.conv2d(x, x.get_shape()[3], kernel_size=kernel_size, strides=1,
      padding='SAME', activation=activation)
  x = instance_normalization(x, "rd0_" + str(index))

  r = tf.layers.conv2d(x, x.get_shape()[3], kernel_size=kernel_size, strides=1,
      padding='SAME', activation=None)
  r = instance_normalization(r, "rd1_" + str(index))

  x = x + r
  return x

def generator(x, scope_name, reuse, is_gray=True, separate_flow=False):
  if is_gray:
    image_channel = 1
  else:
    image_channel = 3

  with tf.variable_scope(scope_name, reuse=reuse) as vscope:
    x = tf.layers.conv2d(x, 64, kernel_size=7, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/2
    x = instance_normalization(x, 0)
    #x = residual_block(x, 0, kernel_size=3)

    x = tf.layers.conv2d(x, 128, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/4
    x = instance_normalization(x, 1)
    #unet = x
    #x = residual_block(x, 1, kernel_size=3)

    x = tf.layers.conv2d(x, 128, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/8
    x = instance_normalization(x, 2)
    #x = residual_block(x, 2, kernel_size=3)

    x = tf.layers.conv2d_transpose(x, 64, kernel_size=3, strides=2, 
        padding='SAME', activation=tf.nn.relu) # H/4
    x = instance_normalization(x, 3)
    #x = residual_block(x, 3, kernel_size=3)

    # Unet Concat
    #x = tf.concat([x, unet], axis=3)

    # Residual
    for i in range(5):
      x = residual_block(x, i, kernel_size=3)

    x = tf.layers.conv2d_transpose(x, 64, kernel_size=3, strides=2, 
        padding='SAME', activation=tf.nn.relu) # H/2, W/2
    x = instance_normalization(x, 4)
    #x = residual_block(x, 4, kernel_size=3)

    x = tf.layers.conv2d_transpose(x, 64, kernel_size=3, strides=2, 
        padding='SAME', activation=tf.nn.relu) # H, W
    x = instance_normalization(x, 5)
    #x = residual_block(x, 5, kernel_size=3)

    output = tf.layers.conv2d_transpose(x, image_channel, kernel_size=7, strides=1, 
        padding='SAME', activation=tf.nn.tanh) # H, W

    if separate_flow: # Our modified cyclegan
      extra_output = tf.layers.conv2d_transpose(x, image_channel, kernel_size=7, strides=1, 
        padding='SAME', activation=tf.nn.tanh) # H, W
      return output, extra_output
    else:
      return output


def discriminator(x, scope_name, reuse):
  with tf.variable_scope(scope_name, reuse=reuse) as vscope:
    x = tf.layers.conv2d(x, 16, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/2
    x = instance_normalization(x, 0)
    x = tf.layers.conv2d(x, 32, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/4
    x = instance_normalization(x, 1)
    x = tf.layers.conv2d(x, 64, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/8, W/8
    x = instance_normalization(x, 2)
    output = tf.layers.conv2d(x, 1, kernel_size=1, strides=1, padding='SAME',
        activation=tf.nn.sigmoid)

    return output
