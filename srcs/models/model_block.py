import tensorflow as tf

def random_contrast(input_tensor):
  [bs, h, w, c] = input_tensor.get_shape().as_list()
  random_add = tf.random_normal([bs, 1, 1, 1], mean=0., stddev=0.3)
  random_mul = tf.random_normal([bs, 1, 1, 1], mean=1., stddev=0.3)
  random_mul = tf.clip_by_value(random_mul, 0.2, 2.0)
  output = tf.add(tf.multiply(input_tensor, random_mul), random_add)
  return output

def add_noise(input_tensor):
  # generate random filters
  [bs, h, w, c] = input_tensor.get_shape().as_list()
  input_tensor = tf.transpose(input_tensor, perm=[3, 1, 2, 0])
  random_filter = tf.random_normal([3, 3, bs, 1], mean=1.0, stddev=1)
  output = tf.nn.depthwise_conv2d(input_tensor, filter=random_filter,
      strides=[1, 1, 1, 1], padding="SAME")

  output = tf.transpose(output, perm=[3, 1, 2, 0])
  output = tf.clip_by_value(output, -1, 1)
  return output

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


def generator(x, scope_name, reuse, is_gray=True, gen_output2=False, z=None):
  if is_gray:
    image_channel = 1
  else:
    image_channel = 3
  input = x

  with tf.variable_scope(scope_name, reuse=reuse) as vscope:
    if z is not None:
        z = tf.layers.conv2d_transpose(z, 64, kernel_size=3, strides=2, 
                padding='SAME', activation=tf.nn.relu) # H/16
        z = instance_normalization(z, 'z0_')
        z = tf.layers.conv2d_transpose(z, 64, kernel_size=3, strides=2, 
                padding='SAME', activation=tf.nn.relu) # H/8
        z = instance_normalization(z, 'z1_')

    x = tf.layers.conv2d(x, 64, kernel_size=5, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/2
    x = instance_normalization(x, 0)
    x = residual_block(x, 0, kernel_size=3)

    x = tf.layers.conv2d(x, 64, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/4
    x = instance_normalization(x, 1)
    unet = x
    x = residual_block(x, 1, kernel_size=3)


    x = tf.layers.conv2d(x, 128, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/8
    x = instance_normalization(x, 2)
    # Concat z here
    if z is not None:
        x = tf.concat([x, z], axis=3)
    if gen_output2:
        # Generate z value
        z = tf.layers.conv2d(x, 2, kernel_size=3, strides=4, padding='SAME',
            activation=tf.nn.tanh) # H/32 (8 x 8)
    x = residual_block(x, 2, kernel_size=3)

    x = tf.layers.conv2d_transpose(x, 64, kernel_size=3, strides=2, 
        padding='SAME', activation=tf.nn.relu) # H/4
    x = instance_normalization(x, 3)
    x = residual_block(x, 3, kernel_size=3)

    x = tf.concat([x, unet], axis=3)

    x = tf.layers.conv2d_transpose(x, 64, kernel_size=3, strides=2, 
        padding='SAME', activation=tf.nn.relu) # H/1, W/2
    x = instance_normalization(x, 4)
    x = residual_block(x, 4, kernel_size=3)

    x = tf.layers.conv2d_transpose(x, 32, kernel_size=3, strides=2, 
        padding='SAME', activation=tf.nn.relu) # H, W
    x = instance_normalization(x, 5)
    x = residual_block(x, 5, kernel_size=3)

    output = tf.layers.conv2d_transpose(x, image_channel, kernel_size=3, strides=1, 
        padding='SAME', activation=tf.nn.tanh) # H, W

    if gen_output2:
      return output, z

    return output

def discriminator(x, scope_name, reuse):
  with tf.variable_scope(scope_name, reuse=reuse) as vscope:
    x = tf.layers.conv2d(x, 16, kernel_size=5, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/2
    x = instance_normalization(x, 0)
    x = tf.layers.conv2d(x, 32, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/4
    x = instance_normalization(x, 1)
    x = tf.layers.conv2d(x, 64, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/8, W/8
    x = instance_normalization(x, 2)
    x = tf.layers.conv2d(x, 128, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu) # H/16, W/16
    output = tf.layers.conv2d(x, 1, kernel_size=1, strides=1, padding='SAME',
        activation=tf.nn.sigmoid) # H/16, W/16

    return output
