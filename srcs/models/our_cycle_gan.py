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

    output_1 = tf.layers.conv2d_transpose(x, image_channel, kernel_size=3, strides=1, 
        padding='SAME', activation=tf.nn.tanh) # H, W
    if gen_output2:
        output_2 = tf.layers.conv2d_transpose(x, image_channel, kernel_size=3, strides=1, 
            padding='SAME', activation=tf.nn.tanh) # H, W
        return output_1, output_2
    else:
        return output_1

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

def build_model(input_X, input_Y, is_training=True, cycle_lambda=10, 
        learning_rate=0.0002):
  batch_size, target_size, _, target_channel = input_X.get_shape().as_list()

  if is_training:
      input_X = random_contrast(input_X)

  # do cycle with Y_from_X_sketch,
  # attach Y_from_X to Discriminator
  Y_from_X, Y_from_X_extra = \
      generator(input_X, "generatorG", reuse=False, gen_output2=True)
  X_from_Y = generator(input_Y, "generatorF", reuse=False)

  X_cycled = generator(Y_from_X_extra, "generatorF", reuse=True)
  Y_cycled, _ = generator(X_from_Y, "generatorG", reuse=True, gen_output2=True)

  noisy_input_Y = add_noise(input_Y)
  # additional guide
  Y_from_Y, not_used = \
      generator(noisy_input_Y, "generatorG", reuse=True, gen_output2=True)
  X_from_X = generator(input_X, "generatorF", reuse=True)
  

  predictions = {'Y_from_X': Y_from_X, 'X_from_Y': X_from_Y,
      'X_cycled': X_cycled, 'Y_cycled': Y_cycled, 'noisy_X': input_X,
      'noisy_Y': noisy_input_Y, 'extra': Y_from_X_extra}

  if is_training:
    real_DX = discriminator(input_X, "discriminatorDX", reuse=False)
    fake_DX = discriminator(X_from_Y, "discriminatorDX", reuse=True)

    real_DY = discriminator(input_Y, "discriminatorDY", reuse=False)
    fake_DY = discriminator(Y_from_X, "discriminatorDY", reuse=True)

    loss_real_DX = tf.reduce_mean(tf.squared_difference(real_DX, tf.ones_like(real_DX)))
    loss_fake_DX = tf.reduce_mean(tf.square(fake_DX))
    loss_DX = (loss_real_DX + loss_fake_DX) / 2

    loss_real_DY = tf.reduce_mean(tf.squared_difference(real_DY, tf.ones_like(real_DY)))
    loss_fake_DY = tf.reduce_mean(tf.square(fake_DY))
    loss_DY = (loss_real_DY + loss_fake_DY) / 2

    cycle_loss_X = tf.reduce_mean(tf.abs(X_cycled - input_X))
    cycle_loss_Y = tf.reduce_mean(tf.abs(Y_cycled - input_Y))
    cycle_loss = cycle_loss_X + cycle_loss_Y

    # guide loss
    guide_loss_X = tf.reduce_mean(tf.abs(X_from_X - input_X))
    guide_loss_Y = tf.reduce_mean(tf.abs(Y_from_Y - input_Y))
    guide_loss = guide_loss_X + guide_loss_Y


    # recon Z test
    loss_GAN_F = tf.reduce_mean(tf.squared_difference(fake_DX, tf.ones_like(fake_DX)))
    loss_GAN_G = tf.reduce_mean(tf.squared_difference(fake_DY, tf.ones_like(fake_DY)))

    loss_F = loss_GAN_F + cycle_lambda * cycle_loss + cycle_lambda * guide_loss 
    loss_G = loss_GAN_G + cycle_lambda * cycle_loss + cycle_lambda * guide_loss 

    losses = {'loss_G': loss_GAN_G, 'loss_F': loss_GAN_F, 'loss_DX': loss_DX,
        'loss_DY': loss_DY, 'loss_cycle': cycle_loss,
        'loss': loss_G}

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

