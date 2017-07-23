import tensorflow as tf
from model_block import generator, discriminator, add_noise, random_contrast

def build_model(input_X, input_Y, is_training=True, cycle_lambda=10, 
        learning_rate=0.0002):
  batch_size, target_size, _, target_channel = input_X.get_shape().as_list()

  # X is  Sketch, Y is Pen
  Z = tf.random_normal([batch_size, target_size/32, target_size/32, 2])

  if is_training:
      input_X = random_contrast(input_X)

  # do cycle with Y_from_X_sketch,
  # attach Y_from_X to Discriminator
  Y_from_X, Z_from_X = \
      generator(input_X, "generatorG", reuse=False, gen_output2=True)
  X_from_Y = generator(input_Y, "generatorF", reuse=False, z=Z)

  X_cycled = generator(Y_from_X, "generatorF", reuse=True, z=Z_from_X)
  Y_cycled, Z_recon = \
      generator(X_from_Y, "generatorG", reuse=True, gen_output2=True)

  noisy_input_Y = add_noise(input_Y)
  # additional guide
  Y_from_Y, not_used = \
      generator(noisy_input_Y, "generatorG", reuse=True, gen_output2=True)
  #X_from_X = generator(input_X, "generatorF", reuse=True)
  

  predictions = {'Y_from_X': Y_from_X, 'X_from_Y': X_from_Y,
      'X_cycled': X_cycled, 'Y_cycled': Y_cycled, 'noisy_X': input_X,
      'noisy_Y': noisy_input_Y}

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
    #guide_loss_X = tf.reduce_mean(tf.abs(X_from_X - input_X))
    guide_loss_Y = tf.reduce_mean(tf.abs(Y_from_Y - input_Y))
    guide_loss = guide_loss_Y


    # recon Z test
    loss_Z = tf.reduce_mean(tf.abs(Z_recon - Z)) 
    loss_GAN_F = tf.reduce_mean(tf.squared_difference(fake_DX, tf.ones_like(fake_DX)))
    loss_GAN_G = tf.reduce_mean(tf.squared_difference(fake_DY, tf.ones_like(fake_DY)))

    loss_F = loss_GAN_F + cycle_lambda * cycle_loss + loss_Z
    loss_G = loss_GAN_G + cycle_lambda * cycle_loss + loss_Z + cycle_lambda * guide_loss 

    losses = {'loss_G': loss_GAN_G, 'loss_F': loss_GAN_F, 'loss_DX': loss_DX,
        'loss_DY': loss_DY, 'loss_cycle': cycle_loss, 'loss_Z': loss_Z,
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

