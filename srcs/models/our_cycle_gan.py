import tensorflow as tf
from models.model_block import generator, discriminator
from models.augmentation import random_contrast

def build_model(input_S, input_P, input_FS_pool=None, input_FP_pool=None,
    is_training=True, cycle_lambda=10, learning_rate=0.0002):
  batch_size, target_size, _, target_channel = input_S.get_shape().as_list()

  if is_training:
      input_S = random_contrast(input_S)

  # do cycle with P_from_S_sketch,
  # attach P_from_S to Discriminator
  P_from_S, P_from_S_extra = \
      generator(input_S, "generatorG", reuse=False, separate_flow=True)
  S_from_P = \
      generator(input_P, "generatorF", reuse=False)

  S_cycled = generator(P_from_S_extra, "generatorF", reuse=True)
  P_cycled, _ = generator(S_from_P, "generatorG", reuse=True, separate_flow=True)

  # additional guide
  P_from_P, _ = \
      generator(input_P, "generatorG", reuse=True, separate_flow=True)
  S_from_S = generator(input_S, "generatorF", reuse=True)
  

  predictions = {'P_from_S': P_from_S, 'S_from_P': S_from_P,
      'S_cycled': S_cycled, 'P_cycled': P_cycled,
      'noisy_S': input_S, 'extra': P_from_S_extra}

  if is_training:
    real_DS = discriminator(input_S, "discriminatorDS", reuse=False)
    fake_DS = discriminator(S_from_P, "discriminatorDS", reuse=True)

    real_DP = discriminator(input_P, "discriminatorDP", reuse=False)
    fake_DP = discriminator(P_from_S, "discriminatorDP", reuse=True)

    loss_real_DS = tf.reduce_mean(tf.squared_difference(real_DS, tf.ones_like(real_DS)))
    loss_fake_DS = tf.reduce_mean(tf.square(fake_DS))
    loss_DS = (loss_real_DS + loss_fake_DS) / 2

    loss_real_DP = tf.reduce_mean(tf.squared_difference(real_DP, tf.ones_like(real_DP)))
    loss_fake_DP = tf.reduce_mean(tf.square(fake_DP))
    loss_DP = (loss_real_DP + loss_fake_DP) / 2

    loss_cycle_S = tf.reduce_mean(tf.abs(S_cycled - input_S))
    loss_cycle_P = tf.reduce_mean(tf.abs(P_cycled - input_P))
    loss_cycle = loss_cycle_S + loss_cycle_P

    # guide loss(push to identity function)
    guide_loss_S = tf.reduce_mean(tf.abs(S_from_S - input_S))
    guide_loss_P = tf.reduce_mean(tf.abs(P_from_P - input_P))
    guide_loss = guide_loss_S + guide_loss_P

    loss_GAN_F = tf.reduce_mean(tf.squared_difference(fake_DS, tf.ones_like(fake_DS)))
    loss_GAN_G = tf.reduce_mean(tf.squared_difference(fake_DP, tf.ones_like(fake_DP)))

    loss_F = loss_GAN_F + cycle_lambda * loss_cycle + cycle_lambda * guide_loss 
    loss_G = loss_GAN_G + cycle_lambda * loss_cycle + cycle_lambda * guide_loss 

    losses = {'loss_G': loss_GAN_G, 'loss_F': loss_GAN_F, 'loss_DS': loss_DS,
        'loss_DP': loss_DP, 'loss_cycle': loss_cycle, 'loss': loss_G,
        'loss_cycle_S': loss_cycle_S, 'loss_cycle_P': loss_cycle_P}


    t_vars = tf.trainable_variables()

    G_vars = [var for var in t_vars if "generatorG" in var.name]
    F_vars = [var for var in t_vars if "generatorF" in var.name]
    DS_vars = [var for var in t_vars if "discriminatorDS" in var.name]
    DP_vars = [var for var in t_vars if "discriminatorDP" in var.name]


    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      G_optimizer = optimizer.minimize(loss_G, var_list=G_vars)
      F_optimizer = optimizer.minimize(loss_F, var_list=F_vars)
      DS_optimizer = optimizer.minimize(loss_DS, var_list=DS_vars)
      DP_optimizer = optimizer.minimize(loss_DP, var_list=DP_vars)

    with tf.control_dependencies([G_optimizer, DP_optimizer, F_optimizer, DS_optimizer]):
      train_op = tf.no_op(name='train_op')

  else:
    train_op = None
    losses = None

  return train_op, losses, predictions

