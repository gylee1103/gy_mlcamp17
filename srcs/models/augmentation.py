import tensorflow as tf

def random_contrast(input_tensor):
  [bs, h, w, c] = input_tensor.get_shape().as_list()
  random_add = tf.random_normal([bs, 1, 1, 1], mean=0., stddev=0.3)
  random_mul = tf.random_normal([bs, 1, 1, 1], mean=1., stddev=0.3)
  random_mul = tf.clip_by_value(random_mul, 0.2, 2.0)
  output = tf.add(tf.multiply(input_tensor, random_mul), random_add)
  return output

def preprocess_pen(input_P): # Original range (-1 ~ 1)
  # Clip value range (-0.9 ~ 0.9)
  input_P = tf.clip_by_value(input_P, -0.99, 0.99)
  return input_P


def get_mask(input_P): # -1 ~ 1, masking -1 regions
  mask_P = - input_P
  mask_P = tf.nn.relu(mask_P) # 0 for background 1 for line
  #mask_P = tf.nn.dilation2d(mask_P, tf.ones_like([5, 5, 1]), strides=[1, 1, 1, 1],
  return mask_P
