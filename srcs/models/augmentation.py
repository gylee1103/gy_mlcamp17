import tensorflow as tf

def random_contrast(input_tensor):
  [bs, h, w, c] = input_tensor.get_shape().as_list()
  random_add = tf.random_normal([bs, 1, 1, 1], mean=0., stddev=0.3)
  random_mul = tf.random_normal([bs, 1, 1, 1], mean=1., stddev=0.3)
  random_mul = tf.clip_by_value(random_mul, 0.2, 2.0)
  output = tf.add(tf.multiply(input_tensor, random_mul), random_add)
  return output
