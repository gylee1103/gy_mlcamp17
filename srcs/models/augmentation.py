import tensorflow as tf

def random_contrast(input_tensor):
  [bs, h, w, c] = input_tensor.get_shape().as_list()
  random_add = tf.random_normal([bs, 1, 1, 1], mean=0., stddev=0.3)
  random_mul = tf.random_normal([bs, 1, 1, 1], mean=1., stddev=0.3)
  random_mul = tf.clip_by_value(random_mul, 0.2, 2.0)
  output = tf.add(tf.multiply(input_tensor, random_mul), random_add)
  return output

def random_noise(input_tensor):
  # generate random filters
  [bs, h, w, c] = input_tensor.get_shape().as_list()
  input_tensor = tf.transpose(input_tensor, perm=[3, 1, 2, 0])
  random_filter = tf.random_normal([3, 3, bs, 1], mean=1.0, stddev=1)
  output = tf.nn.depthwise_conv2d(input_tensor, filter=random_filter,
      strides=[1, 1, 1, 1], padding="SAME")

  output = tf.transpose(output, perm=[3, 1, 2, 0])
  output = tf.clip_by_value(output, -1, 1)
  return output
