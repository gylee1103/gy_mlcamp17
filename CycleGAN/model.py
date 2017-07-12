import numpy as np
import tensorflow as tf
tf_layers = tf.contrib.layers

# NOTE: Assume data is always NHWC order for simplicity.

def Mask2Img(z, mask, n_base, n_stack, n_z, model_id):
    reuse = len([t for t in tf.trainable_variables() if t.name.startswith(model_id)]) > 0

    with tf.variable_scope(model_id, reuse=reuse) as vscope:
        num_output = np.prod([8, 8, n_base])
        x = tf_layers.fully_connected(z, num_output, activation_fn=None)
        x = tf.reshape(x, [-1, 8, 8, n_base])
        x = tf.concat([x, mask], axis=3)

        # Decode - Generate images
        for idx in range(n_stack):
            x = tf_layers.conv2d(x, n_base, 3, 1, activation_fn=tf.nn.elu)
            x = tf_layers.conv2d(x, n_base, 3, 1, activation_fn=tf.nn.elu)
            if idx < n_stack - 1:
                _, h, w, _ = x.get_shape().as_list()
                x = tf.image.resize_nearest_neighbor(x, (h*2, w*2))

        out = tf_layers.conv2d(x, 3, 3, 1, activation_fn=None)

    return out


def Img2Mask(x, n_base, n_stack, n_z, model_id):
    # Generate 8x8 mask
    reuse = len([t for t in tf.trainable_variables() if t.name.startswith(model_id)]) > 0
    BN, H, W, _ = x.get_shape().as_list()

    with tf.variable_scope(model_id, reuse=reuse) as vscope:
        # Encode Image to 8x8xn_base
        for idx in range(n_stack):
            n_current = n_base * (idx + 1)
            x = tf_layers.conv2d(x, n_current, 3, 1, activation_fn=tf.nn.elu)
            if idx < n_stack - 1:
                x = tf_layers.conv2d(x, n_current, 3, 2, activation_fn=tf.nn.elu)
            else:
                x = tf_layers.conv2d(x, n_current, 3, 1, activation_fn=tf.nn.elu)

        mask = tf_layers.conv2d(x, 1, 3, 1, activation_fn=tf.nn.sigmoid)

    return mask

def Discriminator(x, n_base, n_stack, n_z, model_id):
    reuse = len([t for t in tf.trainable_variables() if t.name.startswith(model_id)]) > 0
    BN, H, W, _ = x.get_shape().as_list()

    with tf.variable_scope(model_id, reuse=reuse) as vscope:
        
        # Encode Image to 8x8xn_base
        for idx in range(n_stack):
            n_current = n_base * (idx + 1)
            x = tf_layers.conv2d(x, n_current, 3, 1, activation_fn=tf.nn.elu)
            if idx < n_stack - 1:
                x = tf_layers.conv2d(x, n_current, 3, 2, activation_fn=tf.nn.elu)
            else:
                x = tf_layers.conv2d(x, n_current, 3, 1, activation_fn=tf.nn.elu)

        num_output = np.prod([8, 8, n_base])
        x = tf.reshape(x, [BN, -1])
        x = tf_layers.fully_connected(x, n_z, activation_fn=None)
        
        # Decode
        x = tf_layers.fully_connected(x, num_output, activation_fn=None)
        x = tf.reshape(x, [-1, 8, 8, n_base])

        for idx in range(n_stack):
            x = tf_layers.conv2d(x, n_base, 3, 1, activation_fn=tf.nn.elu)
            x = tf_layers.conv2d(x, n_base, 3, 1, activation_fn=tf.nn.elu)
            if idx < n_stack - 1:
                _, h, w, _ = x.get_shape().as_list()
                x = tf.image.resize_nearest_neighbor(x, (h*2, w*2))

        # Reproduced image
        out = tf_layers.conv2d(x, 3, 3, 1, activation_fn=None)

    return out


