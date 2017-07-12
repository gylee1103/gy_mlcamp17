from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def AutoEncoder(x, n_base, n_stack, scope_name, reuse):
    """
    AutoEncoder for given x.
    Args:
      x : input tensor.
      n_base : int, the base number of channels of feature maps
      n_stack: int, the number of layers
      scope_name: string, the name of the current variable scope
    Returns:
      Reconstructed x, 
      Feature map list of encoder,
      Feature map list of decoder.
    """
    with tf.variable_scope(scope_name, reuse=reuse) as vscope:
        # Encode Image
        encoder_feature_list = []
        for idx in range(n_stack):
            n_current = n_base * (idx + 1)
            x = tf.layers.conv2d(x, n_current, kernel_size=3, strides=1,
                    padding='SAME', activation=tf.nn.elu)
            x = tf.layers.conv2d(x, n_current, kernel_size=3, strides=2,
                    padding='SAME', activation=tf.nn.elu)
            encoder_feature_list.append(x)
        # Decode Image
        decoder_feature_list = []
        for idx in range(n_stack):
            n_current = n_base * (n_stack - 1 - idx)
            if (idx == n_stack - 1):
                n_current = n_base
            x = tf.layers.conv2d(x, n_current, kernel_size=3, strides=1,
                    padding='SAME', activation=tf.nn.elu)
            x = tf.layers.conv2d_transpose(x, n_current, kernel_size=3,
                    strides=2, padding='SAME', activation=tf.nn.elu)
            decoder_feature_list.append(x)

        out = tf.layers.conv2d(x, 1, kernel_size=1, strides=1, activation=None)

    return out, encoder_feature_list, decoder_feature_list


def build_model(features, mode, params=None, config=None):
    n_base = params['n_base']
    n_stack = params['n_stack']
    scope_name = "autoencoder" 
    learning_rate = params['learning_rate']

    if (mode == tf.estimator.ModeKeys.TRAIN or
        mode == tf.estimator.ModeKeys.EVAL):
        img_sketch = features['img_sketch']
        img_pen = features['img_pen']

        recon_pen, enc_fmap_pen, enc_fmap_pen = AutoEncoder(
                img_pen, n_base, n_stack, scope_name, reuse=False)
        recon_sketch, enc_fmap_sketch, dec_fmap_sketch = AutoEncoder(
                img_sketch, n_base, n_stack, scope_name, reuse=True)

        loss_pen = tf.reduce_mean(tf.abs(recon_pen - img_pen))
        loss_sketch = tf.reduce_mean(tf.abs(recon_sketch - img_sketch))
        loss_fmap = tf.reduce_mean(tf.abs(enc_fmap_sketch[0] -
            dec_fmap_sketch[n_stack - 2]))

        global_step = tf.train.get_global_step()
        loss = loss_pen - 0*loss_sketch + loss_fmap
        optimizer = tf.train.AdamOptimizer(learning_rate)


        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)

        #train_op = optimizer.minimize(loss)
        image_dict = {'recon_sketch': recon_sketch, 'recon_pen': recon_pen}
        loss_dict = {'loss_pen': loss_pen, 'loss_sketch': loss_sketch,
                'loss_fmap':loss_fmap, 'loss': loss}

    else:
        img_sketch = features['img_sketch']
        recon_sketch, _, _ = AutoEncoder(
                img_sketch, n_base, n_stack, scope_name, reuse=True)
        image_dict = {'recon_sketch': recon_sketch}
        loss = None
        train_op = None

    return train_op, loss_dict, image_dict


