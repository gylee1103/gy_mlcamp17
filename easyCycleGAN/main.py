import numpy as np
import tensorflow as tf
from config import *
import sketch_model

def img_normalize(img):
    return img / 127.5 - 1

def img_denorm(nimg):
    return int((nimg + 1)*127.5)

def train():
    pen_data_handler = config.pen_data_handler
    sketch_data_handler = config.sketch_data_handler

    img_pen = tf.placeholder(tf.float32, [config.batch_size, config.target_size,
        config.target_size, 1])
    img_sketch = tf.placeholder(tf.float32, [config.batch_size, config.target_size,
        config.target_size, 1])

    # Build model
    train_op, loss_dict, image_dict = \
            sketch_model.build_model({'img_sketch': img_sketch,
                'img_pen': img_pen}, tf.estimator.ModeKeys.TRAIN,
                params={'n_base':config.n_base, 'n_stack':config.n_stack,
                    'learning_rate':config.learning_rate}, config=None)

    #lr_update = tf.assign(learning_rate, tf.maximum(learning_rate * 0.5,
    #    config.lr_lower_bound), name="lr_update")

    # Set summary
    summary_writer = tf.summary.FileWriter(config.summary_path)
    summary_op = tf.summary.merge([
        tf.summary.image("img_sketch", img_sketch),
        tf.summary.image("img_pen", img_pen),
        tf.summary.image("recon_sketch", image_dict['recon_sketch']),
        tf.summary.image("recon_pen", image_dict['recon_pen']),
        tf.summary.scalar("loss/loss_pen", loss_dict['loss_pen']),
        tf.summary.scalar("loss/loss_sketch", loss_dict['loss_sketch']),
        tf.summary.scalar("loss/loss_fmap", loss_dict['loss_fmap']),
        tf.summary.scalar("loss/loss", loss_dict['loss']),
    ])

    # Start Training Iteration
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for step in range(config.step):
            fetch_dict = {
                "train_op": train_op,
                "loss": loss_dict['loss'],
            }

            if step % config.log_step == 0:
                fetch_dict.update({
                    "summary": summary_op,
                })

            result = sess.run(fetch_dict, 
                    feed_dict={img_sketch: \
                        img_normalize(sketch_data_handler.next()),
                        img_pen: img_normalize(pen_data_handler.next()),
                            })

            if step % config.log_step == 0 :
                summary_writer.add_summary(result["summary"], step)
                summary_writer.flush()

            print "Iter %d, loss %f" % (step, result["loss"])

            #if step % config.lr_update_step == 0:
            #    sess.run(lr_update)
   


def test():
    pass

if __name__ == "__main__":
    train()

