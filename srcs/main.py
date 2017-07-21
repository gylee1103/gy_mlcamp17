from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tensorflow as tf
import numpy as np
import datetime

time_now = datetime.datetime.now()
def get_output_model_path():
  FLAGS = tf.flags.FLAGS
  if FLAGS.output_model_path == None:
    output_dir = "checkpoints/" + gen_model_name
      time_now.hour, time_now.minute)
    if FLAGS.run_mlengine:
      return os.path.join(FLAGS.mlengine_path, output_dir)
    else:
      return output_dir # working dir
  else:
    return FLAGS.output_model_path

def get_dataset_path():
  if tf.flags.FLAGS.run_mlengine:
    gs_dbpath = os.path.join(tf.flags.FLAGS.mlengine_path, "SketchDB")
    os.system("mkdir dataset")
    os.system("gsutil -m cp -r %s $(pwd)/dataset" % gs_dbpath)
    return os.path.abspath("./dataset/SketchDB")
  else:
    return os.path.abspath("../SketchDB")

def gen_model_name:
    FLAGS = tf.flags.FLAGS
    name = "%s_%s_%02d_%02d_%02d_%02d" % (FLAGS.model_type, FLAGS.data_type,
            time_now.month, time_now.day, time_now.hour, time_now.minute)
    return name

def parse_arguments():

  tf.flags.DEFINE_string('model_type', 'cycle_gan',
      'Choose one of four types: fcn, cgan, cycle_gan, our_cycle_gan, default: our_cycle_gan')
  tf.flags.DEFINE_string('data_type', 'bezier',
      'Choose dataset for pen data: bezier, real, all, default: bezier')

  tf.flags.DEFINE_integer('batch_size', 8, 'batch size, default: 12')
  tf.flags.DEFINE_integer('target_size', 256, 'Image size, default: 256')
  tf.flags.DEFINE_integer(
      'num_block', 4, 'the number of residual block, default: 4')

  tf.flags.DEFINE_integer(
      'log_step', 100, 'How often write the summary, default: 100')
  tf.flags.DEFINE_integer(
      'save_step', 1000, 'How often save the trained model, default: 1000')

  tf.flags.DEFINE_string('X', 'sketch_list.txt',
      'text file that contains paths of files for Dataset X. default: sketch_list.txt')
  tf.flags.DEFINE_string('Y', 'pen_list.txt',
      'text file that contains paths of files for Dataset Y. default: pen_list.txt')

  tf.flags.DEFINE_string('output_model_path', None,
      'output model path to save trained model. default: None(automatic)')
  tf.flags.DEFINE_string('saved_model_path', None,
      'model path to restore and continue training. default: None')

  tf.flags.DEFINE_string('mode', 'train', 
      'execution mode(train or test), default: train')

  tf.flags.DEFINE_boolean('run_mlengine', False, 'whether to run in mlengine environment')
  tf.flags.DEFINE_string('mlengine_path', 'gs://sketch-simplification-mlengine/', 'default gs path')


def test():
  FLAGS = tf.flags.FLAGS
  is_training = False
  graph = tf.Graph()
  max_size = 1024
  if FLAGS.run_mlengine:
    from srcs.db.sketch_data_handler import SketchDataHandler
    from srcs.db.pen_data_handler import PenDataHandler
    from srcs.db.test_data_handler import TestDataHandler
    from srcs.db.bezier_data_handler import BezierDataHandler

    import srcs.models.fcn
    import srcs.models.cgan
    import srcs.models.cycle_gan
    import srcs.models.our_cycle_gan
    models = srcs.models
  else:
    from db.sketch_data_handler import SketchDataHandler
    from db.pen_data_handler import PenDataHandler
    from db.test_data_handler import TestDataHandler
    from db.bezier_data_handler import BezierDataHandler

    import models.fcn
    import models.cgan
    import models.cycle_gan
    import models.our_cycle_gan
  # Model type
  if FLAGS.model_type == 'fcn':
    model = models.fcn
  elif FLAGS.model_type == 'cgan':
    model = models.cgan
  elif FLAGS.model_type == 'cycle_gan':
    model = models.cycle_gan
  elif FLAGS.model_type == 'our_cycle_gan':
    model = models.our_cycle_gan
  else:
    print("no match model for %s" % FLAGS.model_type)
    exit(-1)


  with graph.as_default():
    data_handler_X = TestDataHandler(
        get_dataset_path, FLAGS.X, max_size=max_size)
    num_test = data_handler_X.num_test()

    input_X = tf.placeholder_with_default(
        tf.zeros([1, max_size, max_size, 1]),
        [1, max_size, max_size, 1], name='input_X')

    input_Y = tf.placeholder_with_default(
        tf.zeros([1, max_size, max_size, 1]),
        [1, max_size, max_size, 1], name='input_Y')

    # --------------------------------------------------------------------
    # Model here
    # --------------------------------------------------------------------

    [ train_op, losses, predictions ] = model.build_model(input_X, 
        input_Y, is_training=False)

    model_saver = tf.train.Saver()

  with tf.Session(graph=graph) as sess:
    if FLAGS.saved_model_path is not None:
      model_saver.restore(sess, FLAGS.saved_model_path)
      step = 0
    else:
      print("model path is required for test run")
      exit(-1)


    for step in range(num_test):
      fetch_dict = {
          "output": predictions["Y_from_X"],
      }

      img, original_size, resized_size = data_handler_X.next()

      result = sess.run(fetch_dict,
          feed_dict={input_X: img})

      # Crop and 
      pen_img = result["output"]
      height, width = resized_size
      pen_img = pen_img.reshape([max_size, max_size])
      pen_img = pen_img[0:height, 0:width]
      pen_img = scipy.misc.imresize(pen_img, original_size)
      pen_img = pen_img * 128.0 + 128.0

      scipy.misc.imsave('%06d.png' % step, pen_img) 

      print("Image %d, processed" % (step))


def train():
  FLAGS = tf.flags.FLAGS
  graph = tf.Graph()
  output_model_path = get_output_model_path()

  if FLAGS.run_mlengine:
    from srcs.db.sketch_data_handler import SketchDataHandler
    from srcs.db.pen_data_handler import PenDataHandler
    from srcs.db.test_data_handler import TestDataHandler
    from srcs.db.bezier_data_handler import BezierDataHandler

    import srcs.models.fcn
    import srcs.models.cgan
    import srcs.models.cycle_gan
    import srcs.models.our_cycle_gan
    models = srcs.models
  else:
    from db.sketch_data_handler import SketchDataHandler
    from db.pen_data_handler import PenDataHandler
    from db.test_data_handler import TestDataHandler
    from db.bezier_data_handler import BezierDataHandler

    import models.fcn
    import models.cgan
    import models.cycle_gan
    import models.our_cycle_gan

  data_handler_X = SketchDataHandler(
      get_dataset_path(), FLAGS.X, FLAGS.batch_size, FLAGS.target_size)

  # Dataset type
  if FLAGS.data_type == 'bezier':
    data_handler_Y = BezierDataHandler(
        FLAGS.batch_size, FLAGS.target_size)
  elif FLAGS.data_type == 'real':
    data_handler_Y = PenDataHandler(
        get_dataset_path, FLAGS.Y, FLAGS.batch_size, FLAGS.target_size)
  elif FLAGS.data_type == 'all':
    data_handler_Y = MixedDataHandler(
        get_dataset_path, FLAGS.Y, FLAGS.batch_size, FLAGS.target_size)
  else:
    print("no match dataset for %s" % FLAGS.data_type)
    exit(-1)

  # Model type
  if FLAGS.model_type == 'fcn':
    model = models.fcn
  elif FLAGS.model_type == 'cgan':
    model = models.cgan
  elif FLAGS.model_type == 'cycle_gan':
    model = models.cycle_gan
  elif FLAGS.model_type == 'our_cycle_gan':
    model = models.our_cycle_gan
  else:
    print("no match model for %s" % FLAGS.model_type)
    exit(-1)
  

  try:
    with graph.as_default():
      input_X = tf.placeholder(tf.float32, 
          shape=data_handler_X.get_batch_shape(), name='input_X')
      input_Y = tf.placeholder(tf.float32,
          shape=data_handler_Y.get_batch_shape(), name='input_Y')

      # --------------------------------------------------------------------
      # Model here
      # --------------------------------------------------------------------
      [ train_op, losses, predictions ] = model.build_model(input_X, input_Y)

      # choose summary
      summary_list = [
        tf.summary.image("X/input_X", input_X),
        tf.summary.image("X/Y_from_X", predictions['Y_from_X']), # output
        tf.summary.image("Y/input_Y", input_Y),
        tf.summary.scalar("loss/loss", losses['loss']), # total loss
      ]

      if model == models.cgan: # gan family
        summary_list.extend([
            tf.summary.scalar("loss/loss_G", losses['loss_G']),
            tf.summary.scalar("loss/loss_D", losses['loss_D']),
        ])
      elif model == models.fcn:
        summary_list.extend([
          tf.summary.image("Y/Y_after_FCN", predictions['Y_after_FCN']),
          tf.summary.scalar("loss/loss_FCN_X", losses["loss_FCN_X"]),
        ])
      elif model == models.cycle_gan or model == models.our_cycle_gan:
        summary_list.extend([
            tf.summary.image("Y/X_from_Y", predictions['X_from_Y']),
            tf.summary.image("X/X_cycled", predictions['X_cycled']),
            tf.summary.image("Y/Y_cycled", predictions['Y_cycled']),
            tf.summary.scalar("loss/loss_cycle", losses['loss_cycle']),
            tf.summary.scalar("loss/loss_DX", losses['loss_DX']),
            tf.summary.scalar("loss/loss_DY", losses['loss_DY']),
            tf.summary.scalar("loss/loss_F", losses['loss_F']),
            tf.summary.scalar("loss/loss_G", losses['loss_G']),

        ])
      else:
        print("not reached here")
        exit(-1)

      summary_op = tf.summary.merge(summary_list)
      summary_writer = tf.summary.FileWriter(output_model_path)
      model_saver = tf.train.Saver(max_to_keep=1000)

    with tf.Session(graph=graph) as sess:
      if FLAGS.saved_model_path is not None:
          checkpoint = tf.train.get_checkpoint_state(FLAGS.saved_model_path)
          model_saver.restore(sess, tf.train.latest_checkpoint(FLAGS.saved_model_path))
          meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
          step = int(meta_graph_path.split("-")[1].split(".")[0])
      else:
        sess.run(tf.global_variables_initializer())
        step = 0

      try:
        while True:
          fetch_dict = {
              "train_op": train_op,
              "loss": losses['loss'],
          }
          if step % FLAGS.log_step == 0:
            fetch_dict.update({
              "summary": summary_op,
            })

          result = sess.run(fetch_dict,
              feed_dict={input_X: data_handler_X.next(),
                input_Y: data_handler_Y.next(),})

          if step % FLAGS.log_step == 0:
            summary_writer.add_summary(result["summary"], step)
            summary_writer.flush()

          if step % FLAGS.save_step == 0:
            save_path = model_saver.save(sess, 
                os.path.join(output_model_path, "model.ckpt"),
                global_step= step)


          print("Iter %d, loss %f" % (step, result["loss"]))
          step += 1

      finally:
        save_path = model_saver.save(sess, 
            os.path.join(output_model_path, "model.ckpt"),
            global_step= step)
        
  finally:
    data_handler_X.kill()
    data_handler_Y.kill()

def main(args=None):
  FLAGS = tf.flags.FLAGS
  if FLAGS.mode == 'train':
    train()
  else:
    test()

if __name__ == '__main__':
  parse_arguments()
  tf.app.run()
