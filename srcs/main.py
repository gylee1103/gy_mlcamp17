from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tensorflow as tf
import numpy as np
import scipy.misc
import datetime

from db.sketch_data_handler import SketchDataHandler
from db.test_data_handler import TestDataHandler
from db.bezier_data_handler import BezierDataHandler
from db.line_data_handler import LineDataHandler
from image_pool import ImagePool

import models.cycle_gan
import models.our_cycle_gan

time_now = datetime.datetime.now() # Call only once

def get_output_model_dir():
  FLAGS = tf.flags.FLAGS
  if FLAGS.output_model_dir == None:
    name = "%s_%s_%02d_%02d_%02d_%02d" % (FLAGS.model_type, FLAGS.data_type,
            time_now.month, time_now.day, time_now.hour, time_now.minute)
    output_dir = "checkpoints/" + name()
    return output_dir
  else:
    return FLAGS.output_model_dir

def get_data_dir():
  FLAGS = tf.flags.FLAGS
  return os.path.abspath(FLAGS.data_dir)

def parse_arguments():

  # Model Configuration
  tf.flags.DEFINE_string('model_type', 'our_cycle_gan',
      'Choose cycle_gan or our_cycle_gan, default: our_cycle_gan')

  # Training Data Configuration
  tf.flags.DEFINE_string('data_type', 'bezier',
      'Choose dataset for pen data: bezier, line, default: bezier')
  tf.flags.DEFINE_string('data_width', 'fixed', 
      'Synthetic dataset line width: fixed, various, default: fixed')
  tf.flags.DEFINE_string('data_dir', '../SketchDB', 'Dataset path, default: ../SketchDB')
  tf.flags.DEFINE_string('S', 'sketch_list.txt',
      'Text file that contains paths of files for Dataset S. default: sketch_list.txt')
  tf.flags.DEFINE_string('P', 'pen_list.txt',
      'Text file that contains paths of files for Dataset P. default: pen_list.txt')

  # Training Configuration
  tf.flags.DEFINE_integer('batch_size', 6, 'batch size, default: 6')
  tf.flags.DEFINE_integer('target_size', 256, 'Image size, default: 256')
  tf.flags.DEFINE_integer(
      'log_step', 100, 'How often write the summary, default: 100')
  tf.flags.DEFINE_integer(
      'save_step', 1000, 'How often save the trained model, default: 1000')
  tf.flags.DEFINE_string('output_model_dir', None,
      'output model path to save trained model. default: None(automatic)')
  tf.flags.DEFINE_string('restore_model_dir', None, 'Restored model directory. default: None')

  # Testing Configuration
  tf.flags.DEFINE_string('mode', 'train', 'Execution mode(train or test), default: train')
  tf.flags.DEFINE_integer('test_size', 512, 'Test input max size, default: 512')
  tf.flags.DEFINE_string('T', 'test_list.txt',
      'Text file that contains paths of files for testing. default: test_list.txt')
  tf.flags.DEFINE_string('test_dir', './results', 'Path to save output results')
  tf.flags.DEFINE_string('saved_model_file', None,
      'model path to restore and continue training. default: None')

def test():
  FLAGS = tf.flags.FLAGS
  is_training = False
  graph = tf.Graph()
  max_size = FLAGS.test_size
  output_path = os.path.abspath(FLAGS.test_dir)
  try:
    os.makedirs(output_path)
  except:
    pass
    
  # Model type
  if FLAGS.model_type == 'cycle_gan':
    model = models.cycle_gan
  elif FLAGS.model_type == 'our_cycle_gan':
    model = models.our_cycle_gan
  else:
    print("no match model for %s" % FLAGS.model_type)
    exit(-1)


  with graph.as_default():
    data_handler_T = TestDataHandler(
        get_data_dir(), FLAGS.T, max_size=max_size)
    num_test = data_handler_X.num_test()

    input_S = tf.placeholder_with_default(
        tf.zeros([1, max_size, max_size, 1]),
        [1, max_size, max_size, 1], name='input_S')

    input_P = tf.placeholder_with_default(
        tf.zeros([1, max_size, max_size, 1]),
        [1, max_size, max_size, 1], name='input_P') # Not used

    # Model
    [ train_op, losses, predictions ] = model.build_model(input_S, 
        input_P, is_training=False)

    model_saver = tf.train.Saver()

  with tf.Session(graph=graph) as sess:
    if FLAGS.saved_model_file is not None:
      model_saver.restore(sess, FLAGS.saved_model_file)
      step = 0
    else:
      print("model path is required for test run")
      exit(-1)

    for step in range(num_test):
      fetch_dict = {
          "output": predictions["P_from_S"],
      }

      img, original_size, resized_size = data_handler_T.next()

      result = sess.run(fetch_dict, feed_dict={input_S: img})

      # Crop and 
      pen_img = result["output"]
      height, width = resized_size
      pen_img = pen_img.reshape([max_size, max_size])
      pen_img = pen_img[0:height, 0:width]
      pen_img = scipy.misc.imresize(pen_img, original_size, interp='nearest')
      pen_img = pen_img * 128.0 + 128.0

      filepath = os.path.join(output_path, ('%06d.png' % step))

      scipy.misc.imsave(filepath, pen_img) 

      print("Image %d, processed" % (step))


def train():
  FLAGS = tf.flags.FLAGS
  graph = tf.Graph()
  output_model_dir = get_output_model_dir()

  # Sketch dataset handler
  data_handler_S = SketchDataHandler(
      get_data_dir(), FLAGS.S, FLAGS.batch_size, FLAGS.target_size)

  # Pen dataset handler
  if FLAGS.data_type == 'bezier':
    data_handler_P = BezierDataHandler(FLAGS.batch_size, FLAGS.target_size)
  elif FLAGS.data_type == 'line':
    data_handler_P = LineDataHandler(FLAGS.batch_size, FLAGS.target_size)
  else:
    print("no match dataset for %s" % FLAGS.data_type)
    exit(-1)

  # Model type
  if FLAGS.model_type == 'cycle_gan':
    model = models.cycle_gan
  elif FLAGS.model_type == 'our_cycle_gan':
    model = models.our_cycle_gan
  else:
    print("no match model for %s" % FLAGS.model_type)
    exit(-1)
  
  fake_pen_pool = ImagePool()
  fake_sketch_pool = ImagePool()


  try:
    with graph.as_default():
      input_S = tf.placeholder(tf.float32, 
          shape=data_handler_S.get_batch_shape(), name='input_S')
      input_P = tf.placeholder(tf.float32,
          shape=data_handler_P.get_batch_shape(), name='input_P')

      input_FP_pool = tf.placeholder(tf.float32, 
          shape=(FLAGS.batch_size, FLAGS.target_size, FLAGS.target_size, 1), name='input_FP_pool')
      input_FS_pool = tf.placeholder(tf.float32, 
          shape=(FLAGS.batch_size, FLAGS.target_size, FLAGS.target_size, 1), name='input_FS_pool')

      # Model here
      [ train_op, losses, predictions ] = model.build_model(input_S, input_P, 
          input_FS_pool, input_FP_pool)

      # choose summary
      summary_list = [
        tf.summary.image("S/input_S", input_S),
        tf.summary.image("S/P_from_S", predictions['P_from_S']), # output
        tf.summary.image("S/S_cycled", predictions['S_cycled']),
        tf.summary.image("S/noisy_S", predictions['noisy_S']),
        tf.summary.image("P/input_P", input_P),
        tf.summary.image("P/S_from_P", predictions['S_from_P']),
        tf.summary.image("P/P_cycled", predictions['P_cycled']),
        tf.summary.image("P/noisy_P", predictions['noisy_P']),

        tf.summary.scalar("loss/loss_cycle_S", losses['loss_cycle_S']),
        tf.summary.scalar("loss/loss_cycle_P", losses['loss_cycle_P']),
        tf.summary.scalar("loss/loss_cycle", losses['loss_cycle']),
        tf.summary.scalar("loss/loss_DS", losses['loss_DS']),
        tf.summary.scalar("loss/loss_DP", losses['loss_DP']),
        tf.summary.scalar("loss/loss_F", losses['loss_F']),
        tf.summary.scalar("loss/loss_G", losses['loss_G']),
      ]
      if model == models.our_cycle_gan:
          summary_list.extend([
            tf.summary.image("S/extra", predictions['extra']),
          ])

      summary_op = tf.summary.merge(summary_list)
      summary_writer = tf.summary.FileWriter(output_model_dir)
      model_saver = tf.train.Saver(max_to_keep=1000)

    with tf.Session(graph=graph) as sess:
      if FLAGS.restore_model_dir is not None:
          checkpoint = tf.train.get_checkpoint_state(FLAGS.restore_model_dir)
          model_saver.restore(sess, tf.train.latest_checkpoint(FLAGS.restore_model_dir))
          meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
          step = int(meta_graph_path.split("-")[1].split(".")[0])
      else:
        sess.run(tf.global_variables_initializer())
        step = 0

      try:
        while True: # We manually shut down
          # First, generate fake image and update the fake pool. Then train.
          FP, FS = sess.run([predictions['P_from_S'], predictions['S_from_P']],
              feed_dict={input_S: data_handler_S.next(),
                input_P: data_handler_P.next(),})

          # Now train using data and the fake pools.
          fetch_dict = {
              "train_op": train_op,
              "loss": losses['loss'],
              "P_from_S": predictions['P_from_S'],
              "S_from_P": predictions['S_from_P'],
          }

          if step % FLAGS.log_step == 0:
            fetch_dict.update({
              "summary": summary_op,
            })

          result = sess.run(fetch_dict,
              feed_dict={input_S: data_handler_S.next(),
                input_P: data_handler_P.next(),
                input_FS_pool: fake_pen_pool(FP),
                input_FP_pool: fake_sketch_pool(FS),})

          if step % FLAGS.log_step == 0:
            summary_writer.add_summary(result["summary"], step)
            summary_writer.flush()

          if step % FLAGS.save_step == 0:
            save_path = model_saver.save(sess, 
                os.path.join(output_model_dir, "model.ckpt"),
                global_step= step)


          print("Iter %d, loss %f" % (step, result["loss"]))
          step += 1

      finally:
        save_path = model_saver.save(sess, 
            os.path.join(output_model_dir, "model.ckpt"),
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
