from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tensorflow as tf
import numpy as np
import datetime

from sketch_data_handler import *
import cycle_gan

time_now = datetime.datetime.now()

def get_summary_path():
  path = "summary_%02d_%02d_%02d_%02d" % (time_now.month, time_now.day,
      time_now.hour, time_now.minute)
  return path

def get_output_model_path():
  path = "output_%02d_%02d_%02d_%02d" % (time_now.month, time_now.day,
      time_now.hour, time_now.minute)
  return path

def parse_arguments():
  tf.flags.DEFINE_integer('batch_size', 16, 'batch size, default: 16')
  tf.flags.DEFINE_integer('target_size', 256, 'Image size, default: 256')
  tf.flags.DEFINE_integer(
      'num_block', 4, 'the number of residual block, default: 4')
  tf.flags.DEFINE_integer(
      'log_step', 1000, 'How often write the summary, default: 1000')
  tf.flags.DEFINE_string('X', 'sketch_list.txt',
      'text file that contains paths of files for Dataset X. default: sketch_list.txt')
  tf.flags.DEFINE_string('Y', 'pen_list.txt',
      'text file that contains paths of files for Dataset Y. default: pen_list.txt')
  tf.flags.DEFINE_string('output_model_path', get_output_model_path(),
      'output model path to save trained model. default: None(automatic)')
  tf.flags.DEFINE_string('summary_path', get_summary_path(),
      'summary path. default: None(automatic)')
  tf.flags.DEFINE_string('saved_model_path', None,
      'model path to restore and continue training. default: None')
  tf.flags.DEFINE_string('mode', 'train', 
      'execution mode(train or test), default: train')

def main(args=None):

  FLAGS = tf.flags.FLAGS
  if FLAGS.mode == 'train':
    is_training = True
  else:
    is_training = False
  graph = tf.Graph()

  with graph.as_default():
    data_handler_X = SketchDataHandler(
        FLAGS.X, FLAGS.batch_size, FLAGS.target_size)
    data_handler_Y = SketchDataHandler(
        FLAGS.Y, FLAGS.batch_size, FLAGS.target_size)
    input_X = tf.placeholder_with_default(
        tf.zeros(data_handler_X.get_batch_shape()),
        data_handler_X.get_batch_shape(), name='input_X')
    input_Y = tf.placeholder_with_default(
        tf.zeros(data_handler_Y.get_batch_shape()),
        data_handler_Y.get_batch_shape(), name='input_Y')


    [ train_op, losses, predictions ] = cycle_gan.build_model(input_X, input_Y)

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.summary_path)
    model_saver = tf.train.Saver()

  with tf.Session(graph=graph) as sess:
    if FLAGS.saved_model_path is not None:
      model_saver.restore(sess, tf.train.latest_checkpoint(FLAGS.saved_model_path))
      step = int(meta_graph_path.split("-")[2].split(".")[0])
    else:
      sess.run(tf.global_variables_initializer())
      step = 0

    try:
      while True:
        fetch_dict = {
            "train_op": train_op,
            "loss_DX": losses['loss_DX'],
            "loss_F": losses['loss_F'],
        }
        if step % FLAGS.log_step == 0:
          fetch_dict.update({
            "summary": summary_op,
          })

        result = sess.run(fetch_dict,
            feed_dict={input_X: data_handler_X.get_next_batch(),
              input_Y: data_handler_Y.get_next_batch(),})

        if step % FLAGS.log_step == 0:
          summary_writer.add_summary(result["summary"], step)
          summary_writer.flush()
          save_path = model_saver.save(sess, 
              os.path.join(FLAGS.output_model_path, "model.ckpt"),
              global_step= step)

        print("Iter %d, loss_DX %f, loss_F %f" % (step, result["loss_DX"],
          result['loss_F']))
        step += 1
    finally:
      save_path = model_saver.save(sess, 
          os.path.join(FLAGS.output_model_path, "model.ckpt"),
          global_step= step)

if __name__ == '__main__':
  parse_arguments()
  tf.app.run()
