from multiprocessing import Process, Queue
from time import sleep
import tensorflow as tf
import random
import sys
import numpy as np
import scipy
import scipy.misc
import scipy.ndimage
from PIL import Image
from data_handler import DataHandler

class SketchDataHandler(DataHandler):
    def __init__(self, paths_file, batch_size, target_size):
      super(SketchDataHandler, self).__init__(batch_size, target_size)
      self._index = 0
      self._image_paths = self._get_image_paths(paths_file)
      if (len(self._image_paths) < batch_size * 100):
        self._image_paths = self._image_paths * 100
      self._shuffle_image_paths()
      self._total_num = len(self._image_paths)

      self.queue = Queue(40)
      self.procs = []
      self.start_threads()

    def _get_image_paths(self, paths_file):
      with open(paths_file) as f:
          return [line.rstrip('\n') for line in f]

    def _shuffle_image_paths(self):
        random.shuffle(self._image_paths)

    def _random_preprocessing(self, image, size):
      # rotate image
      rand_degree = np.random.randint(0, 180)
      rand_flip = np.random.randint(0, 2)
      image = scipy.ndimage.interpolation.rotate(image, rand_degree)
      if rand_flip == 1:
        image = np.flip(image, 1)

      # Select cropping range between (target_size/2 ~ original_size)
      original_h, original_w = image.shape
      crop_width = np.random.randint(self.target_size/2, original_w)
      crop_height = np.random.randint(self.target_size/2, original_h)
      topleft_x = np.random.randint(0, original_w - crop_width)
      topleft_y = np.random.randint(0, original_h - crop_height)
      cropped_img = image[topleft_y:topleft_y+crop_height,
          topleft_x:topleft_x+crop_width]
      output = scipy.misc.imresize(cropped_img, [self.target_size, self.target_size])
      output = (output - 128.0) / 128.0
      return output

    def next(self):
      print ("wait queue")
      output = self.queue.get()
      print("dequeued, remain %d" % self.queue.qsize())
      return output

    def _enqueue_op(self, queue):
      while True:
        # randomly select index
        indexes = np.random.randint(0, self._total_num, self.batch_size)
        sz = self.target_size
        output = np.zeros([self.batch_size, sz, sz, 1])
        for i in range(len(indexes)):
          index = indexes[i]
          output[i] = self._random_preprocessing(scipy.misc.imread(
            self._image_paths[index], mode='L').astype(np.float),
            self.target_size).reshape([sz, sz, 1])
        queue.put(output)


    def start_threads(self):
      print("start threads called")
      proc = Process(target=self._enqueue_op, args=((self.queue),))
      self.procs.append(proc)
      proc.start()
      print("enqueue thread started!")


    def get_batch_shape(self):
      return (self.batch_size, self.target_size, self.target_size, 1)

if __name__ == '__main__':
  test_handler = SketchDataHandler('pen_list.txt', 128, 256)
  mybatch = test_handler.next()
