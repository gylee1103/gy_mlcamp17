from multiprocessing import Process, Queue
from time import sleep
import tensorflow as tf
import random
import os
import sys
import numpy as np
import scipy
import scipy.misc
import scipy.ndimage
from PIL import Image
from db.data_handler import DataHandler

class SketchDataHandler(DataHandler):
    def __init__(self, root_path, paths_file, batch_size, target_size):
      super(SketchDataHandler, self).__init__(batch_size, target_size)
      self._index = 0
      self.root_path = root_path
      self._image_paths = self._get_image_paths(os.path.join(
        root_path, paths_file))
      if (len(self._image_paths) < batch_size * 100):
        self._image_paths = self._image_paths * 100
      self._shuffle_image_paths()
      self._total_num = len(self._image_paths)

      self.queue = Queue(40)
      self.msg_queue = Queue(4)
      self.procs = []
      self.start_threads()

    def _get_image_paths(self, paths_file):
      with open(paths_file) as f:
          return [os.path.join(self.root_path, line.rstrip('\n')) for line in f]

    def _shuffle_image_paths(self):
        random.shuffle(self._image_paths)

    def _random_preprocessing(self, image, size):
      # rotate image
      rand_degree = np.random.randint(0, 90)
      rand_flip = np.random.randint(0, 2)
      if rand_flip == 1:
        image = np.flip(image, 1)
      image = scipy.ndimage.interpolation.rotate(image, rand_degree, cval=255)

      # Select cropping range between (target_size/2 ~ original_size)
      original_h, original_w = image.shape
      #crop_width = np.random.randint(self.target_size/3, min(self.target_size, original_w))
      #crop_height = np.random.randint(self.target_size/3, min(self.target_size, original_h))
      crop_width = self.target_size
      crop_height = self.target_size
      topleft_x = np.random.randint(0, original_w - crop_width)
      topleft_y = np.random.randint(0, original_h - crop_height)
      cropped_img = image[topleft_y:topleft_y+crop_height,
          topleft_x:topleft_x+crop_width]
      #output = scipy.misc.imresize(cropped_img, [self.target_size, self.target_size])
      output = cropped_img

      output = (output - 128.0) / 128.0
      return output

    def next(self):
      output = self.queue.get()
      return output

    def _enqueue_op(self, queue, msg_queue):
      while msg_queue.qsize() == 0:
        # randomly select index
        indexes = np.random.randint(0, self._total_num, self.batch_size)
        sz = self.target_size
        output = np.zeros([self.batch_size, sz, sz, 1])
        for i in range(len(indexes)):
          index = indexes[i]
          output[i] = self._random_preprocessing(scipy.misc.imread(
            self._image_paths[index], mode='L').astype(np.float),
            self.target_size).reshape([sz, sz, 1])
          while np.amin(output[i]) == np.amax(output[i]): # some data are strange..
            output[i] = self._random_preprocessing(scipy.misc.imread(
              self._image_paths[index], mode='L').astype(np.float32),
              self.target_size).reshape([sz, sz, 1])

        queue.put(output)


    def start_threads(self):
      print("start threads called")
      for i in range(2):
        proc = Process(target=self._enqueue_op, args=(self.queue, self.msg_queue))
        self.procs.append(proc)
        proc.daemon = True
        proc.start()
      print("enqueue thread started!")


    def get_batch_shape(self):
      return (self.batch_size, self.target_size, self.target_size, 1)

    def kill(self):
      self.msg_queue.put("illkillyou")
      for proc in self.procs:
        proc.terminate()
        proc.join()
      print('sketch data killed')
 
if __name__ == '__main__':
  pass
