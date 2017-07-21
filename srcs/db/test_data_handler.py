import random
import sys
import numpy as np
import os
import scipy
import scipy.misc
from PIL import Image
from data_handler import DataHandler

class TestDataHandler(DataHandler):
    def __init__(self, root_path, paths_file, max_size=1024):
      super(TestDataHandler, self).__init__(1, max_size)

      self._index = 0
      self.root_path = root_path
      self._image_paths = self._get_image_paths(
          os.path.join(root_path, paths_file))
      self._image_paths.sort()

      self._total_num = len(self._image_paths)
      
    def num_test(self):
      return self._total_num
    

    def _get_image_paths(self, paths_file):
      with open(paths_file) as f:
          return [os.path.join(self.root_path, line.rstrip('\n')) for line in f]

    def next(self):
        sz = self.target_size
        output = np.ones([1, sz, sz, 1]).astype(np.float32)
        img = scipy.misc.imread(
            self._image_paths[self._index], mode='L').astype(np.float32)
        original_size = img.shape
        bigger_size = max(original_size[0], original_size[1])

        mult = 1
        if bigger_size > self.target_size:
          mult = self.target_size / float(bigger_size)



        resized_size = (int(original_size[0] * mult), int(original_size[1]*mult))
        img = scipy.misc.imresize(img, resized_size)
        img = (img - 128.0) / 128.0
        output[0, 0:resized_size[0], 0:resized_size[1], 0] = img
           
        self._index += 1

        return output, original_size, resized_size

