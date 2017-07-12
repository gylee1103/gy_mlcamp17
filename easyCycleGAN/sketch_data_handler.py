import random
import sys
import numpy as np
import scipy
import scipy.misc
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
      
    def _get_image_paths(self, paths_file):
      with open(paths_file) as f:
          return [line.rstrip('\n') for line in f]

    def _shuffle_image_paths(self):
        random.shuffle(self._image_paths)

    def _random_preprocessing(self, image, size):
      # Select cropping range between (target_size/2 ~ original_size)
      original_h, original_w = image.shape
      crop_width = np.random.randint(self.target_size/2, original_w)
      crop_height = np.random.randint(self.target_size/2, original_h)
      topleft_x = np.random.randint(0, original_w - crop_width)
      topleft_y = np.random.randint(0, original_h - crop_height)
      cropped_img = image[topleft_y:topleft_y+crop_height,
          topleft_x:topleft_x+crop_width]
      output = scipy.misc.imresize(cropped_img, [self.target_size, self.target_size])
      return output

    def next(self):
        sz = self.target_size
        output = np.zeros([self.batch_size, sz, sz, 1])
        if (self._index + self.batch_size >= self._total_num):
            self._shuffle_image_paths()
            self._index = 0
        for i in range(self.batch_size):
            cindex = self._index + i

            output[i] = self._random_preprocessing(scipy.misc.imread(
              self._image_paths[cindex], mode='L').astype(np.float),
              self.target_size).reshape([sz, sz, 1])
           
        self._index += self.batch_size
        return output

    def get_batch_shape(self):
      return (self.batch_size, self.target_size, self.target_size, 1)

if __name__ == '__main__':
  test_handler = SketchDataHandler('pen_list.txt', 128, 256)
  mybatch = test_handler.next()
