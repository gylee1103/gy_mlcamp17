import random
import copy
import numpy as np

class ImagePool(object):
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.num = 0
        self.images = []

    def __call__(self, new_images):
        if self.pool_size == 0:
            return new_images
        return_images = []
        for img in new_images:
            if self.num < self.pool_size:
                self.images.append(img)
                self.num = self.num + 1
                return_images.append(img)
            else:
                if np.random.rand() > 0.5:
                    idx = np.random.randint(0, self.pool_size)
                    tmp = copy.copy(self.images[idx])
                    self.images[idx] = img
                    return_images.append(tmp)
                else:
                    return_images.append(img)
        return return_images
