import sys
from abc import ABCMeta, abstractmethod

class DataHandler(object):
    __metaclass__ = ABCMeta

    def __init__(self, batch_size, target_size):
        self.batch_size = batch_size
        self.target_size = target_size

    @abstractmethod
    def next(self):
        pass
