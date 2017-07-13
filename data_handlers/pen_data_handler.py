from multiprocessing import Process, Queue
from time import sleep
import tensorflow as tf
import random
import sys
import numpy as np
import scipy
import scipy.misc
from PIL import Image
from data_handler import DataHandler

class PenDataHandler(DataHandler):
    def __init__(self, batch_size, target_size): # Not use datafiles
      super(PenDataHandler, self).__init__(batch_size, target_size)
      self.queue = Queue(40)
      self.procs = []
      self.start_threads()

    def getPt(self, n1, n2, perc):
        diff = n2 - n1
        return n1 + diff*perc

    def drawLine(self, points, canvas):
        x1, y1, x2, y2, x3, y3 = points
        for i in range(0, 1000):
            perc = i / 1000.0
            xa = self.getPt(x1, x2, perc)
            ya = self.getPt(y1, y2, perc)
            xb = self.getPt(x2, x3, perc)
            yb = self.getPt(y2, y3, perc)

            x = int(self.getPt(xa, xb, perc))
            y = int(self.getPt(ya, yb, perc))

            canvas[x, y] = 1


    def _draw_canvas(self):
      canvas = np.zeros([self.target_size*2, self.target_size*2, 1])
      # first, randomly select three points
      points = np.random.randint(0, self.target_size*2, 6)
      x1, y1, x2, y2, x3, y3 = points
      self.drawLine(points, canvas)
      for i in range(20):
          points = np.random.randint(0, self.target_size*2, 6)
          points[0] = x3
          points[1] = y3
          x1, y1, x2, y2, x3, y3 = points

          self.drawLine(points, canvas)

      crop_point = int(self.target_size*0.3)
      cropped_canvas = canvas[crop_point:crop_point+self.target_size, 
          crop_point:crop_point+self.target_size]

      return cropped_canvas

    def next(self):
      print ("wait queue")
      output = self.queue.get()
      print("dequeued, remain %d" % self.queue.qsize())
      return output

    def _enqueue_op(self, queue):
      while True:
        sz = self.target_size
        output = np.zeros([self.batch_size, sz, sz, 1])
        for i in range(self.batch_size):
          output[i] = self._draw_canvas()
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
  pass
