from multiprocessing import Process, Queue
from time import sleep
import tensorflow as tf
import random
import sys
import math
import numpy as np
import scipy
import scipy.misc
from PIL import Image
from data_handler import DataHandler

class LineDataHandler(DataHandler):
    def __init__(self, batch_size, target_size): # Not use datafiles
      super(LineDataHandler, self).__init__(batch_size, target_size)
      self.queue = Queue(40)
      self.msg_queue = Queue(4)
      self.procs = []
      self.start_threads()

    def getPt(self, n1, n2, perc):
        diff = n2 - n1
        return n1 + diff*perc

    def drawLine(self, points, canvas):
        h, w, _ = canvas.shape
        x1, y1, x2, y2 = points
        for i in range(0, 1000):
            perc = i / 1000.0
            x = int(self.getPt(x1, x2, perc))
            y = int(self.getPt(y1, y2, perc))
            for x_ in range(max(0, x-1), min(w, x+1)):
                for y_ in range(max(0, y-1), min(h, y+1)):
                    canvas[y_, x_] = 1

    def drawCircle(self, canvas):
      h, w, _ = canvas.shape
      r = np.random.randint(10, 50, 1)
      a, b = np.random.randint(r, 256-r, 2)
      for i in range(0, 1000):
          t = math.pi * 2.0 * i / 1000.0
          x = int(a + r * math.cos(t))
          y = int(b + r * math.sin(t))

          for x_ in range(max(0, x-1), min(w, x+1)):
              for y_ in range(max(0, y-1), min(h, y+1)):
                  canvas[y_, x_] = 1
   
    def _draw_canvas(self):
      canvas_size = int(self.target_size * 1.5)
      canvas = np.zeros([canvas_size, canvas_size, 1])
      # first, randomly select three points
      points = np.random.randint(0, canvas_size, 4)
      x1, y1, x2, y2 = points

      self.drawLine(points, canvas)
      
      n_line = np.random.randint(5, 15)
      for i in range(n_line):
          points = np.random.randint(0, canvas_size, 4)
          points[0] = x2
          points[1] = y2
          x1, y1, x2, y2 = points

          self.drawLine(points, canvas)

      #n_circle = np.random.randint(0, 2)
      #for i in range(n_circle):
      #  self.drawCircle(canvas)


      topleft_y, topleft_x = \
          np.random.randint(0, canvas_size - self.target_size, 2)

      cropped_canvas = canvas[topleft_y:topleft_y+self.target_size, 
          topleft_x:topleft_x+self.target_size]
      # invert color and zero centering
      cropped_canvas = 1.0 - cropped_canvas
      cropped_canvas = cropped_canvas*2.0 - 1.0
      return cropped_canvas

    def next(self):
      output = self.queue.get()
      return output

    def _enqueue_op(self, queue, msg_queue):
      while msg_queue.qsize() == 0:
        sz = self.target_size
        output = np.zeros([self.batch_size, sz, sz, 1])
        for i in range(self.batch_size):
          output[i] = self._draw_canvas()
        queue.put(output)

    def start_threads(self):
      print("start threads called")
      for i in range(2):
        proc = Process(target=self._enqueue_op, args=(self.queue, self.msg_queue))
        self.procs.append(proc)
        proc.start()

      print("enqueue thread started!")


    def get_batch_shape(self):
      return (self.batch_size, self.target_size, self.target_size, 1)

    def kill(self):
      self.msg_queue.put("illkillyou")
      for proc in self.procs:
        proc.terminate()
        proc.join()
      print("bezier data killed")
 
if __name__ == '__main__':
    pass
