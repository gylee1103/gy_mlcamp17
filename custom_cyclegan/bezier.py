import cv2
import math
import numpy as np

def getPt(n1, n2, perc):
    diff = n2 - n1
    return n1 + diff*perc

def drawCircle(canvas):
    r = np.random.randint(10, 50, 1)
    a, b = np.random.randint(r, 256-r, 2)
    for i in range(0, 1000):
        t = math.pi * 2.0 * i / 1000.0
        x = int(a + r * math.cos(t))
        y = int(b + r * math.sin(t))

        for x in range(max(0, x-1), min(256, x+1)):
            for y in range(max(0, y-1), min(256, y+1)):
                canvas[x, y] = 0
    
    

def drawLine(points, canvas):
    x1, y1, x2, y2, x3, y3, x4, y4 = points
    for i in range(0, 1000):
        perc = i / 1000.0
        xa = getPt(x1, x2, perc)
        ya = getPt(y1, y2, perc)
        xb = getPt(x2, x3, perc)
        yb = getPt(y2, y3, perc)
        xc = getPt(x3, x4, perc)
        yc = getPt(y3, y4, perc)

        xm = getPt(xa, xb, perc)
        ym = getPt(ya, yb, perc)
        xn = getPt(xb, xc, perc)
        yn = getPt(yb, yc, perc)

        x = int(getPt(xm, xn, perc))
        y = int(getPt(ym, yn, perc))

        for x in range(max(0, x-1), min(256, x+1)):
            for y in range(max(0, y-1), min(256, y+1)):
                canvas[x, y] = 0

def main():
    canvas = np.ones([256, 256])
    # first, randomly select three points
    points = np.random.randint(0, 256, 8)
    x1, y1, x2, y2, x3, y3, x4, y4 = points
    drawLine(points, canvas)

    n_line = np.random.randint(10, 20)
    print (n_line)
    for i in range(n_line):
        points = np.random.randint(0, 256, 8)
        points[0] = x4
        points[1] = y4
        x1, y1, x2, y2, x3, y3, x4, y4 = points

        drawLine(points, canvas)
    n_circle = np.random.randint(0,3)
    for i in range(n_circle):
        drawCircle(canvas)

    cv2.imshow("haha", canvas)
    cv2.waitKey(0)


if __name__ == "__main__":
    while True:
        main()

