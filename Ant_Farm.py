import cv2
import numpy as np
import random
import math


def line(p1, p2, lr_canvas, ud_canvas, value):
    x1, y1 = p1
    x2, y2 = p2

    dx = x1 - x2
    dy = y1 - y2

    dir = math.atan2(dy, dx)
    dist = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))

    for i in range(int(dist)):
        px = x1 + dx * i / dist
        py = y1 + dy * i / dist

        if not is_oob(lr_canvas, px, py):
            lr_canvas[int(px), int(py)] += value * math.cos(dir)
            ud_canvas[int(px), int(py)] += value * -math.sin(dir)


def is_oob(canvas, x, y):
    w, h = canvas.shape
    if x < 0 or x >= w or y < 0 or y >= h:
        return True
    return False


class Ant:
    def __init__(self, px, py):
        self.px = px
        self.py = py
        self.dir = random.uniform(0, 2*math.pi)
        self.dist = 5
        self.scan_angle = math.pi/3
        self.num_scans = 9

    def scan(self, lr_canvas, ud_canvas):
        w, h = lr_canvas.shape
        scan_results = []

        for i in range(self.num_scans): # for every direction of sight
            frac = i/self.num_scans
            direction = self.dir - self.scan_angle + (frac * self.scan_angle*2)

            scan_x, scan_y = self.pos_from_dir(direction)

            scan_x = scan_x % w
            scan_y = scan_y % h

            val_lr = lr_canvas[int(scan_x), int(scan_y)]
            val_ud = ud_canvas[int(scan_x), int(scan_y)]

            pheromone_direction = math.atan2(val_ud, val_lr)
            pheromone_intensity = math.sqrt(math.pow(val_ud, 2) + math.pow(val_lr, 2))

            scan_results.append((direction, pheromone_direction, pheromone_intensity))

        # find the best direction to walk in
        best_direction = 0
        best_eval = -1
        for dir, pdir, pint in scan_results:
            eval = pint + 250 - abs(pdir - dir)
            if eval > best_eval:
                best_eval = eval
                best_direction = dir
        self.walk(lr_canvas, ud_canvas, best_direction)

    def pos_from_dir(self, direction):
        new_px = self.px + math.cos(direction) * self.dist
        new_py = self.py - math.sin(direction) * self.dist

        return new_px, new_py

    def walk(self, lr_canvas, ud_canvas, direction):
        w, h = lr_canvas.shape

        dx = math.cos(direction)*self.dist
        dy = -math.sin(direction)*self.dist

        # target_x = int(self.px + dx)
        # target_y = int(self.py + dy)
        #
        # if target_x >= w or target_x < 0:
        #     dx *= -1
        # if target_y >= h or target_y < 0:
        #     dy *= -1

        new_px = (self.px + dx) % w
        new_py = (self.py + dy) % h

        line((self.px, self.py), (new_px, new_py), lr_canvas, ud_canvas, 1)

        self.px = new_px
        self.py = new_py


def main():
    # WDITH = 500
    # HEIGHT = 500
    lr_canvas = np.zeros((500, 500))
    ud_canvas = np.zeros((500, 500))

    w, h = lr_canvas.shape

    ants = []
    for i in range(2000):
        ants.append(Ant(random.randrange(0, w-1), random.randrange(0, h-1)))

    while True:
        for ant in ants:
            ant.scan(lr_canvas, ud_canvas)

        lr_canvas = lr_canvas * .99
        ud_canvas = ud_canvas * .99

        lr_canvas = cv2.GaussianBlur(lr_canvas, (3, 3), .5)
        ud_canvas = cv2.GaussianBlur(ud_canvas, (3, 3), .5)

        cv2.imshow("ant farm", np.abs(lr_canvas)+np.abs(ud_canvas))
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


if __name__ == "__main__":
    main()