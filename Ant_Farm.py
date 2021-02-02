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

    if dist > 100:
        return

    for i in range(int(dist)):
        px = x1 - dx * (i / dist)
        py = y1 - dy * (i / dist)

        if not is_oob(lr_canvas, px, py):
            lr_canvas[int(px), int(py)] += value * math.cos(dir)
            lr_canvas[int(px), int(py)] = max(-100, min(100, lr_canvas[int(px), int(py)]))
            ud_canvas[int(px), int(py)] += value * -math.sin(dir)
            ud_canvas[int(px), int(py)] = max(-100, min(100, ud_canvas[int(px), int(py)]))


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
        self.dist = 10
        self.scan_angle = math.pi/4
        self.num_scans = 5

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

        random.shuffle(scan_results)

        # find the best direction to walk in
        best_direction = 0
        best_eval = -1
        for dir, pdir, pint in scan_results:
            eval = pint + 500 - abs(pdir - dir)
            # eval = min(20 - abs(20 - pint), 15)
            # eval = 15 - abs(pdir - dir) * (pint / 50)
            # eval = 2*math.pi - abs(pdir - dir) + pint/500
            # eval = 0
            # eval = pint/10
            # if abs(dir - pdir) < math.pi/8 or abs(abs(dir - pdir) - math.pi/2) < math.pi/8:
            #     eval = pint
            if eval > best_eval:
                best_eval = eval
                best_direction = dir

        if best_eval > 10:
            best_direction += math.pi * 2

        self.dir = best_direction

        self.walk(lr_canvas, ud_canvas, best_direction)

    def pos_from_dir(self, direction):
        new_px = self.px + math.cos(direction) * self.dist
        new_py = self.py - math.sin(direction) * self.dist

        return new_px, new_py

    def walk(self, lr_canvas, ud_canvas, direction):
        w, h = lr_canvas.shape

        dx = math.cos(direction)*self.dist
        dy = -math.sin(direction)*self.dist

        new_px = (self.px + dx) % w
        new_py = (self.py + dy) % h

        line((self.px, self.py), (new_px, new_py), lr_canvas, ud_canvas, 3)

        self.px = new_px
        self.py = new_py


def main():
    GREYSCALE = True

    # WDITH = 500
    # HEIGHT = 500
    lr_canvas = np.zeros((500, 500), dtype=float)
    ud_canvas = np.zeros((500, 500), dtype=float)

    w, h = lr_canvas.shape

    ants = []
    for i in range(500):
        ants.append(Ant(random.randrange(0, w-1), random.randrange(0, h-1)))

    while True:
        for ant in ants:
            ant.scan(lr_canvas, ud_canvas)

        lr_canvas = lr_canvas * .99
        ud_canvas = ud_canvas * .99

        lr_canvas = cv2.GaussianBlur(lr_canvas, (3, 3), .6)
        ud_canvas = cv2.GaussianBlur(ud_canvas, (3, 3), .6)

        value = np.sqrt(np.square(lr_canvas) + np.square(ud_canvas))/5
        if not GREYSCALE:
            hue = (np.arctan2(lr_canvas, ud_canvas, dtype=float) + math.pi) * 255/(2*math.pi)
            saturation = np.zeros((500, 500), dtype=float)
            saturation.fill(.5)
            hsv_im = np.dstack((hue, saturation, value))
            hsv_im = np.float32(hsv_im)

            # cv2.imshow("ant farm", hsv_im)

            bgr_im = cv2.cvtColor(hsv_im, cv2.COLOR_HSV2RGB)
            cv2.imshow("ant farm", bgr_im)
        else:
            cv2.imshow("ant farm", value)
        key = cv2.waitKey(1)
        if key == ord('c'):
            GREYSCALE = not GREYSCALE
        if key == ord('q'):
            break


if __name__ == "__main__":
    main()