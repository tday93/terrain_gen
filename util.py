import math


def dist_2d(x0, y0, x1, y1):
    return math.sqrt((x1 - x0)**2 + (y1 - y0)**2)


# distance between point (x0, y0) and the line passing through (x1, y1) and (x2, y2)

def dist_line_point(x0, y0, x1, y1, x2, y2):
    return math.abs( ((x2 - x1) * (y1 - y2)) - ((x1 - x0) * (y2 -y1))) / math.sqrt( ((x2 - x1)**2) + ((y2 - y1)**2))

