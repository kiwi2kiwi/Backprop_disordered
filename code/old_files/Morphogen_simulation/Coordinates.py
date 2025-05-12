import numpy as np

class Coordinate():
    def __init__(self, x, y, z):
        super(Coordinate, self).__init__()
        self.x = x
        self.y = y
        self.z = z

def distance_finder(one,two):
    a = np.array((one.x, one.y, one.z))  # first coordinates
    b = np.array((two.x, two.y, two.z))  # second coordinates
    return np.linalg.norm(a-b)

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)