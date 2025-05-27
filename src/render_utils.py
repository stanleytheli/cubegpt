import numpy as np

# Vector utility
def dot(a, b):
    """Scalar dot product of vectors a and b"""
    return np.sum(a * b)
def cross(a, b):
    """Cross product of vectors a and b"""
    return np.cross(a, b)
def normalized(a):
    """Normalize a vector"""
    return a / np.sqrt(dot(a, a))


class Color:
    BLACK = (0, 0, 0)
    GRAY = (200, 200, 200)
    WHITE = (245, 245, 245)
    RED = (180, 0, 0)
    GREEN = (0, 180, 0)
    BLUE = (0, 0, 180)
    YELLOW = (255, 240, 7)
    PURPLE = (128, 0, 128)
    ORANGE = (255, 128, 0)

