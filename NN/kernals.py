import numpy as np

# Wikipedia kernals
identity = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])

edge = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])

sharpen = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

box_blur_normalised = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
]) / 9

gaussian_blur_3x3 = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
]) / 16

gaussian_blur_5x5 = np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
]) / 256

unsharp_masking = np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 24],
    [6, 24, -476, 24, 6],
    [4, 16, 24, 16, 24],
    [1, 4, 6, 4, 1]
]) / -256

# My kernals
top_edge = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
])

left_edge = top_edge.T

bottom_edge = np.flip(top_edge, 0)

right_edge = bottom_edge.T