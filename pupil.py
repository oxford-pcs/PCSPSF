import numpy as np

def circle(x_c, y_c, n_side, rad):
    y, x = np.ogrid[-x_c:n_side-x_c, -y_c:n_side-y_c]
    mask = x*x + y*y <= (rad)*(rad)

    array = np.zeros((n_side, n_side))
    array[mask] = 1

    return array
