import numpy as np

def make_aperture_mask(xp, yp, radius, pupil_grid):
    '''
    Makes circular holes in the pupil plane
    '''
    xg,yg = pupil_grid.x, pupil_grid.y
    pupil = np.zeros_like(xg, dtype=complex)
    # pupil = np.zeros((len(xa0), len(xa0)), dtype=complex)
    pupil[(xg-xp)**2+(yg-yp)**2 < radius**2] = 1.0
    return pupil