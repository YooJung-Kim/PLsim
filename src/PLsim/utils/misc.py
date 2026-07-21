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

def generate_focal_pixel_modes(ndim, undersample_factor):
    '''
    Make pixel modes in the focal plane, with undersampling factor
    '''
    out = np.zeros((ndim//undersample_factor, ndim//undersample_factor, ndim, ndim))
    for i0,i in enumerate(range(0, ndim, undersample_factor)):
        for j0,j in enumerate(range(0, ndim, undersample_factor)):
            out[i0,j0, i:i+undersample_factor, j:j+undersample_factor] = 1
    out /= undersample_factor
    return out