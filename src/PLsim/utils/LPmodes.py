from lightbeam.LPmodes import lpfield, get_V, get_modes, get_b
import numpy as np
import hcipy as hc
from lightbeam.misc import normalize

def modestring(mode):
    prefix = 'LP%d%d' % (mode[0], mode[1])
    if mode[0] ==0: return prefix
    else:
        if mode[2] == 'cos': suffix = 'a'
        if mode[2] == 'sin': suffix = 'b'
        return prefix+suffix
    
def compute_lpbases(ncore, nclad, rcore, wavelength, 
                    ndim, focal_plane_resolution):
    ''' Compute LP mode bases on a given grid.
    '''

    focal_grid = hc.make_pupil_grid(ndim, diameter = focal_plane_resolution * ndim)
    xg = focal_grid.y.reshape((ndim, ndim))
    yg = focal_grid.x.reshape((ndim, ndim))
    k0 = 2.0 * np.pi / wavelength
    V = get_V(k0, rcore, ncore, nclad)
    _modes = get_modes(V)

    modes2 = []
    bs = []
    modes = []
    modenames = []
    for mo in _modes:
        if (mo[0] == 0): 
            modes2.append((mo[0], mo[1], 'cos'))
            bs.append(get_b(mo[0], mo[1], V))
        else:
            modes2.append((mo[0], mo[1], 'cos'))
            modes2.append((mo[0], mo[1], 'sin'))
            bs.append(get_b(mo[0], mo[1], V) * 1.01) # just to break cos/sin unambiguity. doesn't do anything to field calculation
            bs.append(get_b(mo[0], mo[1], V))
    # self.bs = bs
    bsort = np.flip(np.argsort(np.array(bs)))
    for i in range(len(bsort)):
        modes.append(modes2[bsort[i]]) # = (modes2)[bsort]
        modenames.append(modestring(modes2[bsort[i]]))
    
    print('Supported modes: ', modenames)
    print('Number of modes: ', len(modes))

    u0s = []
    for mo in modes:
        u0 = normalize(lpfield(xg, yg, mo[0], mo[1], rcore, wavelength, ncore, nclad, which=mo[2])).flatten()
        u0s.append(u0)
    return np.array(u0s), modenames