import numpy as np
import hcipy as hc
from scipy.signal import fftconvolve
from ..utils.misc import make_aperture_mask

def crosscorr(f1, f2, dim):
    '''
    Compute cross-correlation of two 2D fields f1 and f2 using fftconvolve.
    '''
    f1 = f1.reshape((dim,dim))
    f2 = f2.reshape((dim,dim))
    return fftconvolve(f1, np.conj(f2)[::-1,::-1], mode='full')

class OTF:
    

    def __init__(self, pupil_grid, pupil_modes, aperture, focal_modes = None):
        
        # self.pupil_diameter = pupil_diameter
        # self.dim = pupil_ndim
        # self.edim = pupil_ndim*2 - 1
        self.pupil_grid = pupil_grid #hc.make_pupil_grid(self.dim, diameter = self.pupil_diameter)
        self.dim = pupil_grid.shape[0]
        self.edim = self.dim*2 -1
        pixel_scale = pupil_grid.delta[0]
        self.egrid = hc.make_pupil_grid(self.edim, diameter = pixel_scale * self.edim)
        
        # self.egrid = hc.make_pupil_grid(self.edim, diameter = self.pupil_diameter * self.edim/self.dim)

        self.nmodes = len(pupil_modes)
        self.pupil_functions = np.array(pupil_modes * aperture[None,:], dtype=complex)
        self.unaberrated_pupil_functions = np.array(pupil_modes, dtype=complex)

        self.full_ccpupils = np.zeros((self.nmodes, self.nmodes, self.edim**2), dtype=complex)

        self.focal_modes = focal_modes

    def compute(self):
        
        for i in range(self.nmodes):
            for j in range(self.nmodes):
                self.full_ccpupils[i,j] = (crosscorr(self.pupil_functions[i], self.pupil_functions[j], self.dim).flatten())

        self.full_ccpupils = np.array(self.full_ccpupils)

    def aberrate(self, phase_screen):

        self.pupil_functions = np.array([self.unaberrated_pupil_functions[i] * np.exp(1j * phase_screen) for i in range(len(self.unaberrated_pupil_functions))])
        self.compute()
    
    def unaberrate(self):
        
        self.pupil_functions = np.array(self.unaberrated_pupil_functions).copy()
        self.compute()

    @classmethod
    def from_focal_modes(cls, focal_modes, focal_grid, pupil_grid, wavelength, 
                    #  ndim, focal_plane_resolution, 
                    #  focal_length, pupil_diameter, 
                    focal_length,
                     aperture,
                     optimize_focal_length = True):
        
        # focal_grid = hc.make_pupil_grid(ndim, diameter = focal_plane_resolution * ndim)
        # pupil_grid = hc.make_pupil_grid(ndim, diameter = pupil_diameter)
        propagator = hc.FraunhoferPropagator(pupil_grid, focal_grid, focal_length = focal_length)

        wf = hc.Wavefront(aperture, wavelength)
        
        if optimize_focal_length:

            def find_eta(focal_length):

                propagator.focal_length = focal_length
                wf_focal = propagator.forward(wf)

                _olPSF = np.abs(np.vdot(wf_focal.electric_field, wf_focal.electric_field))

                eta = 0.0
                for mode in focal_modes:
                    eta += np.abs(np.vdot(wf_focal.electric_field, mode))**2 / _olPSF
                return -eta
            
            from scipy.optimize import minimize
            out = minimize(find_eta, focal_length, method='Nelder-Mead')

            print("Optimized focal length at %.3e, with efficiency %.3f"%(out.x[0], -out.fun))
            focal_length = out.x[0]
            propagator.focal_length = focal_length
        
        pupil_modes = []
        for mode in focal_modes:
            pupil_mode = propagator.backward(hc.Wavefront(hc.Field(mode.flatten(), grid = focal_grid), wavelength=wavelength)).electric_field
            pupil_modes.append(pupil_mode)
        
        return cls(pupil_grid, pupil_modes, aperture, focal_modes = focal_modes)
    

    @classmethod
    def from_fiber_params(cls, ncore, nclad, rcore, wavelength, 
                    #  ndim, focal_plane_resolution, 
                    #  focal_length, pupil_diameter, 
                     focal_grid, pupil_grid, focal_length,
                     aperture,
                     optimize_focal_length = True):
        
        from PLsim.utils.LPmodes import compute_lpbases

        ndim = pupil_grid.shape[0]
        focal_plane_resolution = (focal_grid.x[-1] - focal_grid.x[0] + focal_grid.delta[0]) / ndim
        focal_modes, modenames = compute_lpbases(ncore, nclad, rcore, wavelength, 
                                            ndim, focal_plane_resolution)
        
        return cls.from_focal_modes(focal_modes, 
                                    focal_grid, pupil_grid,
                    wavelength,
                    focal_length,
                     aperture,
                     optimize_focal_length = optimize_focal_length)   

    @classmethod
    def from_aperture_masks(cls, mask_locs, mask_diameter, pupil_grid):

        pupil_modes = []
        for (x, y) in mask_locs:
            pupil_modes.append(make_aperture_mask(x, y, mask_diameter, pupil_grid).flatten())
        
        print(np.shape(pupil_modes))
        
        aperture = hc.Field(np.ones((pupil_grid.shape[0])**2), grid= pupil_grid)
        
        return cls(pupil_grid, pupil_modes, aperture)