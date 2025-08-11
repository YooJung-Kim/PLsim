# compute pupil plane mutual coherence

import numpy as np
import hcipy as hc



def compute_fftmatrix(axis_len, telescope_diameter, wavelength, im_fov):
    ''' Fourier transform matrix for image to pupil plane mutual coherence.
    Using this, any mutual coherence function can be simulated from the image'''

    fftfactor = 2.0 * telescope_diameter / wavelength * (im_fov * 1e-3 / 206265.) * (np.floor(axis_len/2)/(np.floor(axis_len/2 + 1))) * (axis_len-1) / axis_len
    step1 = 1.0 / (axis_len - 1)
    step2 = step1 * fftfactor
    
    # Create coordinate arrays
    i_indices = np.arange(axis_len)
    j_indices = np.arange(axis_len)
    
    # Create meshgrids for all combinations
    iii, jjj = np.meshgrid(i_indices, j_indices, indexing='ij')
    xxx, yyy = np.meshgrid(i_indices, j_indices, indexing='ij')
    
    # Compute coordinates
    xx = -0.5 + step1 * iii.flatten()  # Shape: (axis_len^2,)
    yy = -0.5 + step1 * jjj.flatten()
    uu = -0.5 * fftfactor + step2 * xxx.flatten()
    vv = -0.5 * fftfactor + step2 * yyy.flatten()
    
    phase_matrix = 2.0 * np.pi * (uu[None, :] * xx[:, None] + vv[None, :] * yy[:, None])
    fftmatrix = np.exp(1j * phase_matrix)
    
    return fftmatrix


class Scene:

    def __init__(self, dim = 385, wavelength = 1.55e-6, diameter = 10):

        self.dim = dim
        self.edim = dim*2 - 1
        self.diameter = diameter
        self.wavelength = wavelength

        self.grid = hc.make_pupil_grid(self.dim, diameter = self.diameter)
        self.egrid = hc.make_pupil_grid(self.edim, diameter = self.diameter * self.edim/self.dim)

        # grid of spatial frequencies, in x/lambda, y/lambda
        self.egrid_uv = hc.make_pupil_grid(self.edim, diameter = self.diameter * self.edim/self.dim/self.wavelength)

    
    def J_point(self, ax, ay):
        '''
        ax, ay: angular coordinate in radians
        '''
        return np.exp(-2*np.pi*1j*(self.egrid_uv.x * ax + self.egrid_uv.y * ay))
    

    #     # Pre-compute for J_point optimization
    #     self._precompute_j_point_data()
    
    # def _precompute_j_point_data(self):
    #     """Pre-compute data for faster J_point computation."""
    #     self._x_grid_flat = self.egrid_uv.x.ravel()
    #     self._y_grid_flat = self.egrid_uv.y.ravel()
    #     self._grid_shape = self.egrid_uv.x.shape
    #     self._minus_2pi_j = -2*np.pi*1j


    # def J_point_fast(self, ax, ay):
    #     """Optimized version using pre-computed data."""
    #     return np.exp(self._minus_2pi_j * (self._x_grid_flat * ax + self._y_grid_flat * ay))#.reshape(self._grid_shape)
    
    # def J_point_batch(self, ax_array, ay_array):
    #     """
    #     Vectorized version for multiple points at once.
        
    #     Parameters:
    #     -----------
    #     ax_array, ay_array : array_like
    #         Arrays of angular coordinates in radians
            
    #     Returns:
    #     --------
    #     result : ndarray
    #         Array of shape (n_positions, grid_height, grid_width)
    #     """
    #     ax_array = np.asarray(ax_array).ravel()
    #     ay_array = np.asarray(ay_array).ravel()
        
    #     # For small numbers of positions, fall back to loop
    #     if len(ax_array) < 50:
    #         return np.array([self.J_point_fast(ax, ay) for ax, ay in zip(ax_array, ay_array)])
        
    #     # Broadcast for all positions: (n_pos, 1) * (1, n_grid) = (n_pos, n_grid)
    #     phase = self._minus_2pi_j * (
    #         self._x_grid_flat[None, :] * ax_array[:, None] + 
    #         self._y_grid_flat[None, :] * ay_array[:, None]
    #     )
        
    #     result = np.exp(phase)
        
    #     # Reshape to (n_positions, grid_height, grid_width)
    #     return result #.reshape(len(ax_array), *self._grid_shape)
        
