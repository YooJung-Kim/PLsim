
import numpy as np
import hcipy as hc

class Scene:

    def __init__(self, pupil_grid, ref_wavelength): #dim = 385, wavelength = 1.55e-6, diameter = 10):
        
        self.grid = pupil_grid
        self.dim = pupil_grid.shape[0]
        self.edim = self.dim*2 - 1
        pixel_scale = pupil_grid.delta[0]
        self.egrid = hc.make_pupil_grid(self.edim, diameter = pixel_scale * self.edim)
        self.wavelength = ref_wavelength
        self.egrid_uv = hc.make_pupil_grid(self.edim, diameter = pixel_scale * self.edim/self.wavelength)

        self.u = self.egrid_uv.x
        self.v = self.egrid_uv.y

    def J_point(self, ax, ay):
        """
        Computes Mutual Intensity J for point sources.
        
        Parameters
        ----------
        ax, ay : float or ndarray
            Angular coordinates in radians.
            - If scalar: returns (N_pixels,)
            - If array (M,): returns (N_pixels, M)
        """
        # Vectorized Phase Calculation
        # We use standard broadcasting: (N, 1) * (1, M) -> (N, M)
        
        # Ensure ax/ay are at least 1D for consistent broadcasting
        ax = np.atleast_1d(ax)
        ay = np.atleast_1d(ay)
        
        # 1. Reshape UV coords to column vectors (N_pixels, 1)
        u_col = self.u[:, np.newaxis]
        v_col = self.v[:, np.newaxis]
        
        # 2. Reshape Angles to row vectors (1, N_points)
        ax_row = ax[np.newaxis, :]
        ay_row = ay[np.newaxis, :]
        
        # 3. Compute Phase
        # Result shape: (N_pixels, N_points)
        phase = -2 * np.pi * (u_col * ax_row + v_col * ay_row)
        
        J = np.exp(1j * phase)
        
        # Squeeze to return simple 1D array if input was scalar
        return J.squeeze()
    
        # '''
        # ax, ay: angular coordinate in radians
        # '''
        # return np.exp(-2*np.pi*1j*(self.egrid_uv.x * ax + self.egrid_uv.y * ay))
    
        # # return (1/(1+c)) * (1 + c*np.exp(2*np.pi*1j/self.wavelength * (delx * ax+ dely * ay)))
       
    # def J_disk(self, rad, ellip = 1):
        
    #     '''
    #     Mutual intensities of a circular disk
    #     ra : radius of the disk (radians)
    #     '''
    #     from scipy.special import jv

    #     #const = np.pi*a**2/(lam*dist)**2 
    #     val = 2*jv(1, 2*np.pi*rad * np.sqrt(self.egrid_uv.x**2+(self.egrid_uv.y*ellip)**2)) / (2*np.pi*rad * np.sqrt(self.egrid_uv.x**2+(self.egrid_uv.y*ellip)**2))
    #     # if (val*0 != 0): val = 1
    #     return val
    
    def J_disk(self, radius, ax, ay, ellip=1):

        from scipy.special import j1
        # 1. Calculate the Jinc Envelope (The Shape)
        # This depends only on the instrument (u,v) and source size (radius)
        # It is independent of position (ax, ay)
        rho = np.sqrt(self.u**2 + (self.v * ellip)**2)
        arg = 2 * np.pi * radius * rho
        
        # Safe Divide-by-Zero (Limit x->0 is 1.0)
        envelope = np.ones_like(arg)
        mask = arg > 1e-9
        envelope[mask] = 2 * j1(arg[mask]) / arg[mask]
        
        # 2. Calculate the Phase Ramps (The Positions)
        # We REUSE the vectorized logic from J_point!
        # Returns shape (N_pixels, N_points)
        shifts = self.J_point(ax, ay)
        
        # 3. Combine (Broadcasting)
        # Envelope: (N_pixels,)
        # Shifts:   (N_pixels, N_points)
        if shifts.ndim == 2:
            # Broadcast envelope column-wise
            return envelope[:, np.newaxis] * shifts
        else:
            # Simple element-wise multiplication for single point
            return envelope * shifts
        
    
        
class OverlapCalculator:

    def __init__(self, otf, scene:Scene):
        self.otf = otf
        self.scene = scene
    
    def compute_overlap(self, ax, ay):
        J = self.scene.J_point(ax, ay)  # (N_pixels,) or (N_pixels, N_points)
        
        # Reshape J for broadcasting if needed
        if J.ndim == 1:
            J = J[:, np.newaxis]  # (N_pixels, 1)
        
        # Compute overlap: (N_modes, N_modes, N_points)
        overlap = np.tensordot(self.otf.full_ccpupils, J, axes=([2], [0]))
        
        return overlap  # (N_modes, N_modes, N_points)
    
    def compute_overlap_grid(self, fov, ngrid, xoffset=0, yoffset=0):

        ax_grid = np.linspace(-fov/2, fov/2, ngrid) + xoffset
        ay_grid = np.linspace(-fov/2, fov/2, ngrid) + yoffset

        axs, ays = np.meshgrid(ax_grid, ay_grid)
        axs_flat = axs.ravel()
        ays_flat = ays.ravel()

        J_stack = self.scene.J_point(axs_flat, ays_flat)  # (N_pixels, N_points)

        raw_overlaps = np.tensordot(self.otf.full_ccpupils, J_stack, axes=([2], [0]))  # (N_modes, N_modes, N_points)

        overlap_grid = raw_overlaps.reshape(self.otf.nmodes, self.otf.nmodes, ngrid, ngrid)
        # overlap_grid = overlap_grid.transpose(2,3,0,1)  # (ngrid, ngrid, N_modes, N_modes)

        return overlap_grid # (N_modes, N_modes, ngrid, ngrid)
    
    def compute_overlap_grid_disk(self, fov, ngrid, radius, xoffset=0, yoffset=0):
        """
        Computes the system response to a disk of specific radius scanning the grid.
        """

        ax_grid = np.linspace(-fov/2, fov/2, ngrid) + xoffset
        ay_grid = np.linspace(-fov/2, fov/2, ngrid) + yoffset
        axs, ays = np.meshgrid(ax_grid, ay_grid)

        axs = axs.ravel()
        ays = ays.ravel()
        
        J_stack = self.scene.J_disk(radius, axs, ays)

        # 3. Compute Overlaps (Tensor Contraction)
        # (N_modes, N_modes, N_pixels) @ (N_pixels, N_points)
        raw_overlaps = np.tensordot(self.otf.full_ccpupils, J_stack, axes=([2], [0]))

        # 4. Reshape
        overlap_grid = raw_overlaps.reshape(self.otf.nmodes, self.otf.nmodes, ngrid, ngrid)
        return overlap_grid # (N_modes, N_modes, ngrid, ngrid)
    

