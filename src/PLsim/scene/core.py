
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
        
    def J_gauss(self, fwhm, ax, ay):
        """
        Mutual Intensity of Gaussian Source.
        FWHM: Full-Width Half-Maximum in radians.
        ax, ay: Angular coordinates in radians.
        """
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        exponent = -2 * (np.pi**2) * (sigma**2) * (self.u**2 + self.v**2)
        envelope = np.exp(exponent)
        
        # Reuse J_point logic for phase shifts
        shifts = self.J_point(ax, ay)
        
        # Combine
        if shifts.ndim == 2:
            return envelope[:, np.newaxis] * shifts
        else:
            return envelope * shifts
        
# class SceneProjector:

#     def __init__(self, otf, scene:Scene):
#         self.otf = otf
#         self.scene = scene
    
#     def compute_point(self, ax, ay):
#         J = self.scene.J_point(ax, ay)  # (N_pixels,) or (N_pixels, N_points)
        
#         # Reshape J for broadcasting if needed
#         if J.ndim == 1:
#             J = J[:, np.newaxis]  # (N_pixels, 1)
        
#         # Compute overlap: (N_modes, N_modes, N_points)
#         overlap = np.tensordot(self.otf.full_ccpupils, J, axes=([2], [0]))
        
#         return overlap  # (N_modes, N_modes, N_points)
    
#     def compute_point_grid(self, fov, ngrid, xoffset=0, yoffset=0):

#         ax_grid = np.linspace(-fov/2, fov/2, ngrid) + xoffset
#         ay_grid = np.linspace(-fov/2, fov/2, ngrid) + yoffset

#         axs, ays = np.meshgrid(ax_grid, ay_grid)
#         axs_flat = axs.ravel()
#         ays_flat = ays.ravel()

#         J_stack = self.scene.J_point(axs_flat, ays_flat)  # (N_pixels, N_points)

#         raw_overlaps = np.tensordot(self.otf.full_ccpupils, J_stack, axes=([2], [0]))  # (N_modes, N_modes, N_points)

#         overlap_grid = raw_overlaps.reshape(self.otf.nmodes, self.otf.nmodes, ngrid, ngrid)
#         # overlap_grid = overlap_grid.transpose(2,3,0,1)  # (ngrid, ngrid, N_modes, N_modes)

#         return overlap_grid # (N_modes, N_modes, ngrid, ngrid)
    
#     def compute_disk(self, radius, ax, ay):
#         J = self.scene.J_disk(radius, ax, ay)  # (N_pixels,) or (N_pixels, N_points)
        
#         # Reshape J for broadcasting if needed
#         if J.ndim == 1:
#             J = J[:, np.newaxis]  # (N_pixels, 1)
        
#         # Compute overlap: (N_modes, N_modes, N_points)
#         overlap = np.tensordot(self.otf.full_ccpupils, J, axes=([2], [0]))
        
#         return overlap  # (N_modes, N_modes, N_points)

#     def compute_disk_grid(self, fov, ngrid, radius, xoffset=0, yoffset=0):
#         """
#         Computes the system response to a disk of specific radius scanning the grid.
#         """

#         ax_grid = np.linspace(-fov/2, fov/2, ngrid) + xoffset
#         ay_grid = np.linspace(-fov/2, fov/2, ngrid) + yoffset
#         axs, ays = np.meshgrid(ax_grid, ay_grid)

#         axs = axs.ravel()
#         ays = ays.ravel()
        
#         J_stack = self.scene.J_disk(radius, axs, ays)

#         # 3. Compute Overlaps (Tensor Contraction)
#         # (N_modes, N_modes, N_pixels) @ (N_pixels, N_points)
#         raw_overlaps = np.tensordot(self.otf.full_ccpupils, J_stack, axes=([2], [0]))

#         # 4. Reshape
#         overlap_grid = raw_overlaps.reshape(self.otf.nmodes, self.otf.nmodes, ngrid, ngrid)
#         return overlap_grid # (N_modes, N_modes, ngrid, ngrid)
    

class SceneProjector:

    def __init__(self, otf, scene):
        """
        Engine that projects Scene Mutual Intensities onto the OTF Mode Basis.
        """
        self.otf = otf
        self.scene = scene

    def compute_point(self, ax, ay):
        """Computes response for specific point source(s)."""
        return self._compute_projection(self.scene.J_point, ax, ay)

    def compute_disk(self, radius, ax, ay, ellip=1):
        """Computes response for specific disk source(s)."""
        # wrap the radius and ellip argument
        generator = lambda x, y: self.scene.J_disk(radius, x, y, ellip = ellip)
        return self._compute_projection(generator, ax, ay)
    
    def compute_gauss(self, fwhm, ax, ay):
        """Computes response for specific Gaussian source(s)."""
        # wrap the fwhm argument
        generator = lambda x, y: self.scene.J_gauss(fwhm, x, y)
        return self._compute_projection(generator, ax, ay)

    def compute_point_grid(self, fov, ngrid, xoffset=0, yoffset=0, update_cache = True):
        """Computes response grid for point sources (Impulse Response)."""
        point_grid = self._compute_grid_projection(
            self.scene.J_point, fov, ngrid, xoffset, yoffset
        )
        if update_cache:
            self.point_grid = point_grid
        return point_grid

    def compute_disk_grid(self, fov, ngrid, radius, xoffset=0, yoffset=0, ellip=1, update_cache = True):
        """Computes response grid for disk sources (Sensitivity Map)."""
        generator = lambda x, y: self.scene.J_disk(radius, x, y, ellip=ellip)
        disk_grid = self._compute_grid_projection(
            generator, fov, ngrid, xoffset, yoffset
        )
        if update_cache:
            self.disk_grid = disk_grid
        return disk_grid

    def compute_gauss_grid(self, fov, ngrid, fwhm, xoffset=0, yoffset=0, update_cache = True):
        """Computes response grid for Gaussian sources."""
        generator = lambda x, y: self.scene.J_gauss(fwhm, x, y)
        gauss_grid = self._compute_grid_projection(
            generator, fov, ngrid, xoffset, yoffset
        )
        if update_cache:
            self.gauss_grid = gauss_grid
        return gauss_grid
    
    def compute_scene_from_image(self, image, grid_type = 'point'):
        
        if grid_type == 'point':
            # assert image.shape == self.point_grid.shape[2:], "Image shape must match precomputed grid shape."
            grid = self.point_grid
        elif grid_type == 'disk':
            # assert image.shape == self.disk_grid.shape[2:], "Image shape must match precomputed grid shape."
            grid = self.disk_grid
        elif grid_type == 'gauss':
            # assert image.shape == self.gauss_grid.shape[2:], "Image shape must match precomputed grid shape."
            grid = self.gauss_grid
        else:
            raise ValueError("grid_type must be 'point', 'disk', or 'gauss'.")
        
        # Weighted sum over the spatial grid
        # grid: (N_modes, N_modes, Ny, Nx)
        # image: (Ny, Nx)
        # scene_response = np.tensordot(grid, image, axes=([2, 3], [0, 1]))  # (N_modes, N_modes)
        
        # Case 1: Single Image (Y, X)
        if image.ndim == 2:
            if image.shape != grid.shape[-2:]:
                raise ValueError(f"Image shape {image.shape} mismatches grid {grid.shape[-2:]}")
            
            # Simple contraction
            # (M, M, Y, X) @ (Y, X) -> (M, M)
            return np.tensordot(grid, image, axes=([-2, -1], [0, 1]))

        # Case 2: Image Stack (W, Y, X)
        elif image.ndim == 3:
            if image.shape[1:] != grid.shape[-2:]:
                raise ValueError(f"Image spatial dims {image.shape[1:]} mismatch grid {grid.shape[-2:]}")

            # Sub-case 2A: One Grid, Many Images (Broadcasting)
            if grid.ndim == 4:
                # Grid:  (M, M, Y, X)
                # Image: (W, Y, X)
                raw = np.tensordot(grid, image, axes=([2, 3], [1, 2]))
                
                # Transpose to (W, M, M) to match standard format
                return raw.transpose(2, 0, 1)        
        
        # return scene_response
        
    # --- Internal funcs ---

    def _compute_projection(self, J_func, ax, ay):
        """
        Helper: Generates J from coordinates and performs contraction.
        """
        # 1. Call the Scene Logic 
        J = J_func(ax, ay)  # (N_pixels,) or (N_pixels, N_points)
        
        # 2. Safety Reshape for single points
        if J.ndim == 1:
            J = J[:, np.newaxis]  # (N_pixels, 1)
        
        # 3. Core Math: Tensor Contraction
        # (N_modes, N_modes, N_pixels) @ (N_pixels, N_points)
        return np.tensordot(self.otf.full_ccpupils, J, axes=([2], [0]))

    def _compute_grid_projection(self, J_func, fov, ngrid, xoffset, yoffset):
        """
        Helper: Generates coordinates, calls projection, and reshapes to grid.
        """
        # 1. Generate Coordinate Mesh
        ax_vec = np.linspace(-fov/2, fov/2, ngrid) + xoffset
        ay_vec = np.linspace(-fov/2, fov/2, ngrid) + yoffset
        axs, ays = np.meshgrid(ax_vec, ay_vec)
        
        # 2. Compute Flat Projection (Reuse the logic above!)
        # We flatten inputs: (N_grid*N_grid,)
        raw_overlaps = self._compute_projection(J_func, axs.ravel(), ays.ravel())
        
        # 3. Reshape to Final Grid Dimensions
        # Raw: (N_modes, N_modes, N_points) -> Target: (N_modes, N_modes, Ny, Nx)
        return raw_overlaps.reshape(self.otf.nmodes, self.otf.nmodes, ngrid, ngrid)