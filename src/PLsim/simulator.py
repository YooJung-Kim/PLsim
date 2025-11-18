import numpy as np
import PLsim.lantern as pl
import PLsim.scene as Scene
# from PLsim.pic import pic

def create_index_mapping(source_list, target_list):
    return [target_list.index(item) for item in source_list]

class Device:

    def __init__(self, lantern_matrix, pic_device = None,
                 port_mapping = None, wavelengths=None,
                 verbose = True):
        """
        Device class for Photonic lantern, PIC, or both

        Parameters:
        -----------
        lantern_matrix : ndarray or list
            - 2D array: monochromatic lantern matrix
            - List of 2D arrays: chromatic lantern matrices
        pic_device : PIC, optional
            - Instance of the PIC class
        port_mapping : list, optional
            Which lantern ports connect to PIC inputs
        wavelengths : array_like, optional
            Required if chromatic matrices are provided
        """
        self.verbose = verbose
        self._validate_inputs(lantern_matrix, pic_device, port_mapping, wavelengths)

    def _validate_inputs(self, lantern_matrix, pic_device, port_mapping, wavelengths):

        
        if pic_device is not None:
            pic_matrix = pic_device.get_total_transfer_matrix()
            self.pic = pic_device
            ndim_pic_matrix = np.array(pic_matrix).ndim if pic_matrix is not None else 0
            assert ndim_pic_matrix in [2, 3], "pic_matrix must be a 2D array or a list of 2D arrays."
            self._has_pic = True
            self.port_mapping = port_mapping
            self.selected_ports = [p[0] for p in port_mapping] if port_mapping is not None else None
            self.input_pic_names = [p[1] for p in port_mapping] if port_mapping is not None else None
            # TODO: add assertions for port_mapping (inputs!)
        else:
            pic_matrix = None
            self._has_pic = False

        ndim_lantern_matrix = np.array(lantern_matrix).ndim
        len_wavelengths = len(wavelengths) if wavelengths is not None else 1

        # print(ndim_lantern_matrix, ndim_pic_matrix, len_wavelengths)

        assert ndim_lantern_matrix in [2, 3], "lantern_matrix must be a 2D array or a list of 2D arrays."

        # chromatic sim
        if len_wavelengths > 1:
            assert len(lantern_matrix) == len(wavelengths), \
                "If chromatic matrices are provided, they must match in length with wavelengths."
            if self._has_pic:
                assert len(pic_matrix) == len(wavelengths), \
                    "If chromatic matrices are provided, they must match in length with wavelengths."
            self._is_chromatic = True
            self.lantern_matrices = np.array(lantern_matrix)
            self.pic_matrices = np.array(pic_matrix) if pic_matrix is not None else None
            self.wavelengths = np.array(wavelengths)

        # monochromatic sim
        else:
            assert ndim_lantern_matrix  == 2, \
                "If monochromatic matrices are provided, they must be 2D arrays."
            if self._has_pic:
                assert ndim_pic_matrix == 2, \
                    "If monochromatic matrices are provided, they must be 2D arrays."
            self._is_chromatic = False

            self.lantern_matrices = np.array([lantern_matrix])
            self.pic_matrices = np.array([pic_matrix]) if pic_matrix is not None else None
            self.wavelengths = [None]

        if self.verbose:
            print("Device initialized with:")
            print(f"  Lantern matrices: {self.lantern_matrices.shape}")
            print(f"  PIC matrices: {self.pic_matrices.shape if self.pic_matrices is not None else None}")
            print(f"  Wavelengths: {self.wavelengths}")

        return
    
    def update_pic_matrix(self, **kwargs):
        if self._has_pic:
            self.pic.update_parameters(**kwargs)
            assert self._is_chromatic == False, "Updating PIC parameters for chromatic simulations is not supported yet."
            self.pic_matrices = np.array([self.pic.get_total_transfer_matrix()])
            if self.verbose:
                print("PIC parameters updated and matrices recalculated.")
        else:
            raise ValueError("No PIC device to update.")
    
    def calculate_output_intensities(self, overlaps, weights = None):
        '''
        Calculate output intensities for given overlaps and optional weights (for simulating extended objects of arbitrary shape).

        Parameters:
        ----------
        overlaps : ndarray
            Overlap matrices computed from OverlapCalculator. can be 1D, 2D, or 3D array (the last dimension is the number of lantern input ports^2)
        weights : ndarray, optional
            Weights when overlaps are in grids and the user wants to simulate extended objects as weighted sum of point sources.
            The first dimension should be n_wavelengths, and the second and third dimensions should match the first two dimensions of overlaps.
        Returns:
        -------
        ndarray
            Output intensities. If weights are not provided, the shape will be (n_wavelengths, n_overlaps, n_ports).
            If weights are provided, the shape will be (n_wavelengths, n_ports).
        '''

        if self._has_pic:
            
            # simulating PL + PIC
            
            spectra = []
            for wavind in range(len(self.wavelengths)):

                lantern_matrix = self.lantern_matrices[wavind]
                pic_matrix = self.pic_matrices[wavind]

                # calculate mutual intensities
                mutual_intens = np.array([pl.compute_mutual_intensities(lantern_matrix, freq_overlaps) for freq_overlaps in overlaps.reshape(-1, lantern_matrix.shape[0], lantern_matrix.shape[1])])

                # calculate total output intensity
                output_intens = np.array([pl.calculate_total_output_intensity(pic_matrix, mutual_inten[np.ix_(self.selected_ports, self.selected_ports)]) for mutual_inten in mutual_intens])

                spectra.append(output_intens)
            
            spectra = np.array(spectra) 

            if weights is None:
                return spectra
            else:
                reshaped_weights = weights.reshape((weights.shape[0], -1))
                weighted_spectra = np.sum(spectra * reshaped_weights[:,:,None], axis=1)
                return weighted_spectra

        else:

            # simulating only the lantern.
            spectra = []
            for wavind in range(len(self.wavelengths)):

                lantern_matrix = self.lantern_matrices[wavind]
                mutual_intens = np.array([pl.compute_mutual_intensities(lantern_matrix, freq_overlaps) for freq_overlaps in overlaps.reshape(-1, lantern_matrix.shape[0], lantern_matrix.shape[1])])
                intens = np.abs(np.array([np.diag(mutual_intens) for mutual_intens in mutual_intens]))

                spectra.append(intens)
            spectra = np.array(spectra) #.reshape(len(self.wavelengths), *overlaps.shape[:2], -1)

            # outputs are in (n_wav, n_overlaps, n_ports)

            if weights is None:
                return spectra
            else:
                reshaped_weights = weights.reshape((weights.shape[0], -1))
                weighted_spectra = np.sum(spectra * reshaped_weights[:,:,None], axis=1)
                return weighted_spectra

class OverlapCalculator:
    # Overlap calculator calculates cross-correlation of pupil functions weighted by scene mutual intensities.
    # not only PLPMs, but this applies to any apertures; e.g., pupil-remapping aperture masking interferometry.
    # thus this is defined generically here.

    def __init__(self, dim = 385, telescope_diameter = 10, wavelength = 1.55e-6):
        
        self.scene = Scene.Scene(dim=dim, wavelength=wavelength, diameter=telescope_diameter)
        self.overlap0 = self.compute_overlap(0, 0)
        

    def compute_overlap(self, ax, ay):
        """
        Compute the overlap between the scene and the lantern modes at given angular coordinates.

        Parameters:
        ----------
        ax : float
            Angular coordinate in radians (x-axis)
        ay : float
            Angular coordinate in radians (y-axis)

        Returns:
        -------
        ndarray
            Overlap matrix for the given angular coordinates.
        """
        J = self.scene.J_point(ax, ay)
        return (self.prop.full_ccpupils @ J)

    def compute_overlap_grid_x(self, xmax, ngrid, xoffset = 0, yoffset = 0):
        """
        Compute the overlap grid for a given x-coordinate range and number of grid points.

        Parameters:
        ----------
        xmax : float
            Maximum x-coordinate in radians
        ngrid : int
            Number of grid points

        Returns:
        -------
        ndarray
            Overlap grid for the given x-coordinate range and number of grid points.
        """
        ax_grid = np.linspace(-xmax, xmax, ngrid) + xoffset
        ay = 0.0 + yoffset
        overlaps = [self.compute_overlap(ax, ay) for ax in ax_grid]
        overlaps = np.array(overlaps).reshape(ngrid, -1)
        return overlaps

    def compute_overlap_grid_y(self, ymax, ngrid, xoffset = 0, yoffset = 0):
        """
        Compute the overlap grid for a given y-coordinate range and number of grid points.

        Parameters:
        ----------
        ymax : float
            Maximum y-coordinate in radians
        ngrid : int
            Number of grid points

        Returns:
        -------
        ndarray
            Overlap grid for the given y-coordinate range and number of grid points.
        """
        ax = 0.0 + xoffset
        ay_grid = np.linspace(-ymax, ymax, ngrid) + yoffset
        overlaps = [self.compute_overlap(ax, ay) for ay in ay_grid]
        overlaps = np.array(overlaps).reshape(ngrid, -1)
        return overlaps
    
    def compute_overlap_grid_2d(self, fov, ngrid, xoffset = 0, yoffset = 0):
        """
        Compute the overlap grid for a given field of view and number of grid points.

        Parameters:
        ----------
        fov : float
            Field of view in radians
        ngrid : int
            Number of grid points

        Returns:
        -------
        ndarray
            Overlap grid for the given field of view and number of grid points.
        """

        ax_grid = np.linspace(-fov/2, fov/2, ngrid) + xoffset
        ay_grid = np.linspace(-fov/2, fov/2, ngrid) + yoffset

        axs, ays = np.meshgrid(ax_grid, ay_grid)

        overlaps = [self.compute_overlap(ax, ay) for ax, ay in zip(axs.ravel(), ays.ravel())]
        overlaps = np.array(overlaps).reshape(ngrid, ngrid, -1)

        self.xs = axs
        self.ys = ays
        self.overlaps = overlaps

        return overlaps
    
    def compute_overlap_disk(self, radius, ellip=1):
        J = self.scene.J_disk(radius, ellip)
        return (self.prop.full_ccpupils @ J)
    




class PLOverlapCalculator(OverlapCalculator):
    # this is specific to photonic lanterns, which use PLprop

    def __init__(self, dim = 385, telescope_diameter = 10, wavelength = 1.55e-6, focal_plane_resolution = 0.5e-6,
                 nclad = 1.444, njack = 1.444 - 5.5e-3, rclad = 10e-6,
                 focal_length = None,
                 skip_pupil_bases_calc = False):
        """
        Initialize the PLOverlapCalculator with parameters for the lantern and scene.

        Parameters:
        ----------
        dim : int
            Dimension of the grid (default: 385)
        telescope_diameter : float
            Diameter of the telescope in meters (default: 10)
        wavelength : float
            Wavelength of light in meters (default: 1.55e-6)
        focal_plane_resolution : float
            Resolution of the focal plane in meters (default: 0.5e-6)
        nclad : float
            Refractive index of the cladding (default: 1.444)
        njack : float
            Refractive index of the jacket (default: 1.444 - 5.5e-3)
        rclad : float
            Radius of the cladding in meters (default: 10e-6)
        """

        # self.scene = Scene.Scene(dim=dim, wavelength=wavelength, diameter=telescope_diameter)
        
        self.prop = pl.PLprop(dim=dim, wavelength=wavelength, telescope_diameter=telescope_diameter,
                                 focal_plane_resolution=focal_plane_resolution)

        self.prop.specify_PL_parameters(nclad, njack, rclad)
        self.prop.calculate_lpbases()
        if focal_length is not None:
            self.prop.adjust_focal_length(focal_length)
        else:
            self.prop.focal_length_optimization()
        self.prop.compute_coupling_efficiency()
        print("Coupling efficiency:", self.prop.efficiency)

        if not skip_pupil_bases_calc:
            self.prop.calculate_pupil_bases()
            self.prop.calculate_ccpupil_bases()
            self.prop.unaberrated_u0s_pupil = self.prop.u0s_pupil.copy()
            self.prop.unaberrated_full_ccpupils = self.prop.full_ccpupils.copy()

        super().__init__(dim=dim, telescope_diameter=telescope_diameter, wavelength=wavelength)

    def aberrate_zernike(self, amplitudes):
        """
        Apply Zernike aberrations.

        Parameters:
        ----------
        amplitudes : list or ndarray
            List of Zernike amplitudes to apply.
        """
        assert len(amplitudes) == len(self.prop.zernike_maps), "Length of amplitudes must match number of Zernike modes prepared."
        phase_screen = np.zeros(self.prop.dim * self.prop.dim)
        for index, amplitude in enumerate(amplitudes):
            phase_screen += amplitude * self.prop.zernike_maps[index].flatten()
        
        self.aberrate(phase_screen)

    def aberrate(self, phase_screen):
        
        self.prop.u0s_pupil = np.array([self.prop.unaberrated_u0s_pupil[i] * np.exp(1j * phase_screen) for i in range(len(self.prop.unaberrated_u0s_pupil))])
        self.prop.calculate_ccpupil_bases()
    
    def unaberrate(self):
        self.prop.u0s_pupil = self.prop.unaberrated_u0s_pupil.copy()
        self.prop.full_ccpupils = self.prop.unaberrated_full_ccpupils.copy()


class AMIOverlapCalculator(OverlapCalculator):
    # this is specific to aperture masking interferometry, which uses AMIprop

    def __init__(self, positions, radii, dim = 385, wavelength = 1.55e-6, telescope_diameter = 10):
        """
        Initialize the AMIOverlapCalculator with parameters for the lantern and scene.

        Parameters:
        ----------
        dim : int
            Dimension of the grid (default: 385)
        telescope_diameter : float
            Diameter of the telescope in meters (default: 10)
        wavelength : float
            Wavelength of light in meters (default: 1.55e-6)
        """

        # self.scene = Scene.Scene(dim=dim, wavelength=wavelength, diameter=telescope_diameter)
        
        self.prop = pl.AMIprop(dim=dim, wavelength=wavelength, telescope_diameter=telescope_diameter)

        self.prop.make_apertures(positions, radii)
        self.prop.calculate_ccpupil_bases()

        super().__init__(dim=dim, telescope_diameter=telescope_diameter, wavelength=wavelength)


