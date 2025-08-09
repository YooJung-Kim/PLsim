# get input from scene and aberration, then simulate lantern observables 
# (intensity and mutual intensity matrix)

import numpy as np
import hcipy as hc
from lightbeam.LPmodes import lpfield, get_V, get_modes, get_b
from lightbeam.misc import normalize, overlap
import lightbeam.zernike as zk
import glob, os
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.stats import unitary_group
import scipy.linalg

class PLprop:
    '''
    Compute LP modes, focal and pupil planes, and cross-correlations

    The default focal plane grid is set to match the beam propagation simulation:
    dimension of 385 and focal plane resolution of 0.5 um.
    
    Example usage:
    > plprop = PLprop()
    > plprop.specify_PL_parameters(nclad, njack, rclad)
    > plprop.calculate_lpbases()
    > plprop.focal_length_optimization() # optimizes focal length for maximum coupling efficiency
    > plprop.calculate_pupil_bases()
    > plprop.calculate_ccpupil_bases()

    '''

    def __init__(self, dim=385, focal_length=43, wavelength = 1.55e-6, telescope_diameter=10,
                 focal_plane_resolution=0.5e-6):

        self.telescope_diameter = telescope_diameter # m
        self.wavelength = wavelength
        self.dim = dim # pixels
        self.edim = dim*2 - 1

        self.focal_plane_resolution = focal_plane_resolution
        self.focal_plane_width = (self.dim) * self.focal_plane_resolution
        self.focal_length =  focal_length # m

        self.pupil_grid = hc.make_pupil_grid(self.dim, self.telescope_diameter)
        self.focal_grid = hc.make_pupil_grid(self.dim, self.focal_plane_width)
        self.egrid = hc.make_pupil_grid(self.edim, diameter = self.telescope_diameter * self.edim/self.dim)
        
        self.aperture = hc.circular_aperture(self.telescope_diameter)(self.pupil_grid)
        self.wf = hc.Wavefront(self.aperture, self.wavelength)
        self.wf.total_power = 1.0

        self.propagator = hc.FraunhoferPropagator(self.pupil_grid, self.focal_grid, focal_length = self.focal_length)
        self.wf_focal = self.propagator.forward(self.wf)

        self.nclad = self.njack = None
        self.rclad = None
        self.u0s = self.k0 = self.V = self.modes = None
        self.u0s_pupil = None
        self.efficiency = None

        xa1 = ya1 = np.linspace(-1, 1, self.dim)
        self.xg1, self.yg1 = np.meshgrid(xa1, ya1) 
        self.pupil_functions = None

        self.ecir = hc.make_circular_aperture(self.telescope_diameter * self.edim/self.dim)(self.egrid)

    def specify_PL_parameters(self, nclad, njack, rclad):

        self.nclad = nclad
        self.njack = njack
        self.rclad = rclad
        
    def propagate_field(self, field):

        wf_field = hc.Wavefront(hc.Field(field, grid= self.pupil_grid), wavelength = self.wavelength)
        focal_field = self.propagator.forward(wf_field).electric_field
        return focal_field
    
    def calculate_lpbases(self, mode_to_calculate=None):

        if self.nclad == None:
            raise Exception("PL parameters are not specified. use `specify_PL_parameters function` first")

        xg = self.focal_grid.y.reshape((self.dim, self.dim))
        yg = self.focal_grid.x.reshape((self.dim, self.dim))
        
        if mode_to_calculate == None:
            self.k0 = 2*np.pi / self.wavelength
            self.V = get_V(self.k0, self.rclad, self.nclad, self.njack)
            modes = get_modes(self.V)
            modes2 = []
            bs = []
            self.modes = []
            self.modenames = []
            for mo in modes:
                if (mo[0] == 0): 
                    modes2.append((mo[0], mo[1], 'cos'))
                    bs.append(get_b(mo[0], mo[1], self.V))
                else:
                    modes2.append((mo[0], mo[1], 'cos'))
                    modes2.append((mo[0], mo[1], 'sin'))
                    bs.append(get_b(mo[0], mo[1], self.V) * 1.01) # just to break cos/sin unambiguity. doesn't do anything to field calculation
                    bs.append(get_b(mo[0], mo[1], self.V))
            # self.bs = bs
            bsort = np.flip(np.argsort(np.array(bs)))
            for i in range(len(bsort)):
                self.modes.append(modes2[bsort[i]]) # = (modes2)[bsort]
                self.modenames.append(modestring(modes2[bsort[i]]))
            
            print('Available modes: ', str(self.modes))
            print('Calculated k0, V, modes')
        else:
            print('Calculating', mode_to_calculate)
            self.modes = mode_to_calculate
        u0s = []
        for mo in self.modes:
            u0 = normalize(lpfield(xg, yg, mo[0], mo[1], self.rclad, self.wavelength, self.nclad, self.njack, which=mo[2])).flatten()
            u0s.append(u0)
        self.u0s = u0s
        self.nmodes = len(u0s)
        print('LP modes stored in self.u0s')

    def adjust_focal_length(self, new_focal_length):
        self.propagator = hc.FraunhoferPropagator(self.pupil_grid, self.focal_grid, focal_length=new_focal_length)
        self.focal_length = new_focal_length
        self.wf_focal = self.propagator.forward(self.wf)

    def focal_length_optimization(self, initial_value = None):
        
        initial_focal_length = self.focal_length 
        if (initial_value != None): initial_focal_length = initial_value

        print('Start Focal length optimization')
        print('Initial focal length = %.3e' % (initial_focal_length))
        
        def find_eta(focal_length):

            self.adjust_focal_length(focal_length)
            _olPSF = overlap(self.wf_focal.electric_field, self.wf_focal.electric_field)
                
            eta = 0.0
            for u0 in self.u0s:
                nu = overlap(self.wf_focal.electric_field, u0)
                eta += nu**2 / _olPSF

            return -eta
    
        out = minimize(find_eta, self.focal_length, method='Nelder-Mead')

        self.adjust_focal_length(out.x[0])
        
        print('Optimal focal length at %.5e, with efficiency = %.3f' % (self.focal_length, -out.fun))
        self.efficiency = -out.fun

        return self.focal_length, -out.fun
    
    def calculate_pupil_bases(self):

        self.u0s_pupil = []
        for u0 in self.u0s:
            u0_pupil = self.propagator.backward(hc.Wavefront(hc.Field(u0.flatten(), grid=self.focal_grid),
                                                             wavelength = self.wavelength)).electric_field * self.aperture
            self.u0s_pupil.append(u0_pupil)
        self.u0s_pupil = np.array(self.u0s_pupil)
        print('Pupil functions calculated and stored in self.u0s_pupil')

    def calculate_ccpupil_bases(self):

        self.full_ccpupils = np.zeros((self.nmodes, self.nmodes, self.edim**2),
                                      dtype = np.complex_)
        for i in range(self.nmodes):
            for j in np.arange(self.nmodes):
                self.full_ccpupils[i,j] = (crosscorr(self.u0s_pupil[i], self.u0s_pupil[j], self.dim).flatten())

        # self.ccpupils = np.array(self.ccpupils)
        self.full_ccpupils = np.array(self.full_ccpupils)
        # self.ccnames = np.array(self.ccnames)
        print('Cross-correlated pupil plane LP modes stored in self.full_ccpupils')


def modestring(mode):
    prefix = 'LP%d%d' % (mode[0], mode[1])
    if mode[0] ==0: return prefix
    else:
        if mode[2] == 'cos': suffix = 'a'
        if mode[2] == 'sin': suffix = 'b'
        return prefix+suffix

def crosscorr(f1, f2, dim):
    f1 = f1.reshape((dim,dim))
    f2 = f2.reshape((dim,dim))
    return fftconvolve(f1, np.conj(f2)[::-1,::-1], mode='full')


def compute_mutual_intensities(matrix, full_freq_overlaps):
    """
    Compute mutual intensities given transfer matrix and frequency overlaps
    """

    mutual_intens = matrix @ full_freq_overlaps @ matrix.conj().T
    
    return mutual_intens

def square_flatten(arr):
    out_arr = []
    for i in range(len(arr)):
        for j in range(len(arr)):
            out_arr.append(arr[i] * np.conj(arr[j]))
    return np.array(out_arr)

def calculate_total_output_intensity(pic_mat, mutual_intens):
    n_output = len(pic_mat)
    output_intens = np.zeros(n_output, dtype=np.complex128)

    for i in range(n_output):
        output_intens[i] = square_flatten(pic_mat[i]) @ mutual_intens.flatten()
    return np.real(output_intens)


def generate_unitary_matrix(dim, seed=123456):
    """
    Generate a random unitary matrix of given dimension.
    """
    return unitary_group.rvs(dim, random_state=seed)


def generate_unitary_matrix_near_identity(dim=3, epsilon=0.1):
    """
    Generate a unitary matrix near identity using matrix exponential.
    
    Parameters:
    epsilon: controls how close to identity (smaller = closer)
    size: matrix dimension
    """
    # Create a random skew-Hermitian matrix (A = -A†)
    # First create a random complex matrix
    A = (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)) * epsilon
    
    # Make it skew-Hermitian: A = (A - A†) / 2
    A = (A - A.conj().T) / 2
    
    # The matrix exponential of a skew-Hermitian matrix is unitary
    U = scipy.linalg.expm(A)
    
    return U


def make_circular_aperture(xp, yp, radius, xa0):
    '''
    Makes circular holes in the pupil plane
    '''
    xg,yg = np.meshgrid(xa0,xa0)
    pupil = np.zeros((len(xa0), len(xa0)), dtype=np.complex_)
    pupil[(xg-xp)**2+(yg-yp)**2 < radius**2] = 1.0
    return pupil

class AMIprop:

    def __init__(self, dim=385, wavelength = 1.55e-6, telescope_diameter=10):

        self.telescope_diameter = telescope_diameter # m
        self.wavelength = wavelength
        self.dim = dim # pixels
        self.edim = dim*2 - 1

        # self.focal_plane_resolution = focal_plane_resolution
        # self.focal_plane_width = (self.dim) * self.focal_plane_resolution
        # self.focal_length =  focal_length # m

        self.pupil_grid = hc.make_pupil_grid(self.dim, self.telescope_diameter)
        # self.focal_grid = hc.make_pupil_grid(self.dim, self.focal_plane_width)
        self.egrid = hc.make_pupil_grid(self.edim, diameter = self.telescope_diameter * self.edim/self.dim)
        
        self.aperture = hc.circular_aperture(self.telescope_diameter)(self.pupil_grid)
        # self.wf = hc.Wavefront(self.aperture, self.wavelength)
        # self.wf.total_power = 1.0

        # self.propagator = hc.FraunhoferPropagator(self.pupil_grid, self.focal_grid, focal_length = self.focal_length)
        # self.wf_focal = self.propagator.forward(self.wf)

        xa1 = ya1 = np.linspace(-1, 1, self.dim)
        self.xg1, self.yg1 = np.meshgrid(xa1, ya1) 
        self.pupil_functions = None

        self.ecir = hc.make_circular_aperture(self.telescope_diameter * self.edim/self.dim)(self.egrid)

    def make_apertures(self, positions, radii):
        
        apertures = []
        for (x, y), r in zip(positions, radii):
            apertures.append(make_circular_aperture(x, y, r, self.pupil_grid.x[:self.dim]))
        self.pupil_functions = apertures
        self.nmodes = len(apertures)
        print('Apertures made and stored in self.pupil_functions')
        return apertures

    # def propagate_field(self, field):

    #     wf_field = hc.Wavefront(hc.Field(field, grid= self.pupil_grid), wavelength = self.wavelength)
    #     focal_field = self.propagator.forward(wf_field).electric_field
    #     return focal_field

    def calculate_ccpupil_bases(self):

        self.full_ccpupils = np.zeros((self.nmodes, self.nmodes, self.edim**2),
                                      dtype = np.complex_)
        for i in range(self.nmodes):
            for j in np.arange(self.nmodes):
                self.full_ccpupils[i,j] = (crosscorr(self.pupil_functions[i], self.pupil_functions[j], self.dim).flatten())

        # self.ccpupils = np.array(self.ccpupils)
        self.full_ccpupils = np.array(self.full_ccpupils)
        # self.ccnames = np.array(self.ccnames)
        print('Cross-correlated pupil plane modes stored in self.full_ccpupils')

