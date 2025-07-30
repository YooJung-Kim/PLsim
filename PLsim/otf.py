import numpy as np
import cv2
import matplotlib.pyplot as plt
import hcipy as hc
from astropy.io import fits
from scipy.signal import fftconvolve
from scipy.optimize import minimize
from scipy.stats import poisson, norm
import lightbeam.zernike as zk
from lightbeam.misc import overlap, normalize
from lightbeam.LPmodes import lpfield, get_V, get_modes, get_b

class OTF:

    
    def __init__(self, dim = 385, wavelength = 1.55e-6, diameter = 10, focal_ratio = 6, num_lamD = 5,
                 custom_pupil = None):

        self.dim = dim
        self.edim = dim*2 - 1
        self.diameter = diameter
        self.wavelength = wavelength

        self.grid = hc.make_pupil_grid(self.dim, diameter = self.diameter)
        self.egrid = hc.make_pupil_grid(self.edim, diameter = self.diameter * self.edim/self.dim)
        
        if custom_pupil == None:
            self.cir = hc.make_circular_aperture(diameter)(self.grid)
        else:
            self.cir = custom_pupil

        self.ecir = hc.make_circular_aperture(diameter * self.edim/self.dim)(self.egrid)
        
        self.focal_grid = hc.make_pupil_grid(self.dim, diameter = wavelength * focal_ratio *num_lamD)
        self.prop = hc.FraunhoferPropagator(self.grid, self.focal_grid, focal_ratio * diameter)

    def specify_PL_parameters(self, nclad, njack, rclad):

        self.nclad = nclad
        self.njack = njack
        self.rclad = rclad

    def calculate_lpbases(self, mode_to_calculate=None, xoffset=0, yoffset=0, verbose=True):

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
                    bs.append(get_b(mo[0], mo[1], self.V) * (1+1e-5)) # just to break cos/sin unambiguity.
                    bs.append(get_b(mo[0], mo[1], self.V))

            bsort = np.flip(np.argsort(np.array(bs)))

            for i in range(len(bsort)):
                self.modes.append(modes2[bsort[i]]) # = (modes2)[bsort]
                self.modenames.append(modestring(modes2[bsort[i]]))
            print('Available modes: ', (self.modenames))
            print('Calculated k0, V, modes')
        
        else:
            if verbose: print('Calculating', mode_to_calculate)
            self.modes = mode_to_calculate

        # calculate focal plane modes    
        u0s = []
        for mo in self.modes:
            u0 = normalize(lpfield(xg-xoffset, yg-yoffset, mo[0], mo[1], self.rclad, self.wavelength, self.nclad, self.njack, which=mo[2])).flatten()
            u0s.append(u0)
        self.u0s = u0s
        self.nmodes = len(u0s)
        if verbose: print('%d LP modes stored in self.u0s' % self.nmodes)

        # calculate pupil plane modes    
        pupil_u0s = []
        for u0 in self.u0s:

            wf_focal = hc.Wavefront(hc.Field(u0.flatten(),\
                                             grid = self.focal_grid),
                                    wavelength = self.wavelength)
            wf_pupil = self.prop.backward(wf_focal).electric_field * self.cir
            pupil_u0s.append(wf_pupil)
        self.pupil_u0s = pupil_u0s
        if verbose: print('Pupil plane LP modes stored in self.pupil_u0s')
    
        # calculate cross-correlated pupil plane modes
        self.ccpupils = []
        self.ccnames = []
        for i in range(self.nmodes):
            self.ccpupils.append(self.crosscorr(self.pupil_u0s[i], self.pupil_u0s[i], self.dim).flatten())
            self.ccnames.append('%s - %s' % (self.modenames[i], self.modenames[i]))
        for i in range(self.nmodes):
            for j in np.arange(i+1, self.nmodes):
                self.ccpupils.append(self.crosscorr(self.pupil_u0s[i], self.pupil_u0s[j], self.dim).flatten())
                self.ccnames.append('%s - %s' % (self.modenames[i], self.modenames[j]))
        
        self.full_ccpupils = np.zeros((self.nmodes, self.nmodes, self.edim**2),
                                      dtype = np.complex_)
        for i in range(self.nmodes):
            for j in np.arange(self.nmodes):
                self.full_ccpupils[i,j] = (self.crosscorr(self.pupil_u0s[i], self.pupil_u0s[j], self.dim).flatten())

        self.ccpupils = np.array(self.ccpupils)
        self.full_ccpupils = np.array(self.full_ccpupils)
        self.ccnames = np.array(self.ccnames)
        if verbose: print('Cross-correlated pupil plane LP modes stored in self.ccpupils')

    def calculate_matrix_ccpupils(self, matrix):

        self.matrix_pupils = matrix @ self.pupil_u0s
        # calculate cross-correlated pupil plane modes
        self.matrix_ccpupils = []

        for i in range(self.nmodes):
            self.matrix_ccpupils.append(self.crosscorr(self.matrix_pupils[i], self.matrix_pupils[i], self.dim).flatten())
        for i in range(self.nmodes):
            for j in np.arange(i+1, self.nmodes):
                self.matrix_ccpupils.append(self.crosscorr(self.matrix_pupils[i], self.matrix_pupils[j], self.dim).flatten())
        self.matrix_ccpupils = np.array(self.matrix_ccpupils)
        
        self.full_matrix_ccpupils = np.zeros((self.nmodes, self.nmodes, self.edim**2),
                                      dtype = np.complex_)
        for i in range(self.nmodes):
            for j in np.arange(self.nmodes):
                self.full_matrix_ccpupils[i,j] = (self.crosscorr(self.matrix_pupils[i], self.matrix_pupils[j], self.dim).flatten())

    def calculate_probed_ccpupils(self, probe, calculate_full = True):
        # calculate cross-correlated pupil plane modes

        self.probed_ccpupils = []

        if not calculate_full:
            for i in range(self.nmodes):
                self.probed_ccpupils.append(self.crosscorr(self.pupil_u0s[i] * np.exp(1j*probe), self.pupil_u0s[i]* np.exp(1j*probe), self.dim).flatten())
            for i in range(self.nmodes):
                for j in np.arange(i+1, self.nmodes):
                    self.probed_ccpupils.append(self.crosscorr(self.pupil_u0s[i]* np.exp(1j*probe), self.pupil_u0s[j]* np.exp(1j*probe), self.dim).flatten())
                
        self.probed_ccpupils = np.array(self.probed_ccpupils)

        if calculate_full:
            self.probed_full_ccpupils = np.zeros((self.nmodes, self.nmodes, self.edim**2),
                                        dtype = np.complex_)
            for i in range(self.nmodes):
                for j in np.arange(self.nmodes):
                    self.probed_full_ccpupils[i,j] = (self.crosscorr(self.pupil_u0s[i]* np.exp(1j*probe), self.pupil_u0s[j]* np.exp(1j*probe), self.dim).flatten())

    @staticmethod
    def crosscorr(f1, f2, dim):
        f1 = f1.reshape((dim,dim))
        f2 = f2.reshape((dim,dim))
        return fftconvolve(f1, np.conj(f2)[::-1,::-1], mode='full')
    
    
def modestring(mode):
    prefix = 'LP%d%d' % (mode[0], mode[1])
    if mode[0] ==0: return prefix
    else:
        if mode[2] == 'cos': suffix = 'a'
        if mode[2] == 'sin': suffix = 'b'
        return prefix+suffix
    
def compute_intensities(matrix, freq_overlaps):

    n = len(matrix)
    intens = np.zeros(n)

    for i in range(n):
        for j in range(n):
            intens[i] += np.abs(matrix[i,j])**2 * np.real(freq_overlaps[j])
        ind=0
        for j in range(n):
            for k in np.arange(j+1, n):
                intens[i] += 2*np.real(matrix[i,k]*np.conj(matrix[i,j])) * np.real(freq_overlaps[n+ind])
                intens[i] -= 2*np.imag(matrix[i,k]*np.conj(matrix[i,j])) * np.imag(freq_overlaps[n+ind])
                ind += 1
    return intens



def compute_mutual_intensities(matrix, full_freq_overlaps):

    n = len(matrix)
    mutual_intens = np.zeros((n,n), dtype=np.complex_)

    for i1 in range(n):
        for i2 in range(n):
            for j in range(n):
                for k in range(n):
                    mutual_intens[i1,i2] += matrix[i1,j] * np.conj(matrix[i2,k]) * full_freq_overlaps[j,k]
    return mutual_intens
    # for i1 in range(n):
    #     for i2 in range(n):
    #         for j in range(n):
    #             mutual_intens[i1,i2] += (matrix[i1,j]) * np.conj(matrix[i2,j]) * (freq_overlaps[j])
    #         ind=0
    #         for j in range(n):
    #             for k in np.arange(j+1, n):
    #                 mutual_intens[i1, i2] += 2*np.real(matrix[i1,k]*np.conj(matrix[i2,j])) * np.real(freq_overlaps[n+ind])
    #                 mutual_intens[i1, i2] -= 2*np.imag(matrix[i1,k]*np.conj(matrix[i2,j])) * np.imag(freq_overlaps[n+ind])
    #                 ind += 1
    # return mutual_intens

class Scenes:

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
        ax, ay: angular separation in radians
        '''
        return np.exp(-2*np.pi*1j*(self.egrid_uv.x * ax + self.egrid_uv.y * ay))
        

        