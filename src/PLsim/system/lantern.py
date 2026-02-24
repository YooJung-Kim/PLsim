from scipy.stats import unitary_group
from scipy.linalg import expm
import numpy as np

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
    U = expm(A)
    
    return U

def make_3x3_unitary(theta_a, theta_b, theta_c = 0):
    '''
    Make 3x3 unitary matrix, mixing between LP01 and LP11a modes.

    Parameters
    theta_a : float
        Angle for the first mixing term (cos(theta_a) is the amplitude of LP01 mode in port 1)
    theta_b : float
        Angle for the second mixing term (phase offset between LP01 and LP11a modes)
    theta_c : float
        Angle for the third mixing term (phase offset between two ports. should not do anything)
    '''

    U = np.identity(3, dtype=complex)

    U[0,0] = np.cos(theta_a)
    U[0,1] = np.sin(theta_a) * np.exp(1j * theta_b)
    U[1,0] = np.sin(theta_a) * np.exp(1j * theta_c)
    U[1,1] = -np.cos(theta_a) * np.exp(1j * (theta_b + theta_c))

    return U