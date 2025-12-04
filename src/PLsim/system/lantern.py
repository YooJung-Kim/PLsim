from scipy.stats import unitary_group
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
    U = scipy.linalg.expm(A)
    
    return U