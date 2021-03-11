import numpy as np
import scipy.signal as s

def convolve2d(mat1, mat2):
    #TODO shape of matrices must be the same
    Nx, Ny = mat1.shape
    return np.flipud(s.convolve2d(np.flipud(mat1), mat2)[Nx//2:Nx//2+Nx, Ny//2:Ny//2+Ny])
    