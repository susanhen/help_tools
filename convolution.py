import numpy as np
import scipy.signal as s

def convolve1d(filt, data):
    N = len(data)
    return np.convolve(filt, data, mode='full')[N//2:N//2+N]

def convolve2d(mat1, mat2):
    #TODO shape of matrices must be the same
    Nx, Ny = mat1.shape
    #return (np.flipud(s.convolve2d((np.flipud(mat1)), mat2, mode='full', boundary='wrap'))[Nx//2:Nx//2+Nx, Ny//2:Ny//2+Ny])
    return s.convolve2d(mat1, mat2, mode='full', boundary='wrap')[Nx//2:Nx//2+Nx, Ny//2:Ny//2+Ny]
    
def convolve2d_one_axis(filt, data, axis=0):
    return np.apply_along_axis(lambda m: convolve1d(m, filt), axis=axis, arr=data)
