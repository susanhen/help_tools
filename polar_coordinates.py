import numpy as np

def cart2finePol(x, y, cart, Nr=200, Ntheta=400):
    '''
    Currently implemented with origin at center
    '''
    xx, yy = np.meshgrid(x, y, indexing='ij')
    r_square = xx**2 + yy**2
    angles = np.arctan2(yy,xx)
    angles = np.where(angles<0, angles+(2*np.pi), angles)
    r_max = np.sqrt(np.max(r_square))
    r_min = np.sqrt(np.min(r_square))
    r = np.linspace(r_min, r_max, Nr)
    theta = np.linspace(np.min(angles), np.max(angles), Ntheta, endpoint=True)
    rr, th = np.meshgrid(r, theta, indexing='ij')
    x_indices = np.argmin(np.abs(np.outer(rr*np.cos(th), np.ones(len(x)))-np.outer(np.ones(rr.shape), x)), axis=1).reshape(rr.shape)
    y_indices = np.argmin(np.abs(np.outer(rr*np.sin(th), np.ones(len(y)))-np.outer(np.ones(rr.shape) , y)), axis=1).reshape(rr.shape)
    # values outside the cartesian domain will get boundary values in correct backmapping the values should be ignored
    pol = cart[x_indices, y_indices]
    return r, theta, pol

def averagePol2cart(r, theta, pol, x, y):
    '''
    return to cartesian coordinates when polar coordinates are fine
    polar values are associated with a cartesian patch
    it is averaged over all values associated with each patch
    '''
    rr, th = np.meshgrid(r, theta, indexing='ij')
    x_pol = rr*np.cos(th)
    y_pol = rr*np.sin(th)
    Nx = len(x)
    Ny = len(y)
    cart = np.zeros((Nx, Ny))
    counter = np.zeros((Nx, Ny))
    x_indices = np.argmin(np.abs(np.outer(x_pol, np.ones(Nx)) - np.outer(np.ones(x_pol.shape), x)), axis=1)
    y_indices = np.argmin(np.abs(np.outer(y_pol, np.ones(Ny)) - np.outer(np.ones(y_pol.shape), y)), axis=1)
    for i in range(0,len(x_indices)):
            cart[x_indices[i], y_indices[i]] += pol.flatten()[i]
            counter[x_indices[i], y_indices[i]] += 1

    cart = np.where(counter>0, cart/counter, 0)
    return cart