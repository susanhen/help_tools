import numpy as np


from scipy.interpolate import RectBivariateSpline

def cart2finePol(x, y, cart, Nr=200, Ntheta=400):
    '''
    Currently implemented with origin at center
    '''
    xx, yy = np.meshgrid(x, y, indexing='ij')
    theta_cart = np.arctan2(yy, xx)
    r_cart = np.sqrt(xx**2 + yy**2)
    rmin = np.min(r_cart)
    rmax = np.max(r_cart)
    tmin = np.min(theta_cart)
    tmax = np.max(theta_cart)
    theta = np.linspace(tmin, tmax, Ntheta, endpoint=True)
    r = np.linspace(rmin, rmax, Nr, endpoint=True)
    F = RectBivariateSpline(x, y, cart)
    rr, th = np.meshgrid(r, theta, indexing='ij')
    x_pol = rr*np.cos(th)
    y_pol = rr*np.sin(th)
    pol = F(x_pol.ravel(), y_pol.ravel(), grid=False).reshape((Nr, Ntheta))


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
    return cart/rr


def cart2cylindrical(t, x, y, cart, Nr=200, Ntheta=400):
    '''
    blabla
    polar coordinates in (x,y)
    '''
    r = np.zeros(Nr)
    theta = np.zeros(Ntheta)
    Nt = len(t)
    cylindrical = np.zeros((Nt, Nr, Ntheta))
    for i in range(0, Nt):
        r, theta, cylindrical[i,:,:] = cart2finePol(x, y, cart[i,:,:], Nr, Ntheta)
    return 