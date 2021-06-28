import numpy as np


from scipy.interpolate import RectBivariateSpline

def cart2pol(x, y, cart, Nr=200, Ntheta=360):
    '''
    Conversion from cartesian to polar coordinates
    Parameter:
    ----------
    input:
            x       array
                    1d array of x-axis
            y       array
                    1d array of y-axis
            cart    2d array (dimensions matching x,y, 'ij'-indexing)
                    matrix to be converted
    output:
            r       array
                    1d array of r-axis
            theta   array
                    1d array of theta-axis
            pol     2d array
                    data in polar coordinats
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

def pol2cart(r, theta, pol, Nx=128, Ny=128):
    '''
    Conversion from polar to cartesian coordinates
    Parameter:
    ----------
    input:
            r       array
                    1d array of r-axis
            theta   array
                    1d array of theta-axis
            pol     2d array
                    data in polar coordinats
    output:
            x       array
                    1d array of x-axis
            y       array
                    1d array of y-axis
            cart    2d array (dimensions matching x,y, 'ij'-indexing)
                    matrix to be converted
    '''
    rr, th = np.meshgrid(r, theta, indexing='ij')
    x_pol = rr*np.cos(th)
    y_pol = rr*np.sin(th)
    x_min = np.min(x_pol)
    x_max = np.max(x_pol)
    y_min = np.min(y_pol)
    y_max = np.max(y_pol)
    x = np.linspace(x_min, x_max, Nx, endpoint=True)
    y = np.linspace(y_min, y_max, Ny, endpoint=True)
    F = RectBivariateSpline(r, theta, pol)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    r_cart = np.sqrt(xx**2+ yy**2)
    th_cart = np.arctan2(xx,yy)
    cart = F(r_cart.ravel(), th_cart.ravel(), grid=False).reshape((Nx, Ny))
    return x, y, cart



def cart2cylindrical(t, x, y, cart, Nr=200, Ntheta=360):
    '''
    blabla
    polar coordinates in (x,y)
    '''
    r = np.zeros(Nr)
    theta = np.zeros(Ntheta)
    Nt = len(t)
    cylindrical = np.zeros((Nt, Nr, Ntheta))
    for i in range(0, Nt):
        r, theta, cylindrical[i,:,:] = cart2pol(x, y, cart[i,:,:], Nr, Ntheta)
    return 