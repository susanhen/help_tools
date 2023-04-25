import numpy as np


from scipy.interpolate import RectBivariateSpline

def cart2pol(x, y, cart, Nr=200, Ntheta=360, r_out=None, theta_out=None):
    '''
    Conversion from cartesian to polar coordinates
    Extrapolation possible (can be avoided by setting r_out, theta_out accordingly)
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
    if theta_out is None:
        theta = np.linspace(tmin, tmax, Ntheta, endpoint=True)
    else:
        theta = theta_out
        Ntheta = len(theta)
    if r_out is None:
        r = np.linspace(rmin, rmax, Nr, endpoint=True)
    else:
        r = r_out
        Nr = len(r)
    F = RectBivariateSpline(x, y, cart)
    rr, th = np.meshgrid(r, theta, indexing='ij')
    x_pol = rr*np.cos(th)
    y_pol = rr*np.sin(th)
    pol = F(x_pol.ravel(), y_pol.ravel(), grid=False).reshape((Nr, Ntheta))
    return r, theta, pol

def pol2cart(r, theta, pol, Nx=128, Ny=128, x_out=None, y_out=None):
    '''
    Conversion from polar to cartesian coordinates
    Extrapolation possible (can be avoided by setting x_out, y_out correctly)
    Parameter:
    ----------
    input:
            r       array
                    1d array of r-axis
            theta   array
                    1d array of theta-axis
            pol     2d array
                    data in polar coordinats (r,theta)
            Nx      int, optional
                    number of points in x_dir (only used if x_out now given)
            Ny      int, optional
                    number of points in x_dir (only used if y_out now given)
            x_out   array, optional
                    x-grid for the output (1d)
            y_out   array, optional
                    y-grid for the output (1d)
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
    x_min = np.min(r)#np.min(x_pol)
    x_max = np.max(r)#np.max(x_pol)
    y_min = np.min(y_pol)
    y_max = np.max(y_pol)
    if x_out is None:
        x = np.linspace(x_min, x_max, Nx, endpoint=True)
    else:
        x = x_out
        Nx = len(x)
    if y_out is None:
        y = np.linspace(y_min, y_max, Ny, endpoint=True)
    else:
        y = y_out
        Ny = len(y)
    F = RectBivariateSpline(r, theta, pol)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    mask = np.logical_and((xx**2 + yy**2)<=np.max(r)**2, np.abs(np.arctan2(yy,xx))<np.abs(np.max(th))).astype(int)
    from help_tools import plotting_interface
    #plotting_interface.plot_3d_as_2d(r, theta, np.log10(np.abs(F(r,theta))**2))
    ##plotting_interface.plot_surf_x_y(x, y, mask)
    #plotting_interface.show()

    r_cart = np.sqrt(xx**2+ yy**2)
    th_cart = np.arctan2(yy,xx)
    if np.max(theta)>np.pi:
        th_cart += np.pi
    cart = np.where(mask, F(r_cart.ravel(), th_cart.ravel(), grid=False).reshape((Nx, Ny)), np.nan)
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
    return r, theta, cylindrical