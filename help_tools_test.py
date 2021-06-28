import unittest
import numpy as np
import matplotlib.pyplot as plt
import polar_coordinates

def fspecial_gauss(size, sigma):
    
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    xx, yy = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((xx**2 + yy**2)/(2.0*sigma**2)))
    return xx, yy, g

class Helper(unittest.TestCase):
    def setUp(self):
        self.nothing = 0

    def test_polar_coordinats(self, plot_it=False):
        size = 128
        sigma = 16
        xx, yy, z = fspecial_gauss(size, sigma)
        x = xx[:,0]
        y = yy[0,:]
        '''
        Nr = 32
        Ntheta = 256
        theta_cart = np.arctan2(yy, xx)
        r_cart = np.sqrt(xx**2 + yy**2)
        rmin = np.min(r_cart)
        rmax = np.max(r_cart)
        tmin = np.min(theta_cart)
        tmax = np.max(theta_cart)
        theta = np.linspace(tmin, tmax, Ntheta, endpoint=True)
        r = np.linspace(rmin, rmax, Nr, endpoint=True)
        F = RectBivariateSpline(x, y, z)
        rr, th = np.meshgrid(r, theta, indexing='ij')
        x_pol = rr*np.cos(th)
        y_pol = rr*np.sin(th)
        z_pol = F(x_pol.ravel(), y_pol.ravel(), grid=False).reshape((Nr, Ntheta))
        '''
        Nr = 200
        Ntheta = 360
        r, theta, z_pol = polar_coordinates.cart2finePol(x, y, z, Nr=Nr, Ntheta=Ntheta)

        ind1a, ind1b, ind2, ind3 = Nr//2, Ntheta//2, 10, 20
        r_test = [r[ind1a], r[ind2], r[ind3]]
        theta_test = [theta[ind1b], theta[ind2], theta[ind3]]
        rr_test_mesh, th_test_mesh = np.meshgrid(r_test, theta_test, indexing='ij')
        x_test_input = rr_test_mesh*np.cos(th_test_mesh)
        y_test_input = rr_test_mesh*np.sin(th_test_mesh)
        test_output = np.exp(-((x_test_input**2 + y_test_input**2)/(2.0*sigma**2))).ravel()
        polar_output = np.array(z_pol[np.ix_([ind1a, ind2, ind3],[ind1b, ind2, ind3])]).ravel()
        for i in range(0,6):
            self.assertAlmostEqual(test_output[i], polar_output[i], places=3)

        if plot_it:
            plt.figure()
            plt.imshow(z.T, origin='lower')
            plt.figure()
            plt.imshow(z_pol, origin='lower')
            plt.show()

if __name__=='__main__':
    unittest.main()