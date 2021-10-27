import unittest
import numpy as np
import matplotlib.pyplot as plt
import polar_coordinates
import plotting_interface

def fspecial_gauss(size, sigma):
    
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    xx, yy = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((xx**2 + yy**2)/(2.0*sigma**2)))
    return xx, yy, g

class Helper(unittest.TestCase):
    def setUp(self):
        self.nothing = 0

    def test_cart2pol(self, plot_it=False):
        size = 128
        sigma = 16
        xx, yy, z = fspecial_gauss(size, sigma)
        x = xx[:,0]
        y = yy[0,:]
        Nx, Ny = len(x), len(y)
        Nr = 200
        Ntheta = 360
        r, theta, z_pol = polar_coordinates.cart2pol(x, y, z, Nr=Nr, Ntheta=Ntheta)

        ind1a, ind1b, ind2, ind3 = Nr//2, Ntheta//2, 10, 20
        r_test = [r[ind1a], r[ind2], r[ind3]]
        theta_test = [theta[ind1b], theta[ind2], theta[ind3]]
        rr_test_mesh, th_test_mesh = np.meshgrid(r_test, theta_test, indexing='ij')
        x_test_input = rr_test_mesh*np.cos(th_test_mesh)
        y_test_input = rr_test_mesh*np.sin(th_test_mesh)
        test_output = np.exp(-((x_test_input**2 + y_test_input**2)/(2.0*sigma**2))).ravel()
        polar_output = np.array(z_pol[np.ix_([ind1a, ind2, ind3],[ind1b, ind2, ind3])]).ravel()
        for i in range(0,6):
            self.assertAlmostEqual(test_output[i], polar_output[i], delta=0.0005)

        if plot_it:
            plt.figure()
            plt.imshow(z.T, origin='lower')
            plt.figure()
            plt.imshow(z_pol.T, origin='lower')
            plt.show()
        # veryfy backtransform 
        x_new, y_new, z_new = polar_coordinates.pol2cart(r, theta, z_pol, x_out=x, y_out=y)
        for i in range(1,Nx):
            self.assertAlmostEqual(x_new[i], x[i])
            for j in range(0,Ny):
                self.assertAlmostEqual(y_new[j], y[j])
                self.assertAlmostEqual(z_new[i,j], z[i,j], delta=0.1)
            
    def test_pol2cart(self, plot_it=False):
        sigma = 16
        size = 128
        Nr = 64
        Ntheta = 360
        Nx = 128
        Ny = 128
        r = np.arange(0, Nr)
        g = np.exp(-r**2/(2*sigma**2))
        g_pol = np.outer(g, np.ones(360))
        theta = np.linspace(0, 2*np.pi, Ntheta)
        x, y, g_cart = polar_coordinates.pol2cart(r, theta, g_pol, Nx=Nx, Ny=Ny)

        ind1, ind2, ind3 = Nx//2, 10, 20
        x_test = [x[ind1], x[ind2], x[ind3]] 
        y_test = [y[ind1], y[ind2], y[ind3]]
        x_test_mesh, y_test_mesh = np.meshgrid(x_test, y_test, indexing='ij')
        r_test_input = np.sqrt(x_test_mesh**2 + y_test_mesh**2)
        test_output = np.exp(-r_test_input**2/(2*sigma**2)).ravel()
        cart_output = np.array(g_cart[np.ix_([ind1, ind2, ind3],[ind1, ind2, ind3])]).ravel()
        for i in range(0, 6):
            self.assertAlmostEqual(test_output[i], cart_output[i], delta=0.0005)
        if plot_it:
            plt.figure()
            plt.imshow(g_pol.T, origin='lower')
            plt.figure()
            plt.imshow(g_cart.T, origin='lower')
            plt.show()

        #verify backtransform
        r_new, theta_new, z_new = polar_coordinates.cart2pol(x, y, g_cart, r_out=r, theta_out=theta)
        for i in range(1,Nr):
            self.assertAlmostEqual(r_new[i], r[i])
            for j in range(0,Ntheta):
                self.assertAlmostEqual(theta_new[j], theta[j])
                self.assertAlmostEqual(z_new[i,j], g_pol[i,j], delta=0.1)

    def test_pol_better(self, plot_it=False):
        y = np.linspace(200, 700, 200)
        x = np.linspace(-250, 250, 200)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        data = np.sin(0.066*yy)
        if plot_it:
            plotting_interface.plot_3d_as_2d(x, y, data)

        r, th, pol = polar_coordinates.cart2pol(x, y, data)
        if plot_it:
            plotting_interface.plot_3d_as_2d(r, th, pol)

        x, y, cart = polar_coordinates.pol2cart(r, th, pol, x_out=x, y_out=y)
        if plot_it:
            plotting_interface.plot_3d_as_2d(x, y, cart)

        for i in range(0, len(x)):
            for j in range(0, len(y)):
                self.assertAlmostEqual(data[i,j], cart[i,j], delta=0.03)
        

if __name__=='__main__':
    unittest.main()