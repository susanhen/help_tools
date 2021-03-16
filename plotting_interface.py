import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
contour_colormap = cm.coolwarm

labelsize=14

def get_radial_filter(x, y):
    print(x,y)
    x_mesh, y_mesh = np.meshgrid(x, y, indexing='ij')
    r_max = np.min([np.max(np.abs(x)), np.max(np.abs(y))])
    return (np.sqrt(x_mesh**2+y_mesh**2)<r_max).astype('int')
    

def plot_3d_surface(x, y, z, radial_filter=False):
    if radial_filter:
        filt = radial_filter(x, y)
    else:
        filt=1
    fig = plt.figure()
    axes = fig.gca(projection='3d')
    x_mesh, y_mesh = np.meshgrid(x, y, indexing='ij')
    axes.plot_surface(x_mesh, y_mesh, (filt*z), cmap=cm.coolwarm)
    
def plot_kx_ky_coeffs(kx, ky, coeffs, radial_filter=False): 
    plot_3d_as_2d(kx, ky, coeffs, radial_filter)    
     
def plot_kx_ky_spec(kx, ky, spec, radial_filter=False):
    
    plot_3d_as_2d(kx, ky, spec, radial_filter)
    
def plot_3d_as_2d(x, y, z, radial_filter=False):
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    if radial_filter:
        filt = get_radial_filter(x, y)
    else:
        filt = 1
    plt.figure()
    plt.imshow((filt*z).T, origin='lower', extent=(x[0]-dx//2, x[-1]+dx//2, y[0]-dy//2, y[-1]+dy//2, ))
    
        
def plot_contours(x, y, z, radial_filter=False, levels=None, z_label=None):
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    if radial_filter:
        filt = get_radial_filter(x, y)
    else:
        filt = 1
    fig = plt.figure()
    if levels==None:
        CS = plt.contour((filt*z).T, origin='lower', extent=(x[0]-dx//2, x[-1]+dx//2, y[0]-dy//2, y[-1]+dy//2, ))
    else:
        CS = plt.contour((filt*z).T, levels, origin='lower', extent=(x[0]-dx//2, x[-1]+dx//2, y[0]-dy//2, y[-1]+dy//2, ))    
    cbar = fig.colorbar(CS, shrink=0.8)
    if z_label!=None:
        cbar.ax.set_ylabel(z_label, size=labelsize, labelpad=-40, y=1.15, rotation=0)
    
def plot_contourf(x, y, z, radial_filter=False, levels=None, z_label=None):
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    if radial_filter:
        filt = get_radial_filter(x, y)
    else:
        filt = 1
    fig = plt.figure()
    if levels==None:
        CF = plt.contourf((filt*z).T, origin='lower', extent=(x[0]-dx//2, x[-1]+dx//2, y[0]-dy//2, y[-1]+dy//2, ), cmap=contour_colormap)
    else:
        CF = plt.contourf((filt*z).T, levels, origin='lower', extent=(x[0]-dx//2, x[-1]+dx//2, y[0]-dy//2, y[-1]+dy//2, ), cmap=contour_colormap)
    cbar = fig.colorbar(CF, shrink=0.8, panchor=(1., 0.4))
    if z_label!=None:
        cbar.ax.set_ylabel(z_label, size=labelsize, labelpad=-20, y=1.15, rotation=0)


def plot_surfaces_along_y_at_random(surface_list, y_label=r'$y~[\mathrm{m}]$', z_label=r'$\eta~[\mathrm{m}]$'):
    plt.figure()
    Nx, Ny = (surface_list[0]).eta.shape
    where = int(np.random.rand()*Ny)
    for surf in surface_list:
        plt.plot(surf.y, surf.eta[where,:], label=surf.name)
    plt.xlabel(y_label)
    plt.ylabel(z_label)
    plt.legend()

def plot_surfaces_along_y_at_pos(surface_list, x_pos, y_label=r'$y~[\mathrm{m}]$', z_label=r'$\eta~[\mathrm{m}]$'):
    plt.figure()
    x = (surface_list[0]).x
    where = np.argmin(np.abs(x-x_pos))
    for surf in surface_list:
        plt.plot(surf.y, surf.eta[where,:], label=surf.name)
    plt.xlabel(y_label)
    plt.ylabel(z_label)
    plt.legend()    
            