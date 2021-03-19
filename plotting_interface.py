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

def plot_3d_as_2d(x, y, z, radial_filter=False):
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    if radial_filter:
        filt = get_radial_filter(x, y)
    else:
        filt = 1
    plt.figure()
    plt.imshow((filt*z).T, origin='lower', extent=(x[0]-dx//2, x[-1]+dx//2, y[0]-dy//2, y[-1]+dy//2, ))
    
def plot_kx_ky_coeffs(kx, ky, coeffs, radial_filter=False):     
    plot_3d_as_2d(kx, ky, coeffs, radial_filter)    
    plt.xlabel(r'$k_x~[\mathrm{rad~m}^{-1}]$') 
    plt.xlabel(r'$k_y~[\mathrm{rad~m}^{-1}]$')
     
def plot_kx_ky_spec(kx, ky, spec, radial_filter=False):    
    plot_3d_as_2d(kx, ky, spec, radial_filter)
    plt.xlabel(r'$k_x~[\mathrm{rad~m}^{-1}]$') 
    plt.xlabel(r'$k_y~[\mathrm{rad~m}^{-1}]$')

def plot_k_w_spec(k, w, spec, disp_filter=False):  
    plot_3d_as_2d(k, w, spec, disp_filter)    
    plt.xlabel(r'$k~[\mathrm{rad~m}^{-1}]$') 
    plt.xlabel(r'$\omega~[\mathrm{rad~Hz}]$')

def plot_wavenumber_spec(k, spec, scaled=True, k_cut_off=None):
    plot_wavenumber_specs([k], [spec], scaled, None, k_cut_off)

def plot_wavenumber_specs(k_list, spec_list, scaled=True, labels=None, k_cut_off=None):
    plt.figure()
    for i in range(0, len(k_list)):
        k = k_list[i]
        spec = spec_list[i]
        if k_cut_off is None:
            last_ind = -1
        else:
            last_ind = np.argmin(np.abs(k-k_cut_off))
        if scaled:
            scaling = np.max(spec[:last_ind])
        else:
            scaling = 1
        if labels is None:
            plt.plot(k[:last_ind], spec[:last_ind]/scaling)
        else:
            plt.plot(k[:last_ind], spec[:last_ind]/scaling, label=labels[i])
    plt.xlabel(r'$k~[\mathrm{rad~m}^{-1}]$')
    if scaled:
        plt.ylabel(r'$F(k)/\max(F(k))$')
    else:
        plt.ylabel(r'$F(k)~[\mathrm{m}^3]$')
    if not labels is None:
        plt.legend()


def plot_ang_frequency_specs(w_list, spec_list, scaled=True, labels=None, w_cut_off=None):
    plt.figure()
    for i in range(0, len(w_list)):
        w = w_list[i]
        spec = spec_list[i]
        if w_cut_off is None:
            last_ind = -1
        else:
            last_ind = np.argmin(np.abs(w-w_cut_off))
        if scaled:
            scaling = np.max(spec[:last_ind])
        else:
            scaling = 1
        if labels is None:
            plt.plot(w[:last_ind], spec[:last_ind]/scaling)
        else:
            plt.plot(w[:last_ind], spec[:last_ind]/scaling, label=labels[i])
    plt.xlabel(r'$\omega~[\mathrm{rad~Hz}]$')
    if scaled:
        plt.ylabel(r'$F(\omega)/\max(F(\omega))$')
    else:
        plt.ylabel(r'$F(\omega)~[\mathrm{m}^2/\mathrm{Hz}]$')   

def plot_ang_frequency_spec(w, spec, scaled=True, w_cut_off=None):
    plot_ang_frequency_specs([w], [spec], scaled, None, w_cut_off)


def plot_frequency_specs(f_list, spec_list, scaled=True, labels=None, f_cut_off=None):
    plt.figure()
    for i in range(0, len(f_list)):
        f = f_list[i]
        spec = spec_list[i]
        if f_cut_off is None:
            last_ind = -1
        else:
            last_ind = np.argmin(np.abs(f-f_cut_off))
        if scaled:
            scaling = np.max(spec[:last_ind])
        else:
            scaling = 1
        if labels is None:
            plt.plot(f[:last_ind], spec[:last_ind]/scaling)
        else:
            plt.plot(f[:last_ind], spec[:last_ind]/scaling, label=labels[i])
    plt.xlabel(r'$f~[\mathrm{rad~Hz}]$')
    if scaled:
        plt.ylabel(r'$F(f)/\max(F(f))$')
    else:
        plt.ylabel(r'$F(f)~[\mathrm{m}^2/\mathrm{Hz}]$')   

def plot_frequency_spec(f, spec, scaled=True, f_cut_off=None):        
    plot_frequency_specs([f], [spec], scaled, None, f_cut_off)    
    
        
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
        CF = plt.contourf((filt*z).T, origin='lower', extent=(x[0]-dx//2, x[-1]-dx//2, y[0]-dy//2, y[-1]-dy//2, ), cmap=contour_colormap)
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

def savefig(fn, tight=True):
    if tight:
        plt.savefig(fn, bbox_inches='tight')
    else:
        plt.savefig(fn)

def show():
    plt.show()