import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

contour_colormap = cm.coolwarm
color_list = ['r', 'b', 'k'] #TODO make a proper one!

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
    axes = fig.add_subplot(111, projection='3d')
    x_mesh, y_mesh = np.meshgrid(x, y, indexing='ij')
    axes.plot_surface(x_mesh, y_mesh, (filt*z), cmap=cm.coolwarm)
    return axes

def plot_3d_as_2d(x, y, z, radial_filter=False, extent=None, ax=None, aspect='auto'):
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    if radial_filter:
        filt = get_radial_filter(x, y)
    else:
        filt = 1
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow((filt*z).T, origin='lower', extent=(x[0]-dx//2, x[-1]+dx//2, y[0]-dy//2, y[-1]+dy//2 ), aspect=aspect)
    if not extent is None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    return ax
    
def plot_kx_ky_coeffs(kx, ky, coeffs, radial_filter=False, extent=None, ax=None):     
    ax = plot_3d_as_2d(kx, ky, coeffs, radial_filter, extent)    
    ax.set_xlabel(r'$k_x~[\mathrm{rad~m}^{-1}]$') 
    ax.set_ylabel(r'$k_y~[\mathrm{rad~m}^{-1}]$')
    return ax
     
def plot_kx_ky_spec(kx, ky, spec, radial_filter=False, extent=None, ax=None):    
    ax = plot_3d_as_2d(kx, ky, spec, radial_filter, extent, ax, aspect=1)
    ax.set_xlabel(r'$k_x~[\mathrm{rad~m}^{-1}]$') 
    ax.set_ylabel(r'$k_y~[\mathrm{rad~m}^{-1}]$')
    return ax

def plot_k_w_spec(k, w, spec, disp_filter=False, extent=None, ax=None):  
    ax = plot_3d_as_2d(k, w, spec, disp_filter, extent, ax)    
    ax.set_xlabel(r'$k~[\mathrm{rad~m}^{-1}]$') 
    ax.set_ylabel(r'$\omega~[\mathrm{rad~Hz}]$')
    return ax

def plot_wavenumber_spec(k, spec, scaled=True, k_cut_off=None, extent=None, ax=None):
    plot_wavenumber_specs([k], [spec], scaled, None, k_cut_off, extent, ax)

def plot_wavenumber_specs(k_list, spec_list, scaled=True, labels=None, k_cut_off=None, extent=None, ax=None):
    if ax is None:
        fig, ax = np.subplots()
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
    ax.set_xlabel(r'$k~[\mathrm{rad~m}^{-1}]$')
    if scaled:
        ax.set_ylabel(r'$F(k)/\max(F(k))$')
    else:
        ax.set_ylabel(r'$F(k)~[\mathrm{m}^3]$')
    if not labels is None:
        ax.set_legend()
    if not extent is None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    return ax


def plot_ang_frequency_specs(w_list, spec_list, scaled=True, labels=None, w_cut_off=None, extent=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
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
    ax.set_xlabel(r'$\omega~[\mathrm{rad~Hz}]$')
    if scaled:
        ax.set_ylabel(r'$F(\omega)/\max(F(\omega))$')
    else:
        ax.set_ylabel(r'$F(\omega)~[\mathrm{m}^2/\mathrm{Hz}]$')   
    if not extent is None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    return ax

def plot_ang_frequency_spec(w, spec, scaled=True, w_cut_off=None, extent=None, ax=None):
    plot_ang_frequency_specs([w], [spec], scaled, None, w_cut_off, extent, ax)


def plot_frequency_specs(f_list, spec_list, scaled=True, labels=None, f_cut_off=None, extent=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
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
    ax.set_xlabel(r'$f~[\mathrm{rad~Hz}]$')
    if scaled:
        ax.set_ylabel(r'$F(f)/\max(F(f))$')
    else:
        ax.set_ylabel(r'$F(f)~[\mathrm{m}^2/\mathrm{Hz}]$') 
    if not extent is None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])  

def plot_frequency_spec(f, spec, scaled=True, f_cut_off=None, extent=None, ax=None):        
    plot_frequency_specs([f], [spec], scaled, None, f_cut_off, extent, ax)   
    
        
def plot_contours(x, y, z, radial_filter=False, levels=None, z_label=None, extent=None, ax=None):
    if radial_filter:
        filt = get_radial_filter(x, y)
    else:
        filt = 1
    if ax is None:
        fig, ax = plt.subplots()
    if levels==None:
        CS = ax.contour(x, y, (filt*z).T, origin='lower')    
    else:
        CS = ax.contour(x, y, (filt*z).T, levels, origin='lower')    
    cbar = fig.colorbar(CS, shrink=0.8)
    if z_label!=None:
        cbar.ax.set_ylabel(z_label, size=labelsize, labelpad=-40, y=1.15, rotation=0)
    if not extent is None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    return ax
    
def plot_contourf(x, y, z, radial_filter=False, levels=None, z_label=None, extent=None, ax=None):
    
    if radial_filter:
        filt = get_radial_filter(x, y)
    else:
        filt = 1
    if ax is None:
        fig, ax = plt.subplots()
    if levels==None:
        CF = plt.contourf(x, y, (filt*z).T, origin='lower', cmap=contour_colormap)
    else:
        CF = plt.contourf(x, y, (filt*z).T, levels, origin='lower', cmap=contour_colormap)
    cbar = fig.colorbar(CF, shrink=0.8, panchor=(1., 0.4))
    if z_label!=None:
        cbar.ax.set_ylabel(z_label, size=labelsize, labelpad=-20, y=1.15, rotation=0)
    if not extent is None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

def plot_surf_time_space(t, x, surf, extent=None):
    ax = plot_3d_as_2d(t, x, surf, extent=extent) 
    ax.set_xlabel(r'$t~[\mathrm{s}]$')   
    ax.set_ylabel(r'$x~[\mathrm{m}]$') 
    return ax


def plot_surf_x_y(x, y, surf, extent=None):
    ax = plot_3d_as_2d(x, y, surf, extent=extent) 
    ax.set_xlabel(r'$x~[\mathrm{m}]$')   
    ax.set_ylabel(r'$y~[\mathrm{m}]$') 
    return ax

def plot_3d_surf_x_y(x, y, surf):
    ax = plot_3d_surface(x, y, surf)
    ax.set_xlabel(r'$x~[\mathrm{m}]$')   
    ax.set_ylabel(r'$y~[\mathrm{m}]$') 
    return ax

    

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

def plot_disp_rel_kx_ky(w, h):
    k_disp_rel = w**2/9.81
    kx_fine = np.linspace(-k_disp_rel, k_disp_rel, 300)
    # plot positive ky
    print('Warning: not fully implemented, current and shallow water not included')
    ky_disp_rel_pos =  np.sqrt(k_disp_rel**2 - kx_fine**2)
    plt.plot(kx_fine, ky_disp_rel_pos, 'w')
    # plot negative ky
    ky_disp_rel_neg =  -np.sqrt(k_disp_rel**2 - kx_fine**2)
    plt.plot(kx_fine, ky_disp_rel_neg, 'w')



def plot_disp_shell(axes, h, z, U, psi, label='', plot_type='surf', linestyles='line', put_clabel=True):
    g = 9.81
    alpha = 0.5 # value that defines opacity in plot
    dk = 0.005
    k = np.arange(0.01, 0.35, dk)
    dtheta=0.05
    theta=np.arange(0, 2*np.pi+dtheta, dtheta)
    kk, th = np.meshgrid(k, theta, indexing='ij')
    U_eff = 2*kk*np.sum(U*np.exp(np.outer(2*kk,z)), axis=1).reshape(kk.shape)*np.abs(z[1]-z[0])
    ww = kk*U_eff*np.cos(theta-psi) + np.sqrt(kk*g*np.tanh(kk*h))
    kx = kk*np.cos(th)
    ky = kk*np.sin(th)
    if plot_type=='surf':
        axes.plot_surface(kx, ky, ww, alpha=alpha, label=label)
        axes.set_xlabel(r'$k_x~[\mathrm{rad~m}^{-1}]$')
        axes.set_ylabel(r'$k_y~[\mathrm{rad~m}^{-1}]$')
        axes.set_zlabel(r'$\omega~[\mathrm{rad~s}^{-1}]$')
    elif plot_type=='contour':
        levels = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
        c = plt.contour(kx, ky, ww, levels=levels, linestyles=linestyles)#, label=label)
        if put_clabel:
            plt.clabel(c)
        plt.xlabel(r'$k_x~[\mathrm{rad~m}^{-1}]$') 
        plt.ylabel(r'$k_y~[\mathrm{rad~m}^{-1}]$')
        plt.axis('equal')

def plot_multiple_disp_rel(h, z_list, U_list, psi_list, label_list, plot_type='surf', linestyle_list=None):
    if linestyle_list is None:
        linestyle_List = len(z_list) * ['solid']
    fig = plt.figure()
    if plot_type=='surf':
        axes = fig.gca(projection='3d')
        for i in range(0, len(U_list)):
            plot_disp_shell(axes, h, z_list[i], U_list[i], psi_list[i], label_list[i], plot_type)
    else:
        axes = plt.subplot(1,1,1)
        for i in range(0, len(U_list)):
            if i==0:
                plot_disp_shell(axes, h, z_list[i], U_list[i], psi_list[i], label_list[i], plot_type, linestyles=linestyle_list[i], put_clabel=True)
            else:
                plot_disp_shell(axes, h, z_list[i], U_list[i], psi_list[i], label_list[i], plot_type, linestyles=linestyle_list[i], put_clabel=False)
    if plot_type!='surf':
        plt.legend()

def plot_disp_rel_at(at_w, h, z, U, psi, color, ax, extent=None):
    '''
    Plot the dispersion relation for provided frequencie(s) and the current profile U(z).
    The calculations are based on the assumption of Stewart and Joy for the effective current
    
    Parameters:
    -----------
        input
                at_w    array/float
                        frequencie(s) of interest
                z       array
                        grid for velocity profile
                U       array   
                        velocity profile over z
                psi     float
                        angle between waves and current
                color   string
                        color for plotting 
                extent  tupel/array optional
                        to limit the extent of the drawing
        output
                CS      array
                        representation of the contour(s)
    '''
    g = 9.81
    dk = 0.005
    k = np.arange(0.01, 0.7, dk)
    dtheta=0.05
    theta=np.arange(0, 2*np.pi+dtheta, dtheta)
    kk, th = np.meshgrid(k, theta, indexing='ij')
    U_eff = 2*kk*np.sum(U*np.exp(np.outer(2*kk,z)), axis=1).reshape(kk.shape)*np.abs(z[1]-z[0])
    ww = kk*U_eff*np.cos(th-psi) + np.sqrt(kk*g*np.tanh(kk*h))
    kx = kk*np.cos(th)
    ky = kk*np.sin(th)
    if type(at_w) is float or type(at_w) is np.float64:
        CS = ax.contour(kx, ky, ww, origin='lower', levels=[at_w], colors=color)
    else:
        CS = ax.contour(kx, ky, ww, origin='lower', levels=at_w, colors=color)
    ax.set_aspect('equal')
    if not extent is None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    return CS

def plot_multiple_disp_rel_at(at_w, h, z_list, U_list, psi_list, label_list, plot_type='surf', extent=None):
    fig, ax = plt.subplots(1,1)
    lines = []
    for i in range(0, len(U_list)):
        CS = plot_disp_rel_at(at_w, h, z_list[i], U_list[i], psi_list[i], color_list[i], ax, extent)
        lines.append(CS.collections[0])    
    ax[0].set_legend(lines, label_list)


def plot_disp_rel_for_Ueff_at(at_w, h, U_eff, psi, color, ax, extent=None):
    '''
    Plot the dispersion relation for provided frequencie(s) and the current profile U(z).
    The calculations are based on the assumption of Stewart and Joy for the effective current
    
    Parameters:
    -----------
        input
                at_w    array/float
                        frequencie(s) of interest
                U_eff   array   
                        velocity profile over z
                psi     float
                        angle between waves and current
                color   string
                        color for plotting 
                extent  tupel/array optional
                        to limit the extent of the drawing
        output
                CS      array
                        representation of the contour(s)
    '''
    g = 9.81
    dk = 0.005
    k = np.arange(0.01, 0.7, dk)
    dtheta=0.05
    theta=np.arange(0, 2*np.pi+dtheta, dtheta)
    kk, th = np.meshgrid(k, theta, indexing='ij')
    ww = kk*U_eff*np.cos(th-psi) + np.sqrt(kk*g*np.tanh(kk*h))
    kx = kk*np.cos(th)
    ky = kk*np.sin(th)
    if type(at_w) is float or type(at_w) is np.float64:
        CS = ax.contour(kx, ky, ww, origin='lower', levels=[at_w], colors=color)
    else:
        CS = ax.contour(kx, ky, ww, origin='lower', levels=at_w, colors=color)
    ax.set_aspect('equal')
    if not extent is None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    return CS    


def savefig(fn, tight=True):
    if tight:
        plt.savefig(fn, bbox_inches='tight')
    else:
        plt.savefig(fn)

def figure():
    plt.figure()

def plot(x, y, ax=None, label=None):
    if ax is None:
        plt.plot(x, y, label=label)
    else:
        ax.plot(x,y, label=label)

def legend():
    plt.legend()

def show():
    plt.show()

def colorbar():
    return plt.colorbar()

def subplots(figsize=None):
    if figsize is None:
        return plt.subplots()
    else:
        return plt.subplots(figsize=figsize)

def get_cmap(cm_name):
    return plt.get_cmap(cm_name)

def xlabel(label):
    plt.xlabel(label)

def xlabel(label):
    plt.ylabel(label)

def label_x_eta():
    plt.xlabel(r'$x~[\mathrm{m}]$')
    plt.ylabel(r'$\eta~[\mathrm{m}]$')

def label_t_eta():
    plt.xlabel(r'$t~[\mathrm{s}]$')
    plt.ylabel(r'$\eta~[\mathrm{m}]$')