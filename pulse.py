import numpy as np
import pylab as plt
from scipy.signal.windows import hann, exponential, tukey, gaussian
from help_tools import plotting_interface

def assymmetric_sigmoid(lower_bound, upper_bound, N, factor=1):
    x = np.linspace(lower_bound, upper_bound, N)
    return 1./(1+np.exp(-factor*x))  


def switch_on(data):
    N = len(data)
    win = tukey(2*N, alpha=.95)[:N]
    plt.show()
    return win*data

def switch_off(data):
    N = len(data)
    win = tukey(2*N, alpha=0.95)[N:]
    return win*data

def dirac(N_window):
    rect=np.zeros(N_window)
    rect[N_window//2] = 1
    return rect

def rect(N_window, N_pulse_width, center_offset=0):
    rect = np.zeros(N_window)
    rect[N_window//2+center_offset-N_pulse_width//2+1:N_window//2+center_offset+N_pulse_width//2+1] = 1
    rect /= np.sum(rect)
    return rect


def hereon_old(N_window, N_pulse_width=15):
    win1 = exponential(N_window, tau=1)
    win1 /= np.sum(win1)

    win2 = exponential(N_window, tau=4)
    win2 /= np.sum(win2)

    rect_win = rect(N_window, N_pulse_width)
    c1 = np.convolve(win1, rect_win, mode='same')
    c2 = np.convolve(win2, rect_win, mode='same')

    N_overlap = 4
    N1 = N_window//2-N_overlap
    N2 = N_window//2+N_overlap
    combined = (np.block([c1[:N1], (switch_off(c1[N1:N2])+switch_on(c2[N1:N2])), c2[N2:]]))
    combined /=np.sum(combined)
    return combined


def hereon(N_window, N_pulse_width):
    N1 = N_pulse_width//2
    N2 = int(1.5*N_pulse_width)
    c1 = assymmetric_sigmoid(-6,3,N1)
    c2 = 1-assymmetric_sigmoid(-3,6,N2)
    combined = (np.block([np.zeros(N_window//2-N1), c1, c2, np.zeros(N_window//2-N2)]))
    N_rect_win = N_window//20
    rect_win = rect(N_window//2, N_rect_win)
    conv_combined = np.convolve(combined, rect_win, mode='same')
    combined[N_window//2+N_rect_win:] = conv_combined[N_window//2+N_rect_win:]
    combined = np.where(combined>0.9, 0.9, combined)
    combined /=np.sum(combined)
    return combined   

def riverRad(N_window, N_pulse_width):
    pulse =  gaussian(N_window, std=N_pulse_width/4)**1
    pulse /= np.sum(pulse)
    return pulse

if __name__=='__main__':
    import pylab as plt
    pulse = hereon(1000, 150)
    pulse2 = riverRad(1000, 150)
    plt.plot(pulse, 'o')
    plt.plot(pulse2, )
    plt.show()
