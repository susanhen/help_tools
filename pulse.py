import numpy as np
import pylab as plt
from scipy.signal.windows import hann, exponential, tukey
from help_tools import plotting_interface

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

def rect(N_window, N_pulse_width):
    rect = np.zeros(N_window)
    rect[N_window//2-N_pulse_width//2+1:N_window//2+N_pulse_width//2+1] = 1
    rect /= np.sum(rect)
    return rect


def hereon(N_window, N_pulse_width=15):
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
