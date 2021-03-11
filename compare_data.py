import numpy as np

def get_rms(data1, data2, data_filter=None):    
    '''
    calculates the rms between the given data
    If a filter is provided it can help to see the value when certain regions are excluded
    '''
    if data_filter is None:
        data_filter = np.ones(data1.shape)
    return np.sqrt(np.sum((data_filter*np.abs(data1 - data2))**2)/np.sum(data_filter))

def get_abs_error(data1, data2, data_filter=None):
    if data_filter is None:
        data_filter = np.ones(data1.shape)        
    return data_filter*np.abs(data1 - data2)

def get_max_error(data1, data2, data_filter=None):
    if data_filter is None:
        data_filter = np.ones(data1.shape)        
    return np.max(get_abs_error(data1, data2, data_filter))

def get_filter_for_abs_error(data1, data2, tolerance):
    abs_error = np.abs(data1-data2)
    return (abs_error<tolerance).astype('int')


if __name__=='__main__':
    x = np.linspace(0,10*np.pi, 100)
    y = np.linspace(0,10*np.pi, 100)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    data1 = np.abs(np.sin(xx+2*yy))
    data2 = np.abs((np.sin(xx+2*yy))**2)
    filter_02 = get_filter_for_abs_error(data1, data2, 0.2)
    
    print('RMS = ', get_rms(data1, data2))
    print('Max error = ', get_max_error(data1, data2))
    print('Max error when excluding errors>=0.2 = ', get_max_error(data1, data2, data_filter=filter_02))
    import pylab as plt
    plt.figure()
    plt.imshow(filter_02)
    plt.show()