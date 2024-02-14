import pandas as pd
import numpy as np

def kernel_distance(kernel_1, kernel_2, type='CORR'): 
    if type == 'RMS': 
        return kernel_rms(kernel_1, kernel_2)
    elif type == 'CORR': 
        return kernel_correlation(kernel_1, kernel_2)
    else: 
        raise AttributeError('metric type %s unknown'%s)


def kernel_rms(kernel_1, kernel_2):
    if isinstance(kernel_1,pd.DataFrame) & isinstance(kernel_2,pd.DataFrame):
        rms = np.sqrt(np.mean((kernel_1.kernel_value - kernel_2.kernel_value_2)**2))

    elif isinstance(kernel_1,(np.ndarray, list)) & isinstance(kernel_2,(np.ndarray, list)):
        rms = np.sqrt(np.mean(np.power(kernel_1-kernel_2,2)))
    
    else: 
        raise TypeError('argument kernels are neither both pd.DataFrames or np.ndarrays')
    
    return rms    

def kernel_correlation(kernel_1, kernel_2): 

    if isinstance(kernel_1,pd.DataFrame) & isinstance(kernel_2,pd.DataFrame):

        correlation = np.corrcoef(kernel_1.kernel_value, kernel_2.kernel_value)[0, 1]

    elif isinstance(kernel_1,(np.ndarray, list)) & isinstance(kernel_2,(np.ndarray, list)):

        correlation = np.corrcoef(kernel_1, kernel_2)[0, 1]
    
    else: 
        raise TypeError('argument kernels are neither both pd.DataFrames or np.ndarrays')
    
    return correlation    
