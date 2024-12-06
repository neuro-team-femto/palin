#!/usr/bin/env python
'''
PALIN toolbox v0.1
Decemberr 2022, Aynaz Adl Zarrabi, JJ Aucouturier (CNRS/UBFC)

Functions for kernel calculating method in Classification images
'''

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class KernelExtractor(ABC):

    @classmethod
    @abstractmethod
    def extract_single_kernel(cls,data_df, trial_id='trial',stim_id='stim', feature_id='feature', value_id='value', response_id='response', **kwargs): 
        raise NotImplementedError()

    @classmethod
    def extract_kernels(cls,data_df, group_ids, trial_id='trial',stim_id='stim', feature_id='feature', value_id='value', response_id='response', normalize = True, **kwargs):
        
        # for each level in group, compute kernels

        def extract_normalize(group): 
            kernel = cls.extract_single_kernel(group, trial_id,stim_id, feature_id, value_id, response_id, **kwargs)
            if normalize: 
                kernel = cls.normalize_kernel(kernel)
            return kernel

        return data_df.groupby(group_ids).apply(lambda group: extract_normalize(group)).reset_index()

    @classmethod    
    def normalize_kernel(cls,kernel):
        if isinstance(kernel,pd.DataFrame):
            rms = np.sqrt((kernel.kernel_value**2).mean())
            kernel.kernel_value /= rms
        elif isinstance(kernel,(np.ndarray, list)):
            rms = np.sqrt(np.mean(np.power(kernel,2)))
            kernel = kernel/rms
        else: 
            raise TypeError('argument kernel is neither a pd.DataFrame or a np.ndarray')
        return kernel