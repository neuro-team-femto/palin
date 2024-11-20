#!/usr/bin/env python
'''
PALIN toolbox v0.1
December 2022, Aynaz Adl Zarrabi, JJ Aucouturier (CNRS/UBFC)

'''

import pandas as pd
import numpy as np
from .kernel_extractor import KernelExtractor

class GLMKernel(KernelExtractor):

    @classmethod
    def extract_single_kernel(cls, data_df, feature_id = 'feature', value_id = 'value', response_id = 'response'):
    '''
    Extracts kernel by fitting a GLM to data_df and returning the GLM weights. 
    Returns a single kernel as a dataframe with feature and kernel_value
    '''

        #kernels['kernel_value'] = kernels['%s_true'%value_id] - kernels['%s_false'%value_id]
        #kernels = kernels[[feature_id,'kernel_value']].set_index(feature_id)
        #kernels.index.names = ['feature']
        #return kernels

    def __str__(self): 
        return 'GLM Kernel'


     