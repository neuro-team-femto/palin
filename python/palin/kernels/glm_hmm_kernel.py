#!/usr/bin/env python
'''
PALIN toolbox v0.1
December 2022, Aynaz Adl Zarrabi, JJ Aucouturier (CNRS/UBFC)

'''

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial
from .kernel_extractor import KernelExtractor

class GLMHMMKernel(KernelExtractor):
    """
    This class extracts kernel weights using a Generalized Linear Model (GLM).
    """

    @classmethod
    def extract_single_kernel(cls, data_df, trial_id='trial',stim_id='stim', feature_id='feature', value_id='value', response_id='response', **kwargs):
        """
        Extracts a single kernel by fitting a GLM to the given data.
        Returns a DataFrame with feature IDs and kernel values.
        """

        # train GLM HMM on data (possibly give priors as parameters)
        model = cls.train_HMM_from_data(data_df, trial_id, stim_id, feature_id, value_id, response_id, **kwargs)

        # take kernel from engaged state       
        kernel = cls.convert_model_to_kernel(model, feature_id)
        
        return kernel
    
    @classmethod
    def convert_model_to_kernel(csl,model, feature_id='feature'):
        '''
        Take kernel from engaged state of a trained HMM model'''
        
        raise NotImplementedError()


    @classmethod
    def train_HMM_from_data(cls, data_df, trial_id='trial',stim_id='stim', feature_id='feature', value_id='value', response_id='response', **kwargs):

        # Warning: this won't work for 1-int data (and no real way to know whether it's the case)

        # use ssm library
        # possibly pass priors as argument (or by default use best priors)        

        raise NotImplementedError()
        
    def __str__(self):
        return 'GLM-HMM Kernel'


     