#!/usr/bin/env python
'''
PALIN toolbox v0.1
December 2022, Aynaz Adl Zarrabi, JJ Aucouturier (CNRS/UBFC)

'''

import pandas as pd
import numpy as np
import os.path
import warnings
import ast
import statsmodels.formula.api as smf
import statsmodels.api as sm
import seaborn as sns
from ..kernels.glm_hmm_kernel import GLMHMMKernel
from .internal_noise_extractor import InternalNoiseExtractor

from abc import ABC,abstractmethod

class GLMHMMMethod(InternalNoiseExtractor):
    """
    Implements a method to estimate internal noise based on the confidence intervals from a GLM fit.
    """

    def __str__(self):
        return "GLM HMM Method"

    @classmethod
    def extract_single_internal_noise(cls, data_df, trial_id='trial', feature_id='feature', value_id='value', response_id='response', **kwargs):
        """
        Extracts internal noise for a single observer/session using the GLM fit.

        Parameters:
        - data_df (pd.DataFrame): DataFrame containing trial data.
        - trial_id (str): Column name for trial IDs.
        - feature_id (str): Column name for feature IDs.
        - value_id (str): Column name for feature values.
        - response_id (str): Column name for response values.

        Returns:
        - float: Estimated internal noise.
        """
        
        # train GLM HMM on data (possibly give priors as parameters)
        model = GLMHMMKernel.train_HMM_from_data(data_df, trial_id, stim_id, feature_id, value_id, response_id, **kwargs)

        # decode original data with model to keep only engaged trials
        engaged_data_df = csl.decode_engaged_data(data_df, model)

        # run GLMMethod on engaged data
        if 'internal_noise_extractor' not in kwargs:
            raise ValueError('no internal_noise_extractor provided for GLM-HMM Method.') 
        
        internal_noise = kwargs['internal_noise_extractor'].extract_single_internal_noise(engaged_data_df, trial_id, feature_id, value_id, response_id, **kwargs)
        
        return internal_noise
    
    @classmethod
    def decode_engaged_data(cls, data_df, model):
        '''
        Decode original data with model to keep only engaged trials
        '''
        raise NotImplementedError()
