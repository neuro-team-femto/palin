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
from .internal_noise_extractor import InternalNoiseExtractor
from abc import ABC,abstractmethod

class GLMMethod(InternalNoiseExtractor):
    ''' 
    This class implements method to estimate internal noise from a GLM fit, based on the confidence interval on the GLM weights'''

    def __str__(self): 
        return 'GLM method'

    @classmethod
    def extract_single_internal_noise(cls,data_df, trial_id, stim_id, feature_id, value_id, response_id, model_file, rebuild_model=False, internal_noise_range=np.arange(0,5,.1),criteria_range=np.arange(-5,5,1), n_repeated_trials=100, n_runs=10):
        '''
        Extracts internal noise for a single observer/session, using CI from a GLM fit. 
        '''
        
        # Fit GLM to data
        # model = glm.fit(data_df)

        # Extract confidence intervals
        # ci = model.conf_int()

        # Convert CI to IN
        # internal_noise = f(ci)

        return internal_noise
    