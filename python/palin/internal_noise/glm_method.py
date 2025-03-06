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
from ..kernels.glm_kernel import GLMKernel
from .internal_noise_extractor import InternalNoiseExtractor

from abc import ABC,abstractmethod

class GLMMethod(InternalNoiseExtractor):
    """
    Implements a method to estimate internal noise based on the confidence intervals from a GLM fit.
    """

    def __str__(self):
        return "GLM Method"

    @classmethod
    def build_model(cls, glm_model_file='./glm_model.pkl', plot=False):
        """
        Fits an OLS regression model to map `norm_max_feature_ci` to `internal_noise_std`.

        """

        from ..simulation.analysers.ci_value import CIValue
        from ..simulation.simulation import Simulation as Sim
        from ..simulation.experiments.simple_experiment import SimpleExperiment
        from ..simulation.observers.linear_observer import LinearObserver
        from ..simulation.trial import Int2Trial 

        observer_params = {'kernel':['random'],
                           'internal_noise_std':np.arange(0,5.1,0.1), 
                                             'criteria':[0]}

        experiment_params = {'n_trials':np.arange(100,1000,100), 
                     'trial_type': [Int2Trial],
                     'n_features': [5],
                     'external_noise_std': [100]}

        analyser_params = {}        
                   
        sim = Sim(SimpleExperiment, experiment_params, 
                 LinearObserver, observer_params,
                 CIValue, analyser_params)
        sim_df = sim.run_all(n_runs=10)

        sim_df = sim_df.groupby(['internal_noise_std', 'n_trials']).confidence_interval.mean().reset_index()

        # Fit the OLS model using statsmodels.formula.api.ols
        # TODO: learn dependency on varying n_trials (internal_noise_std ~ norm_max_feature_ci*n_trials)
        model = smf.ols(formula="internal_noise_std ~ confidence_interval*n_trials", data=sim_df).fit()

        if plot: 
            sns.lmplot(x='confidence_interval', y='internal_noise_std', hue='n_trials', data=sim_df)

        model.save(glm_model_file)
        return model
        
    @classmethod
    def extract_norm_ci_value(cls, data_df, trial_id='trial', feature_id='feature', value_id='value', response_id='response'):
        
         # Use GLMKernel to fit GLM and extract kernel and confidence intervals
        model = GLMKernel.train_GLM_from_data(data_df=data_df,
            feature_id=feature_id,
            value_id=value_id,
            response_id=response_id
        )

        # extract conf intervals
        ci = model.conf_int()
        ci.columns = ['lower_bound', 'upper_bound']
        ci_df = ci.iloc[1:]  # Exclude the intercept
        ci_df = ci_df.reset_index(drop=True)

        kernel_df = GLMKernel.convert_model_to_kernel(model,feature_id)

        # Calculate confidence interval size
        kernel_df['conf_int'] = ci_df['upper_bound'] - ci_df['lower_bound']
        
        # Normalize confidence intervals
        kernel_df['norm_ci'] = kernel_df['conf_int'] / kernel_df['kernel_value'].abs()

        # Calculate normalized maximum feature confidence interval
        max_feature_ci = kernel_df['norm_ci'].iloc[np.argmax(kernel_df['kernel_value'].abs())]
        
        # scale by nb of trials
        norm_max_feature_ci = max_feature_ci * np.sqrt(data_df[trial_id].nunique()) 
        
        return norm_max_feature_ci, max_feature_ci
        
    
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

        if 'glm_model_file' not in kwargs:
            raise ValueError('no model file provided for GLM Method. Use GLMMethod.build_model() before calling') 

        # extract CI on weights from a GLM fit 
        norm_max_feature_ci, max_feature_ci=cls.extract_norm_ci_value(data_df, trial_id, feature_id, value_id, response_id)

        # convert to internal noise 
        model_file = kwargs['glm_model_file']
        if not os.path.isfile(model_file): 
            raise ValueError('unvalid model file provided for GLM Method') 
        else: 
            model = sm.load(model_file)
            ci_df = pd.DataFrame({'confidence_interval': [norm_max_feature_ci], 
                'n_trials': [data_df[trial_id].nunique()],})
            ci_df = sm.add_constant(ci_df)
            internal_noise = model.predict(ci_df).iloc[0]

            # note: to get confidence intervals on estimated noise, do: 
            # pred = model.get_prediction(ci_df)
            # pred.summary_frame(alpha=0.05) 

        return internal_noise

    