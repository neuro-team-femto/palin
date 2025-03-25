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
import warnings

import importlib.util

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
    def build_model(cls, agg_mode = 'argmax', backend = 'python', link = 'logit', plot=False):
        """
        Fits an OLS regression model to map `norm_max_feature_ci` to `internal_noise_std`.

        """
        from ..simulation.analysers.ci_values import CIValues
        from ..simulation.simulation import Simulation as Sim
        from ..simulation.experiments.simple_experiment import SimpleExperiment
        from ..simulation.observers.linear_observer import LinearObserver
        from ..simulation.trial import Int2Trial 

        observer_params = {'kernel':['random'],
                           'internal_noise_std':np.arange(0,5.1,0.1), 
                                             'criteria':[0]}

        experiment_params = {'n_trials':np.arange(100,2000,100), 
                     'trial_type': [Int2Trial],
                     'n_features': [5],
                     'external_noise_std': [100]}

        # FIXME: use parameters for backend and link
        analyser_params = {'agg_mode': [agg_mode], 'backend':[backend], 'link':[link], 'jitter':[0.01]} 
                   
        sim = Sim(SimpleExperiment, experiment_params, 
                 LinearObserver, observer_params,
                 CIValues, analyser_params)
        sim_df = sim.run_all(n_runs=10)

        sim_df = sim_df.groupby(['internal_noise_std', 'n_trials']).confidence_interval.mean().reset_index()

        # Fit the OLS model using statsmodels.formula.api.ols
        # TODO: learn dependency on varying n_trials (internal_noise_std ~ norm_max_feature_ci*n_trials)
        model = smf.ols(formula="internal_noise_std ~ confidence_interval*n_trials", data=sim_df).fit()

        if plot: 
            sns.lmplot(x='confidence_interval', y='internal_noise_std', hue='n_trials', data=sim_df)

        glm_model_file = f'./glm_model_{backend}_{link}_{agg_mode}.pkl'
        model.save(glm_model_file)
        return model

    @classmethod
    def import_rpy2(cls): 
        '''
        Utility function to import rpy2 conditionally, so it's not required when the module is loaded
        '''
        rpy2_spec = importlib.util.find_spec("rpy2")
        if rpy2_spec is not None: 
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    import rpy2.robjects as robjects
                    import rpy2.robjects.pandas2ri
                    from rpy2.robjects.packages import importr
                    rpy2.robjects.pandas2ri.activate()
                    stats = importr('stats')  # Load the 'stats' package (contains glm) 
                    base = importr('base')  # Load the 'base' package
            except ImportError:
                raise ImportError('Cannot import rpy2')
        else: 
            raise ImportError('Cannot import rpy2')
        return robjects, stats
        
    @classmethod
    def extract_norm_ci_value(cls, data_df, trial_id='trial', stim_id= 'stim', feature_id='feature', value_id='value', response_id='response', **kwargs):
         # Use GLMKernel to fit GLM and extract kernel and confidence intervals
        
        if 'backend' not in kwargs:
            raise TypeError('GLMMethod missing required argument backend')
        backend = kwargs['backend']

        if 'link' not in kwargs:
            raise TypeError('GLMMethod missing required argument link')
        link_function = kwargs['link']

        if 'agg_mode' not in kwargs:
            raise TypeError('GLMMethod missing required argument agg_mode')
        agg_mode = kwargs['agg_mode']

        model = GLMKernel.train_GLM_from_data(data_df,
            trial_id,stim_id, feature_id, value_id, response_id, **kwargs)

        if model is None:
            return np.nan,np.nan
        
        if backend == "rpy2":

            robjects, stats = cls.import_rpy2()
            coefs = robjects.r['coef'](model)
            if any(np.isnan(coefs)):  # Ensure model coefficients are valid
                print("Warning: R model contains NaN coefficients. Returning NaN.")
                return np.nan,np.nan
            try:
                ci = stats.confint(model)
                ci_array = np.array(ci)
        
                if ci_array.shape[0] < 2 or np.isnan(ci_array).any():  # Ensure valid values
                    # print("Warning: Not enough valid confidence interval values in R. Returning NaN.")
                    return np.nan,np.nan

                ci_df = pd.DataFrame(ci_array, columns=['lower_bound', 'upper_bound'])
                ci_df = ci_df.iloc[1:].reset_index(drop=True)  # Exclude the intercept
            except Exception as e:
                print(f"Error computing confidence intervals in R: {e}")
                return np.nan,np.nan
        elif backend == "python": 

            ci = model.conf_int()
            ci.columns = ['lower_bound', 'upper_bound']
            ci_df = ci.iloc[1:].reset_index(drop=True)  # Exclude the intercept

        else: 
            raise ValueError('Unrecognized backend: %s'%backend)

        kernel_df = GLMKernel.convert_model_to_kernel(model, feature_id = feature_id, **kwargs)

        # Calculate confidence interval size
        kernel_df['conf_int'] = ci_df['upper_bound'] - ci_df['lower_bound']
        
        # Normalize confidence intervals
        kernel_df['norm_ci'] = kernel_df['conf_int'] / kernel_df['kernel_value'].abs()

        if agg_mode == "min":
            agg_norm_ci = kernel_df['norm_ci'].min()
        elif agg_mode == "mean":
            agg_norm_ci = kernel_df['norm_ci'].mean()
        elif agg_mode == "median":
            agg_norm_ci = kernel_df['norm_ci'].median()
        elif agg_mode == "max":
            agg_norm_ci = kernel_df['norm_ci'].max()
        elif agg_mode == "argmedian" or agg_mode == "argmean":
            agg_norm_ci = kernel_df['norm_ci'].iloc[cls.argmedian(list(kernel_df['kernel_value'].abs()))]
        elif agg_mode == "argmin":
            agg_norm_ci = kernel_df['norm_ci'].iloc[np.argmin(list(kernel_df['kernel_value'].abs()))]
        elif agg_mode == "argmax":
            agg_norm_ci = kernel_df['norm_ci'].iloc[np.argmax(list(kernel_df['kernel_value'].abs()))]
        else:
            raise ValueError(f"Unknown aggregation mode: {agg_mode}")

        # Calculate normalized maximum feature confidence interval
        # max_feature_ci = kernel_df['norm_ci'].iloc[np.argmax(kernel_df['kernel_value'].abs())]
        
        # scale by nb of trials
        norm_max_feature_ci = agg_norm_ci * np.sqrt(data_df[trial_id].nunique()) 
        
        return norm_max_feature_ci, agg_norm_ci

    @classmethod
    def argmedian(cls, x):
        return np.argpartition(x, len(x) // 2)[len(x) // 2]  
    
    @classmethod
    def extract_single_internal_noise(cls, data_df, trial_id='trial', stim_id ='stim', feature_id='feature', value_id='value', response_id='response', **kwargs):
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
        #if 'glm_model_file' not in kwargs:
        #    raise ValueError('no model file provided for GLM Method. Use GLMMethod.build_model() before calling') 
        
        # extract CI on weights from a GLM fit 
        norm_max_feature_ci, agg_norm_ci =cls.extract_norm_ci_value(data_df, trial_id, stim_id, feature_id, value_id, response_id, **kwargs)
        #norm_max_feature_ci =cls.extract_norm_ci_value(data_df, trial_id, stim_id, feature_id, value_id, response_id, agg_mode,**kwargs)

        if np.isnan(norm_max_feature_ci) or norm_max_feature_ci < 0:
            # print(f"Warning: Invalid norm_max_feature_ci value ({norm_max_feature_ci}). Returning NaN.")
            return np.nan

        # convert to internal noise 
        if 'agg_mode' not in kwargs:
            raise TypeError('GLMMethod missing required argument agg_mode')
        agg_mode = kwargs['agg_mode']

        ## FIXME: add backend and link dans model name
        model_file = f'./glm_model_{backend}_{link}_{agg_mode}.pkl'
        if not os.path.isfile(model_file): 
            raise ValueError(f"Invalid model file {model_file}. Run `build_model(agg_mode='{agg_mode}')` first.") 

       
        model = sm.load(model_file)
        ci_df = pd.DataFrame({'confidence_interval': [norm_max_feature_ci], 'n_trials': [data_df[trial_id].nunique()],})
        ci_df = sm.add_constant(ci_df)
        internal_noise = model.predict(ci_df).iloc[0]

            # note: to get confidence intervals on estimated noise, do: 
            # pred = model.get_prediction(ci_df)
            # print(f"Backend: {backend}, Norm Max CI: {norm_max_feature_ci}, Predicted Internal Noise: {internal_noise}")

        return internal_noise

    