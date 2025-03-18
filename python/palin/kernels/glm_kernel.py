#!/usr/bin/env python
'''
PALIN toolbox v0.1
December 2022, Aynaz Adl Zarrabi, JJ Aucouturier (CNRS/UBFC)

'''
import rpy2.robjects as robjects
import rpy2.robjects.pandas2ri
from rpy2.robjects.packages import importr

rpy2.robjects.pandas2ri.activate()
stats = importr('stats')  # Load the 'stats' package (contains glm) 
base = importr('base')  # Load the 'base' package


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial
from statsmodels.api import Logit, add_constant

from statsmodels.genmod.families.links import probit, logit
from .kernel_extractor import KernelExtractor

class GLMKernel(KernelExtractor):
    """
    This class extracts kernel weights using a Generalized Linear Model (GLM).
    """

    @classmethod
    def extract_single_kernel(cls, data_df, trial_id='trial',stim_id='stim', feature_id='feature', value_id='value', response_id='response', **kwargs):
        """
        Extracts a single kernel by fitting a GLM to the given data.
        Returns a DataFrame with feature IDs and kernel values.
        """
        
        model = cls.train_GLM_from_data(data_df, trial_id, stim_id, feature_id, value_id, response_id, **kwargs)
        kernel = cls.convert_model_to_kernel(model, backend=kwargs.get("backend", "python"))
        
        return kernel
    
    @classmethod
    def convert_model_to_kernel(csl,model, backend="python"):
        # Extract coefficients
        feature_id='feature'
        if backend == "rpy2":
            coefs = robjects.r['coef'](model)
            kernel = pd.DataFrame({feature_id: range(len(coefs) - 1), 'kernel_value': list(coefs)[1:]}).set_index(feature_id)
        else:   
            coefs = model.params[1:]  # Exclude the intercept
            kernel = pd.DataFrame({feature_id: range(len(coefs)), 'kernel_value': coefs.values}).set_index(feature_id)
        return kernel


    @classmethod
    def train_GLM_from_data(cls, data_df, trial_id='trial',stim_id='stim', feature_id='feature', value_id='value', response_id='response', **kwargs):
        backend = kwargs.get("backend", "python")
        # Warning: this won't work for 1-int data (and no real way to know whether it's the case)
        link_function = kwargs.get("link", "probit")

        # Format data_df for training: values as columns, subtract values within each trial 
        pivoted_df = data_df.pivot_table(index=[trial_id, stim_id, response_id],
                                         columns=[feature_id],
                                         values=[value_id],
                                         aggfunc='first').reset_index()
        pivoted_df.columns = [trial_id, stim_id, response_id] + [(value_id+'%d'%col) for col in range(data_df[feature_id].nunique())]
        pivoted_df[response_id] = pivoted_df[response_id].astype(int)
        df_stim0 = pivoted_df[pivoted_df[stim_id] == 0].set_index(trial_id)
        df_stim1 = pivoted_df[pivoted_df[stim_id] == 1].set_index(trial_id)
        df_diff = df_stim1.filter(like=value_id).subtract(df_stim0.filter(like=value_id)).add_prefix('diff_')
        df_diff[response_id] = df_stim1[response_id].values
        preprocessed_data = df_diff.reset_index()
        if backend in ["python", "statsmodels"]:
            # Keep jitter for statsmodels
            if 'jitter' in kwargs:
                jitter = kwargs['jitter']
            else:
                jitter = 0.01
            preprocessed_data = cls._add_jitter(preprocessed_data, jitter)

        if backend == "rpy2":
            r_df = rpy2.robjects.pandas2ri.py2rpy(preprocessed_data)
            formula = robjects.Formula('response ~ ' + ' + '.join(preprocessed_data.filter(like="diff_").columns))
            link = "probit" if link_function == "probit" else "logit"
            model = stats.glm(formula=formula, data=r_df, family=stats.binomial(link=link))
        else:
            link = probit() if link_function == "probit" else logit()
            formula = f'{response_id} ~ ' + ' + '.join(preprocessed_data.filter(like="diff_").columns)
            model = smf.glm(formula=formula, data=preprocessed_data, family=Binomial(link=link)).fit()
        
        return model
        # # add jitter to cover for simulated data with obs with 0 internal noise
        # if 'jitter' in kwargs:
        #     jitter = kwargs['jitter']
        # else: 
        #     jitter= 0.01
        # preprocessed_data = cls._add_jitter(preprocessed_data, jitter)
        
        # # Fit the GLM
        # formula = f'{response_id} ~ {" + ".join(preprocessed_data.filter(like="diff_").columns)}'

        # try:
        #     # X = preprocessed_data.filter(like="diff_")
        #     # X = add_constant(X)  # Add intercept term
        #     # y = preprocessed_data[response_id]

        #     # model = Logit(y, X).fit()
        #     model = smf.glm(formula=formula, data=preprocessed_data, family=Binomial(link=probit())).fit()
        #     # model = LogisticRegression(solver='lbfgs').fit(X, y)  # Use lbfgs for better convergence
     

        # except Exception as e:
        #     print('Error fitting GLM:', e)
        #     print(preprocessed_data)
        #     model = None

        # return model
        
    @classmethod
    def _add_jitter(cls, data_df, jitter, trial_id = 'trial', response_id = 'response'): 
        ''' 
        This adds random variation to a GLM-formatted dataframe, to avoid numerical errors when the data is generated without internal noise
        '''

        if jitter == 0: 
            n_jitter = 0
        else: 
            n_jitter = max(5,int(np.ceil(jitter*data_df[trial_id].nunique())))
        #print('Adding jitter to %d trials'%n_jitter)
        # select n_jitter smallest trials by trial intensity (don't randomize high-intensity trials)
        data_df['trial_intensity'] = data_df.filter(like='diff_').abs().sum(axis=1)
        selection = data_df.sort_values('trial_intensity').head(n_jitter).index
        # flip response
        data_df.loc[selection,response_id] = 1 - data_df.loc[selection,response_id]
        return data_df    

    def __str__(self):
        return 'GLM Kernel'


     