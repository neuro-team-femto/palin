from abc import ABC, abstractmethod

import pandas as pd


class Analyser(ABC): 

    @abstractmethod
    def analyse(self,experiment, participant, participant_responses): 
        raise NotImplementedError()

    @abstractmethod
    def get_metric_names(self):
        raise NotImplementedError()

    @classmethod
    def to_df(csl, experiment, responses): 

        trial_ids = []
        stim_orders = []
        features = []
        values = []
        resps  = []
    
        for num_trial, trial in enumerate(experiment.trials):
            response = responses[num_trial]
            for num_stim, stim in enumerate(trial.stims): 
                for num_feature, value in enumerate(stim):
                    trial_ids.append(num_trial)
                    stim_orders.append(num_stim)
                    features.append(num_feature)
                    values.append(value)
                    resps.append(True if response == num_stim else False)
    
        return pd.DataFrame.from_dict({'trial': trial_ids,
                                   'stim': stim_orders,
                                   'feature': features,
                                   'value': values, 
                                   'response': resps})

