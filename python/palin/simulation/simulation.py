from abc import ABC, abstractmethod
import itertools
import pandas as pd
import numpy as np

class Simulation(ABC): 

    def __init__(self, experiment, experiment_params, observer, observer_params, analyser, analyser_params): 
        
        # store simulation classes
        self.experiment = experiment
        self.observer = observer
        self.analyser = analyser
        
        # construct simulation plan
        self.experiment_params = experiment_params
        self.observer_params = observer_params
        self.analyser_params = analyser_params
        self.config_params = self.generate_configs(self.experiment_params,self.observer_params,self.analyser_params)
        
    @classmethod
    def generate_configs(cls, experiment_params, observer_params, analyser_params): 
        # construct simulation plan
        sim_params ={}
        for d in (experiment_params, observer_params, analyser_params): 
            sim_params.update(d)
        keys, values = zip(*sim_params.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]


    def run_all(self, n_runs, verbose=True): 
        if verbose: 
            print("Running %d configs"%len(self.config_params))

        runs = []
        for config_param in self.config_params: 
            if verbose: 
                print(config_param)
            for run in np.arange(n_runs): 
                if verbose: 
                    print('.',end='')
                res = self.run(config_param)
                run_res = config_param.copy() 
                run_res.update({'run':run, 'metric':res})
                runs.append(run_res)
            if verbose: 
                print(';')
        return pd.DataFrame(runs)


    def run(self, config_param): 

        # separate this run's parameters into distinct sets
        config_experiment_params = {k: v for k, v in config_param.items() if k in self.experiment_params}
        config_observer_params = {k: v for k, v in config_param.items() if k in self.observer_params}
        config_analyser_params = {k: v for k, v in config_param.items() if k in self.analyser_params}

        exp = self.experiment(**config_experiment_params)
        obs = self.observer(**config_observer_params)
        ana = self.analyser(**config_analyser_params)

        responses = obs.respond_to_experiment(exp)
        
        return ana.analyse(exp, obs, responses)

