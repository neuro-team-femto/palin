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
        self.run_params = self.generate_runs(self.experiment_params,self.observer_params,self.analyser_params)
        print("generated %d runs"%len(self.run_params))

    @classmethod
    def generate_runs(cls, experiment_params, observer_params, analyser_params): 
        # construct simulation plan
        sim_params ={}
        for d in (experiment_params, observer_params, analyser_params): 
            sim_params.update(d)
        keys, values = zip(*sim_params.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]


    def run_all(self, n_samples): 
        runs = []
        for run_param in self.run_params: 
            print(run_param)
            for sample in np.arange(n_samples): 
                print('.',end='')
                res = self.run(run_param)
                run_res = run_param.copy() 
                run_res.update({'sample':sample, 'metric':res})
                runs.append(run_res)
            print('')
        return pd.DataFrame(runs)


    def run(self, run_param): 

        # separate this run's parameters into distinct sets
        run_experiment_params = {k: v for k, v in run_param.items() if k in self.experiment_params}
        run_observer_params = {k: v for k, v in run_param.items() if k in self.observer_params}
        run_analyser_params = {k: v for k, v in run_param.items() if k in self.analyser_params}

        exp = self.experiment(**run_experiment_params)
        obs = self.observer(**run_observer_params)
        ana = self.analyser(**run_analyser_params)

        responses = obs.respond_to_experiment(exp)
        
        return ana.analyse(exp, obs, responses)

