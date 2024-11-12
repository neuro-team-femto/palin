from abc import ABC, abstractmethod
import itertools
import pandas as pd
import numpy as np
from .observer import Observer
from .experiment import Experiment
from .analyser import Analyser

import multiprocessing as mp
from tqdm import tqdm

import os

class Simulation(ABC): 
    '''
    Class that implements a simulation, i.e. a range of simulated @Observers that respond to @Experiments and whose results are analysed with an @Analyser.
    Simulations are initiated with class names (e.g. SimpleExperiment, LinearObserver, KernelDistance) and a range of parameters that are looped across when run. 
    Results are returned in a panda dataframe 
    '''

    def __init__(self, experiment_class, experiment_params, observer_class, observer_params, analyser_class, analyser_params): 
        '''
        Initialize a Simulation with class names than implement Experiment, Observer and Analyser, and parameters that define the different configs in which the simulation is run.  
        Upon running a config, one observer responds to one experiment, and their responses are analysed with one analyser. 
        When initializing the simulation, each class name is associated with a dictionary of parameters whose keys are the arguments of the __init__() method of the corresponding class name.
        For instance, if observer_class is LinearObserver, observer_params should be a dictionary with keys kernel,internal_noise_std and criteria.
        Values for each key should be an iterable (typically a list) of values, which define the different configs. 

        Example: 
        observer_params = {'kernel':['random'],'internal_noise_std':[np.arange(0,10,1)],'criteria':[0]}
        experiment_params = {'n_trials':[np.arange(1,1000,100)], 'n_repeated':[100], 'trial_type': [Int2Trial],
                     'n_features': [5], 'external_noise_std': [100]}
        analyser_params = {'internal_noise_extractor':[DoublePass], 'model_file': ['model.csv']}
        sim = Sim(DoublePassExperiment, experiment_params,
                  LinearObserver, observer_params, 
                  InternalNoiseValue, analyser_params)
        sim.run_all(n_runs=1) 

        will run a simulation where LinearObservers respond to DoublePassExperiments, and computes their InternalNoiseValue, 
        in 100 different configurations (observers with true internal noise values between 0..10; and experiments with 1..1000 trials).
        Upon running, each of the configuration is run n_runs times, and separate results are stored for each run. 
        '''
        
        if not issubclass(experiment_class,Experiment):
            raise TypeError('argument experiment_class %s does not implement an Experiment')
        self.experiment = experiment_class
        
        if not issubclass(observer_class,Observer):
            raise TypeError('argument observer_class %s does not implement an Observer')
        self.observer = observer_class
        
        if not issubclass(analyser_class,Analyser):
            raise TypeError('argument analyser_class %s does not implement an Analyser')
        self.analyser = analyser_class
        
        # construct simulation plan
        self.experiment_params = experiment_params
        self.observer_params = observer_params
        self.analyser_params = analyser_params
        self.config_params = self.generate_configs(self.experiment_params,self.observer_params,self.analyser_params)
        
    def generate_configs(self, experiment_params, observer_params, analyser_params): 
        '''
        Generates all combinations of parameters from the constructor's parameters. 
        Each of these combinations will be individual configs that the Simulation will then run.
        '''
        # construct simulation plan
        sim_params ={}
        for d in (experiment_params, observer_params, analyser_params): 
            sim_params.update(d)
        keys, values = zip(*sim_params.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def run_all(self, n_runs, multiprocess=True):
        '''
        Run all configs stored in self.config_params. Each config is run n_runs times, and separate results are stored for each run. 
        Each run instanciates one Observer, one Experiment and one Analyser (see @run).
        Results are returned into a dataframe; each row is a run, and columns store config parameters, run number and analyser results. 
        '''
        if multiprocess:
            return self.run_all_multi_process(n_runs)
        else:
            return self.run_all_single_process(n_runs)


    def repeat_configs(self,n_runs): 
        '''
        n_repeats the self.config_params list by duplicating each config n_runs time, and including a 'run' field [0...n_runs-1] in each config
        '''
        # convert list of configs to dataframe
        configs_df = pd.DataFrame(self.config_params)
        
        # add an id to each original config
        configs_df = configs_df.reset_index(names='config')
        
        # repeat each config n_runs time, and add a run counter
        runs_df = pd.DataFrame(np.repeat(configs_df.values,n_runs, axis=0), columns=configs_df.columns)
        runs_df['run'] = 1
        runs_df['run'] = runs_df.groupby('config').run.cumsum() - 1
        
        # convert back to list of dicts
        runs = runs_df.to_dict('records')

        return runs


    def run_all_single_process(self, n_runs): 
        '''
        Single process implementation of run_all. Slow, but simple.
        '''
        #if verbose: 
        #    print("Running %d configs"%len(self.config_params))

        runs = self.repeat_configs(n_runs)
        run_results = []

        progress_runs = tqdm(runs)
        for run in progress_runs:
            run_result = self.run(run)
            progress_runs.set_description("Processing config %d run %d" %(run['config'],run['run']))
            run_results.append(run_result)

        return pd.DataFrame(runs)



    def run_all_multi_process(self, n_runs): 
        '''
        Multiprocess implementation of run_all. Fast, possibly brittle depending on OS. 
        '''
        runs = self.repeat_configs(n_runs)
        run_results = []
        
        with mp.Pool(processes=mp.cpu_count()) as pool: 

            for run_result in tqdm(pool.imap_unordered(self.run, runs), total=len(runs)): 
                run_results.append(run_result)

        return pd.DataFrame(run_results)

        
    


    def run(self, config_param): 
        '''
        Perform individual run for a config defined by config_param. 
        Each run instanciates one Observer, one Experiment and one Analyser. 
        The observer responds to the experiment, and their responses are analysed with the analyser. 
        Results are then returned in a dictionary of metric_name:value pairs, which include a copy of the config parameters. 
        '''

        # separate this run's parameters into distinct sets
        config_experiment_params = {k: v for k, v in config_param.items() if k in self.experiment_params}
        config_observer_params = {k: v for k, v in config_param.items() if k in self.observer_params}
        config_analyser_params = {k: v for k, v in config_param.items() if k in self.analyser_params}

        exp = self.experiment(**config_experiment_params)
        obs = self.observer(**config_observer_params)
        ana = self.analyser(**config_analyser_params)

        responses = obs.respond_to_experiment(exp)
        
        metrics =  ana.get_metric_names()
        values = ana.analyse(exp, obs, responses)
        
        # return the metrics as a dict of name:value pairs
        results = config_param
        for metric,value in zip(metrics,values): 
            results[metric] = value
        return results


