import numpy as np
import pandas as pd
#from typing import Literal 


def index_double_pass_trials(data_df,trial_ids=['experimentor','type','subject','session'] ,dimension_id='trial',response_id='double_pass_id',value_id='stim_parameter_id'):
	# create double_pass_id column by first identifying repeated trials with ordered set etc.
	# list all trials' pairs of stimuli (each pair is represented by a unique frozen set)  
	stimuli = data_df.groupby(trial_ids+[dimension_id]).agg({value_id: lambda group: frozenset(group)}).reset_index()

	# count how many trials have each unique pair of stimuli
	pass_count = stimuli.groupby(trial_ids+[value_id]).agg({dimension_id: ['nunique','first','last']})
	pass_count.columns = ["_".join(x) for x in pass_count.columns.ravel()]
	pass_count = pass_count.reset_index()

	# identify pairs of stimuli that have 2 trials (i.e. for which there has been a double pass)
	double_pass_stim = pass_count[pass_count.trial_nunique==2]

	# assign unique id to each double_pass within each subject
	double_pass_stim['double_pass_id']=double_pass_stim.groupby(trial_ids)['%s_nunique'%dimension_id].cumcount()

	# join to base dataset
	double_pass_stim = double_pass_stim.melt(id_vars=trial_ids+[response_id], value_vars=['%s_first'%dimension_id,'%s_last'%dimension_id], var_name='%s_type'%dimension_id, value_name=dimension_id)
	data_df= pd.merge(data_df, double_pass_stim[trial_ids+[dimension_id]+[response_id]], how="left", on=trial_ids+[dimension_id])
	return data_df	

import pandas as pd

def compute_prob_agreement(data_df, trial_ids=['experimentor', 'type', 'subject', 'session'], dimension_id='trial', response_id='response', order='stim_order', double_pass='double_pass_id'):
	# computes the probability of agreement between two responses to a repeated stimuli on the double pass trials 

	# we have to verify that required columns are present in the dataframe like double_pass and response_id and if the response column contains only numeric values. If not, it raises a ValueError.
    if double_pass not in data_df.columns:
        raise ValueError(f"Column {double_pass} not found in data_df")
    if response_id not in data_df.columns:
        raise ValueError(f"Column {response_id} not found in data_df")
    if not data_df[response_id].dtype.kind == 'i':
        data_df[response_id] = pd.to_numeric(data_df[response_id], errors='coerce').astype('Int64')
    if data_df[response_id].isna().any():
        raise ValueError(f"Column {response_id} contains missing or non-numeric values")

    # grouping the input DataFrame by the trial IDs, double-pass ID, dimension ID, and order, and computing the mean response for each group    
    trial_response_mean_df  = data_df[data_df[double_pass].notna()].groupby(trial_ids + [double_pass] + [dimension_id] + [order])[response_id].mean().reset_index()
    #the response column contains only numeric values
    trial_response_mean_df [response_id] = trial_response_mean_df [response_id].astype(int)

    # The order column is assumed to contain either 0 or 1, indicating which response was made first and which was made second
    trial_responses_df = trial_response_mean_df [trial_response_mean_df [order] == 0].groupby(trial_ids + [double_pass]).agg({response_id: lambda group: list(group)}).reset_index()
    trial_responses_df['response'] = trial_responses_df['response'].astype(str)
    
    # split the response column into two separate columns based on the order
    trial_responses_df = trial_responses_df.join(trial_responses_df[response_id].str.split(expand=True).rename(columns={0: '%s1' % response_id, 1: '%s2' % response_id}))
    
    #clean the response data by removing any non-numeric characters
    trial_responses_df['%s1' % response_id] = trial_responses_df['%s1' % response_id].str.replace(r'\D', '')
    trial_responses_df['%s2' % response_id] = trial_responses_df['%s2' % response_id].str.replace(r'\D', '')
    
    # drop the original response column
    trial_responses_df = trial_responses_df.drop(columns=[response_id])

    #counting the number of unique double-pass IDs for each trial and save the result in a new dataframe
    unique_double_pass_df = trial_responses_df.groupby(trial_ids).agg(sum_double_pass=(double_pass, 'nunique')).reset_index()
    
    #merge the previously created dataframes, count the number of trials where the responses agree, and save the result in a new dataframe
    trial_responses_df = pd.merge(trial_responses_df, unique_double_pass_df, how='left', on=trial_ids)
    agreement_size_df = trial_responses_df[trial_responses_df[['%s1' % response_id, '%s2' % response_id]].nunique(axis=1) == 1].groupby(trial_ids, as_index=False)['%s1' % response_id].size().rename(columns={'size': 'size_agree'})
    
    #compute the probability of agreement for each trial and save the result in a new column of the input dataframe
    trial_responses_df = pd.merge(trial_responses_df, agreement_size_df, how='left', on=trial_ids)
    trial_responses_df['pc_agree'] = trial_responses_df['size_agree'] / trial_responses_df['sum_double_pass']
    
    #merge the computed probabilities of agreement with the original dataframe and return the merged dataframe
    merged_df=data_df.merge(trial_responses_df[trial_ids + [double_pass] + ['sum_double_pass', 'size_agree', 'pc_agree']], how='left', on=trial_ids + [double_pass])

    return merged_df


def compute_prob_interval1(data_df,trial_ids=['experimentor', 'type', 'subject', 'session'], dimension_id='trial', response_id='response', order='stim_order', double_pass='double_pass_id', whole=False):
	# does this assume that the dataset has a "double_pass_id" column ? (yes)
	# compute int1 on whole data, or only on trials with a non null double_pass_id

	# Select only the double-pass trials
	double_pass_trials = data_df[data_df[double_pass].notna()]

	#Compute the number of int1 trials
	nb_int1 = double_pass_trials[(double_pass_trials[order]==0) & (double_pass_trials[response_id]==True)].groupby(trial_ids+[dimension_id]+[order])[response_id].mean().reset_index()
	nb_int1 = nb_int1.groupby(trial_ids)[dimension_id].count().reset_index()

	#Compute the total number of trials
	nb_total_trials = double_pass_trials.groupby(trial_ids)[dimension_id].nunique().reset_index()
	
	#Merge the number of int1 and total trials and compute p_int1
	merged_data  = pd.merge(nb_int1,nb_total_trials,how='left',on=trial_ids)
	merged_data ['p_int1'] = merged_data ['%s_x' % dimension_id] / merged_data ['%s_y' % dimension_id]
	merged_data =merged_data .loc[:, ~merged_data .columns.isin(['%s_x' % dimension_id, '%s_y' % dimension_id])]
	
	# Merge with the double pass trials and return the result
	result_df =pd.merge(merged_data ,double_pass_trials,how='left',on=trial_ids)
	return result_df 


def simulate_observer(internal_noise_sigma,criteria, n_trials, n_blocks=1): 

	# simulate observer with (criteria, internal_noise_sigma)

	# each trial is composed of two (random) signals (interval1 & interval2)
	signal_interval1 = np.random.normal(size=(n_blocks,n_trials))
	signal_interval2 = np.random.normal(size=(n_blocks,n_trials))
	
	# responses in each pass are modified by a random draw of internal noise for each trial and pass 
	internal_noise_interval1_pass1 = internal_noise_sigma*np.random.normal(size=(n_blocks,n_trials))
	internal_noise_interval2_pass1 = internal_noise_sigma*np.random.normal(size=(n_blocks,n_trials))
	internal_noise_interval1_pass2 = internal_noise_sigma*np.random.normal(size=(n_blocks,n_trials))
	internal_noise_interval2_pass2 = internal_noise_sigma*np.random.normal(size=(n_blocks,n_trials))

	all_response_pass1 = (signal_interval1 + internal_noise_interval1_pass1) > criteria + \
						(signal_interval2 + internal_noise_interval2_pass1)
	all_response_pass2 = (signal_interval1 + internal_noise_interval1_pass2) > criteria + \
						(signal_interval2 + internal_noise_interval2_pass2)

	# probability interval 1 (average of prob in both pass)
	prob_interval1 = (np.mean(all_response_pass1) + np.mean(all_response_pass2))/2
	
	#probability of agreement between pass
	prob_agreement = np.mean(all_response_pass1==all_response_pass2) 
	
	return prob_agreement,prob_interval1