import numpy as np
import pandas as pd
#from typing import Literal 


def index_double_pass_trials(data_df, session_identifiers=['experimentor','type','subject','session'], trial_identifier='trial',response_identifier='double_pass_id',value_identifier='stim_parameter_id'):
	''' utility to identify repeated trials in experimental sessions (i.e. 'double pass trials'), and tag them with a unique id stored in a new column. 
	repeated trials are identified based on the features of their intervals (e.g. pitch values for each sound in trials consisting of pairs of sounds), and compared as unordered sets 
	(so that e.g. trials consisting of the same 2 sounds in a different order are considered repeated). 
	session_identifiers: list of columns within which double_pass_trials should be searched (typically: subject, but also perhaps session)
	trial_identifier: the unique identifier of trials within one session
	response_identifier: the name of the unique double_pass trial id
	value_identifier: trial features on which basis repeated trials are identified (trials are considered repeated if they have the same content in this column)  
	'''

	# represent the several value_identifier values of a given trial (ex. 6 features for interval 1, 6 features for interval 2) as a frozenset
	stimuli = data_df.groupby(session_identifiers+[trial_identifier]).agg({value_identifier: lambda group: frozenset(group)}).reset_index()

	# count how many trials have each unique pair of stimuli
	pass_count = stimuli.groupby(session_identifiers+[value_identifier]).agg({trial_identifier: ['nunique','first','last']})
	pass_count.columns = ["_".join(x) for x in pass_count.columns.ravel()]
	pass_count = pass_count.reset_index()

	# identify pairs of stimuli that have 2 trials (i.e. for which there has been a double pass)
	double_pass_stim = pass_count[pass_count.trial_nunique==2]

	# assign unique id to each double_pass within each subject
	double_pass_stim[response_identifier]=double_pass_stim.groupby(session_identifiers)['%s_nunique'%trial_identifier].cumcount()

	# join to base dataset
	double_pass_stim = double_pass_stim.melt(id_vars=session_identifiers+[response_identifier], value_vars=['%s_first'%trial_identifier,'%s_last'%trial_identifier], var_name='%s_type'%trial_identifier, value_name=trial_identifier)
	data_df= pd.merge(data_df, double_pass_stim[session_identifiers+[trial_identifier]+[response_identifier]], how="left", on=session_identifiers+[trial_identifier])
	return data_df	



def compute_prob_agreement(data_df, session_identifiers=['experimentor', 'type', 'subject', 'session'], trial_identifier='trial', response_identifier='response', order_identifier='stim_order', double_pass_identifier='double_pass_id'):
	# computes the probability of agreement between two responses to a repeated stimuli on the double pass trials 

	# we have to verify that required columns are present in the dataframe like double_pass and response_identifier and if the response column contains only numeric values. If not, it raises a ValueError.
    if double_pass_identifier not in data_df.columns:
        raise ValueError(f"Column {double_pass_identifier} not found in data_df")
    if response_identifier not in data_df.columns:
        raise ValueError(f"Column {response_identifier} not found in data_df")
    if not data_df[response_identifier].dtype.kind == 'i':
        data_df[response_identifier] = pd.to_numeric(data_df[response_identifier], errors='coerce').astype('Int64')
    if data_df[response_identifier].isna().any():
        raise ValueError(f"Column {response_identifier} contains missing or non-numeric values")

    # grouping the input DataFrame by the trial IDs, double-pass ID, dimension ID, and order, and computing the mean response for each group    
    trial_response_mean_df  = data_df[data_df[double_pass_identifier].notna()].groupby(session_identifiers + [double_pass_identifier] + [trial_identifier] + [order_identifier])[response_identifier].mean().reset_index()
    #the response column contains only numeric values
    trial_response_mean_df [response_identifier] = trial_response_mean_df [response_identifier].astype(int)

    # The order column is assumed to contain either 0 or 1, indicating which response was made first and which was made second
    trial_responses_df = trial_response_mean_df [trial_response_mean_df [order_identifier] == 0].groupby(session_identifiers + [double_pass_identifier]).agg({response_identifier: lambda group: list(group)}).reset_index()
    trial_responses_df['response'] = trial_responses_df['response'].astype(str)
    
    # split the response column into two separate columns based on the order
    trial_responses_df = trial_responses_df.join(trial_responses_df[response_identifier].str.split(expand=True).rename(columns={0: '%s1' % response_identifier, 1: '%s2' % response_identifier}))
    
    #clean the response data by removing any non-numeric characters
    trial_responses_df['%s1' % response_identifier] = trial_responses_df['%s1' % response_identifier].str.replace(r'\D', '')
    trial_responses_df['%s2' % response_identifier] = trial_responses_df['%s2' % response_identifier].str.replace(r'\D', '')
    
    # drop the original response column
    trial_responses_df = trial_responses_df.drop(columns=[response_identifier])

    #counting the number of unique double-pass IDs for each trial and save the result in a new dataframe
    unique_double_pass_df = trial_responses_df.groupby(session_identifiers).agg(sum_double_pass=(double_pass_identifier, 'nunique')).reset_index()
    
    #merge the previously created dataframes, count the number of trials where the responses agree, and save the result in a new dataframe
    trial_responses_df = pd.merge(trial_responses_df, unique_double_pass_df, how='left', on=session_identifiers)
    agreement_size_df = trial_responses_df[trial_responses_df[['%s1' % response_identifier, '%s2' % response_identifier]].nunique(axis=1) == 1].groupby(session_identifiers, as_index=False)['%s1' % response_identifier].size().rename(columns={'size': 'size_agree'})
    
    #compute the probability of agreement for each trial and save the result in a new column of the input dataframe
    trial_responses_df = pd.merge(trial_responses_df, agreement_size_df, how='left', on=session_identifiers)
    trial_responses_df['pc_agree'] = trial_responses_df['size_agree'] / trial_responses_df['sum_double_pass']
    
    #merge the computed probabilities of agreement with the original dataframe and return the merged dataframe
    merged_df=data_df.merge(trial_responses_df[session_identifiers + [double_pass_identifier] + ['sum_double_pass', 'size_agree', 'pc_agree']], how='left', on=session_identifiers + [double_pass_identifier])

    return merged_df


def compute_prob_interval1(data_df, session_identifiers=['experimentor', 'type', 'subject', 'session'], trial_identifier='trial', response_identifier='response', order_identifier='stim_order', double_pass_identifier='double_pass_id', p_int_1_identifier = 'p_int1'):
	''' Computes probability that each subject defined unique in session_identifiers responds true to the first interval, i.e. to the stimulus in each trial identified by order_identifier = 0
	This computes p_int1 only on the subset of repeated trials, and assumes that the dataset already has a column (e.g. double_pass_id) identifying repeated trials. Use utils.index_double_pass_trials to create that column if doesn't exist. 
	'''

	# Select only the double-pass trials
	double_pass_trials = data_df[data_df[double_pass_identifier].notna()]

	#Compute the total number of trials per subject
	nb_total_trials = double_pass_trials.groupby(session_identifiers)[trial_identifier].nunique().reset_index()

	#Compute the number of int1 trials per subject, i.e. trials where stimulus #0 got response "true"
	nb_int1 = double_pass_trials[(double_pass_trials[order_identifier]==0) & (double_pass_trials[response_identifier]==True)].groupby(session_identifiers)[trial_identifier].nunique().reset_index()
	# note: this returns nan instead of zeros; filling nans by zeros in the merge below

	#Merge the number of int1 and total trials and compute p_int1
	merged_data  = pd.merge(nb_total_trials,nb_int1,how='left',on=session_identifiers, suffixes=('_total','_int1'))
	# replace int1 missing values in by zeros
	merged_data = merged_data.fillna({trial_identifier+'_int1':0})
	# compute p_int1 as nb_int1 / nb_total
	merged_data [p_int_1_identifier] = merged_data ['%s_int1' % trial_identifier] / merged_data ['%s_total' % trial_identifier]
	
	return merged_data[session_identifiers+[p_int_1_identifier]] 


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