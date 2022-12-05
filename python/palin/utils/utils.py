import numpy as np
import pandas as pd
from typing import Literal 

def index_double_pass_trials(data_df, stim_parameter_id):
	# create double_pass_id column by first identifying repeated trials with ordered set etc.
	# list all trials' pairs of stimuli (each pair is represented by a unique frozen set)  
	stimuli = data_df.groupby(["subj","trial"]).agg({stim_parameter_id: lambda group: frozenset(group)}).reset_index()

	# count how many trials have each unique pair of stimuli
	pass_count = stimuli.groupby(["subj",stim_parameter_id]).agg({'trial': ['nunique','first','last']})
	pass_count.columns = ["_".join(x) for x in pass_count.columns.ravel()]
	pass_count = pass_count.reset_index()

	# identify pairs of stimuli that have 2 trials (i.e. for which there has been a double pass)
	double_pass_stim = pass_count[pass_count.trial_nunique==2]

	# assign unique id to each double_pass within each subj
	double_pass_stim['double_pass_id']=double_pass_stim.groupby(['subj']).trial_nunique.cumcount()

	# join to base dataset
	double_pass_stim = double_pass_stim.melt(id_vars=['subj','double_pass_id'], value_vars=['trial_first','trial_last'], var_name='trial_type', value_name='trial')
	data_df= pd.merge(data_df, double_pass_stim[['subj','double_pass_id','trial']], how="left", on=["subj", "trial"])
	return data_df

def compute_prob_agreement(data_df,double_pass_id):
	# does this assume that the dataset has a "double_pass_id" column ? (yes)
	a = data_df[data_df.double_pass_id.notna()].groupby(['subj','double_pass_id','trial','stim_order']).response.mean().reset_index()
	a.response=a.response.astype(int)
	b=a[a.stim_order==0].groupby(['subj','double_pass_id']).agg({'response': lambda group: list(group)}).reset_index()
	b['response']=b['response'].astype(str)
	b=b.join(b['response'].str.split(expand=True).rename(columns={0:'response1',1:'response2'}))
	b['response1']=b['response1'].str.replace(r'\D', '')
	b['response2']=b['response2'].str.replace(r'\D', '')
	b = b.drop(columns=['response'])
	c=b.groupby(["subj"]).double_pass_id.nunique().reset_index().rename(columns={'double_pass_id':'sum_double_pass'})
	b=pd.merge(b, c, how="left", on=["subj"])
	d=b[b[['response1','response2']].nunique(axis=1) == 1].groupby('subj', as_index=False).size()
	b=pd.merge(b, d, how="left", on=["subj"])
	b['pc_agree']=b['size']/b['sum_double_pass']
	data_df= pd.merge(data_df, b[['subj','double_pass_id','sum_double_pass','size','pc_agree']], how="left", on=["subj", "double_pass_id"])
	return data_df


def compute_prob_interval1(data_df, whole=False):
	# does this assume that the dataset has a "double_pass_id" column ? (yes)
	# compute int1 on whole data, or only on trials with a non null double_pass_id
	double_pass_trials = data_df[data_df.double_pass_id.notna()]
	nb_int1 = double_pass_trials[(double_pass_trials.stim_order==0) & (double_pass_trials.response==True)].groupby(['subj','trial','stim_order']).response.mean().reset_index()
	nb_int1 = nb_int1.groupby('subj').trial.count().reset_index()
	nb_total_trials = double_pass_trials.groupby('subj').trial.nunique().reset_index()
	p_int1 = pd.merge(nb_int1,nb_total_trials,how='left',on='subj')
	p_int1['p_int1'] = p_int1.trial_x / p_int1.trial_y
	p_int1=p_int1.loc[:, ~p_int1.columns.isin(['trail_x', 'trial_y'])]
	p_int1=pd.merge(p_int1,double_pass_trials,how='left',on='subj')
	return p_int1


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