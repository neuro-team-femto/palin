import numpy as np

def compute_prob_agreement(data_df):
	return 0

def compute_prob_interval1(data_df):
	return 0


def simulate_observer(internal_noise_sigma,criteria, n_blocks, n_trials): 

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