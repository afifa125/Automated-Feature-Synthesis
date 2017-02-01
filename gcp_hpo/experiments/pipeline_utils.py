"""
Utilities for the basic_pipeline and MNIST_pipeline.

"""
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def do_pca(X,num_components,fit,fitted_model=None):
    if fit:
        pca = PCA(n_components = num_components)
        pca = pca.fit(X)
        return pca,pca.transform(X)
    else:
        return fitted_model.transform(X)

def random_forest(X,y,num_estimators):
    clf = RandomForestClassifier(n_estimators=num_estimators)
    clf.fit(X,y)
    # print 'accuracy on training set',clf.score(X,y)
    return clf

def blb(X,y,p_dict,calculate_statistic,score_fnc_args=None):
	"""
	Args:
		calculate_statistic: 	function that calculates accuracy/F1 score/value in question
		score_fnc_args:    		tuple of arguments to the calculate_statistic function
	"""
	# TODO look to see how you can incorporate an optional number of additional parameters
		# and even put those parameters in a list or something that must be passed in

	#### Define variables ######
	n = X.shape[0] # Size of data
	b = int(n**(0.7)) # subset size
	s = 1 # Number of sampled subsets
	r = 20 # Number of Monte Carlo iterations

	#### Algorithm ####
	subsample_estimate_sum = 0.0
	pval_array = np.ones(b)/float(b)
	# Randomly sample subset of indices
	idx = np.random.randint(low=0,high=n,size=(s,b))
	for j in range(s):
	    # Approximate the measure
	    monte_carlo_estimate_sum= 0.0
	    multinomial_sample = np.random.multinomial(n,pvals=pval_array,size=r)
	    for k in range(r):
	        monte_carlo_estimate_sum += calculate_statistic(X,y,multinomial_sample[k,]\
	                                             ,idx[j,],p_dict,score_fnc_args)
	    subsample_estimate_sum += monte_carlo_estimate_sum/float(r)

	return subsample_estimate_sum/float(s)