"""
Utilities for the basic_pipeline and MNIST_pipeline.

"""
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def do_pca(X,num_components,fitted_model=None):
	"""
	Executes PCA using sklearn's implementation

	Args:
		X: 					data
		num_components: 	number of components to keep
		fitted_model:		if this is the training set, None. If this is the test set, the pca model with
							which to transform the data

	Returns:
		(if a fitted model is provided) the transformed data
		(if not provided) a tuple of (PCA model, transformed data)
	"""
	if not fitted_model:
	    pca = PCA(n_components = num_components)
	    pca = pca.fit(X)
	    return pca,pca.transform(X)
	else:
	    return fitted_model.transform(X)

def random_forest(X,y,num_estimators):
	"""
	Executes random forest using sklearn's implementation

	Args:
		X:					X data
		y:					y data (labels)
		num_estimators:		number of trees in the forest

	Returns:
		RandomForestClassifier model object
	"""
	clf = RandomForestClassifier(n_estimators=num_estimators)
	clf.fit(X,y)
	# print 'accuracy on training set',clf.score(X,y)
	return clf

def blb(X,y,p_dict,calculate_statistic,score_fnc_args=None):
	"""
	Executes bag of little bootstraps, non-parallelized

	Args:
		X:						X data
		y:						y data (labels)
		p_dict:					dictionary of hyperparameters provided by SmartSearch
		calculate_statistic: 	function that calculates accuracy/F1 score/value in question
		score_fnc_args:    		tuple of arguments to the calculate_statistic function
	"""

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

	