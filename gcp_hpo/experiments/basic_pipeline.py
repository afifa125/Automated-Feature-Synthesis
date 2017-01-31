"""
Running a basic pipeline for example entirely online, rather than from the csv files as in the experiments folder

The iris dataset from sklearn is used for this pipeline.

Pipeline steps:
    1. import data
    2. PCA
    3. Random Forest classification
    
Hyperparameters to be optimized:

    Parameter             Description                        Range
    pca_dim               dimension of PCA                   [0,4]
    number_estimators     number of trees in random forest   [5,30]
"""

"""
INSTRUCTIONS FOR IMPLEMENTING SMARTSEARCH ON YOUR PIPELINE

Essentially, just follow the format of this file using the steps below.

	1. Specify a parameter range
		- In the main() function, simply put in the dictionary "parameters" the names, ranges, and types
		of the hyperparameters you would like to optimize over.
	2. Define a scoring function
		- The return value is how you would like alternative models to be evaluated (cross validation is also used)
		- Be sure to import your data outside of this function so that you don't repeat unnecessary steps
		- Define helper functions outside of this function as needed
		- All parameters on which you are optimizing must be defined in p_dict
"""

from gcp_hpo.smart_search import SmartSearch
import math
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


def load_data():
	# Import data
    data = load_iris()
    X = data['data']
    y = data['target']
    return train_test_split(X,y,test_size=0.33) 

X_train,X_test,y_train,y_test = load_data()

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
    print 'accuracy on training set',clf.score(X,y)
    return clf

def blb_main(X,y,calculate_statistic,p_dict,pca_model,forest_model):
	#### Define variables ######
	n = X_test.shape[0] # Size of data
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
	                                             ,idx[j,],p_dict,pca_model,forest_model)
	    subsample_estimate_sum += monte_carlo_estimate_sum/float(r)

	return subsample_estimate_sum/float(s)


def calculate_accuracy(X,y,sample,indices,p_dict,pca_model,model):
	sampled_X = X[indices]
	sampled_Y = y[indices]

	# Test model
	pca_X_test = do_pca(sampled_X,num_components=p_dict['pca_dim'],fit=False,fitted_model=pca_model)
	return model.score(pca_X_test,sampled_Y)


def scoring_function(p_dict):
	# TODO only using train now

	# Train model
	pca_model,pca_X_train = do_pca(X_train,num_components=p_dict['pca_dim'],fit = True)
	forest_model = random_forest(pca_X_train,y_train,num_estimators=p_dict['number_estimators'])

	return [blb_main(X_test,y_test,calculate_accuracy,p_dict,pca_model,forest_model)]

# def scoring_function(p_dict):
# 	# NOTE: old version. this does not use BLB

# 	# parameters: pca_dim,number_estimators

# 	# Train model
# 	pca_model,pca_X_train = do_pca(X_train,num_components=p_dict['pca_dim'],fit = True)
# 	model = random_forest(pca_X_train,y_train,num_estimators=p_dict['number_estimators'])

# 	# Test model
# 	pca_X_test = do_pca(X_test,num_components=p_dict['pca_dim'],fit=False,fitted_model=pca_model)
# 	return [model.score(pca_X_test,y_test)]



def main():

	### Set parameters ###
	parameters = { 'pca_dim' : ['int',[1,4]],
	               'number_estimators' : ['int',[5,30]] }

	nugget = 1.e-10
	n_clusters = 1
	cluster_evol ='constant'
	corr_kernel = 'exponential_periodic'
	mapWithNoise= False
	model_noise = None
	sampling_model = 'GCP'
	n_candidates= 100
	n_random_init= 10
	n_iter = 20
	nb_iter_final = 0
	acquisition_function = 'UCB'

	search = SmartSearch(parameters,
				estimator=scoring_function,
				corr_kernel = corr_kernel,
				acquisition_function = acquisition_function,
				GCP_mapWithNoise=mapWithNoise,
				model_noise = model_noise,
				model = sampling_model, 
				n_candidates=n_candidates,
				n_iter = n_iter,
				n_init = n_random_init,
				n_final_iter=nb_iter_final,
				n_clusters=n_clusters, 
				cluster_evol = cluster_evol,
				verbose=2,
				detailed_res = 2)

	search._fit()


if __name__ == '__main__':
	main()