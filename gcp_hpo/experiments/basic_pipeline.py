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
INSTRUCTIONS FOR IMPLEMENTING SMARTSEARCH ON NEW PIPELINES

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
from pipeline_utils import do_pca,random_forest,blb


def load_data():
	"""
	Import sklearn's iris toy dataset and splits into training and test sets
	"""
	data = load_iris()
	X = data['data']
	y = data['target']
	return train_test_split(X,y,test_size=0.33) 

X_train,X_test,y_train,y_test = load_data()


def calculate_accuracy(X,y,sample,indices,p_dict,score_fnc_args):
	"""
	Returns the mean accuracy on the given dataset and labels

	Args:
		X:						X data
		y:						y data (labels)
		p_dict:					dictionary of hyperparameters provided by SmartSearch
		calculate_statistic: 	function that calculates accuracy/F1 score/value in question
		score_fnc_args:    		tuple of arguments to the calculate_statistic function
	"""
	pca_model, forest_model = score_fnc_args

	sampled_X = X[indices]
	sampled_Y = y[indices]

	# Test model
	pca_X_test = do_pca(sampled_X,num_components=p_dict['pca_dim'],fitted_model=pca_model)
	return forest_model.score(pca_X_test,sampled_Y,sample_weight=sample)


def scoring_function(p_dict):
	"""
	Executes the basic pipeline according to the parameters in p_dict. Returns score
	"""
	# Train model
	pca_model,pca_X_train = do_pca(X_train,num_components=p_dict['pca_dim'])
	forest_model = random_forest(pca_X_train,y_train,num_estimators=p_dict['number_estimators'])

	# Test model and output score
	return [blb(X_test,y_test,p_dict,calculate_accuracy,(pca_model,forest_model))]

# Old scoring_function, does not use BLB
# def scoring_function(p_dict):

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