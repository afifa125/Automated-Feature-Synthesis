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
#     print 'data',data
    X = data['data']
    y = data['target']
    return train_test_split(X,y,test_size=0.33) 

X_train,X_test,y_train,y_test = load_data()
# print 'X_train',X_train.shape
# print 'y_train',y_train.shape
# print 'X_test',X_test.shape
# print 'y_test',y_test.shape

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


def scoring_function(p_dict):
	# parameters: pca_dim,number_estimators

	# Train model
	pca_model,pca_X_train = do_pca(X_train,num_components=p_dict['pca_dim'],fit = True)
	model = random_forest(pca_X_train,y_train,num_estimators=p_dict['number_estimators'])

	# Test model
	pca_X_test = do_pca(X_test,num_components=p_dict['pca_dim'],fit=False,fitted_model=pca_model)
	return [model.score(pca_X_test,y_test)]



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