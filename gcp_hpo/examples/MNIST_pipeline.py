"""
Running MNIST pipeline entirely online, rather than from csv files like currently done in the experiments folder

NOTE: Currently uses random forest classifier for speed considerations, but can be adapted to SVM
"""

from gcp_hpo.smart_search import SmartSearch
import numpy as np
import math
from scipy.ndimage.filters import gaussian_filter 
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from cv2 import GaussianBlur
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pipeline_utils import do_pca,random_forest,blb

def import_data(selected_data,size_test_data):
    """
    Import Kaggle MNIST dataset and return train test split

    Args:
        selected_data:   percentage of original data to use for training and testing combined
        size_test_data:  percentage of selected set to use as test data
    """
    data = np.loadtxt('/Users/aandersonlaptop/Desktop/MNIST_train.csv',skiprows=1,delimiter=',')

    X = data[:,1:]
    y = data[:,0]
    
    # Select subset of data
    n = X.shape[0] # number of rows
    num_samples = int(selected_data*n)
    idx = np.random.randint(n,size=num_samples)
    X,y = X[idx],y[idx]

    return train_test_split(X,y,test_size=size_test_data)

X_train,X_test,y_train,y_test = import_data(0.01,0.33)
    

def gaussian_blur(X,kernel_size,stddev):
	"""
	Executes Gaussian blur on the given data

	Args:
		X: 		 		data to blur
		kernel_size:	Gaussian kernel size
		stddev:			Gaussian kernel standard deviation (in both X and Y directions)
	"""
	output = np.zeros(X.shape)
	for i in range(X.shape[0]):
	    output[i,]= np.reshape(GaussianBlur(X[i,],(kernel_size,kernel_size),sigmaX=stddev,\
	    	sigmaY=stddev),(X.shape[1]))
	return output

# Old, non-BLB scoring function
# def scoring_function(p_dict):
# 	# parameters: blur_ksize,blur_sigma,pca_dim/10,degree,log10(gamma*1000)

# 	# Train model
# 	blurred_X_train = gaussian_blur(X_train,kernel_size = p_dict['blur_ksize'],\
# 		stddev = p_dict['blur_sigma'])
# 	pca_model,pca_X_train = do_pca(blurred_X_train,num_components=p_dict['pca_dim'],fit = True)
# 	model = random_forest(pca_X_train,y_train,num_estimators=pdict['num_trees'])

# 	# Test model
# 	blurred_X_test = gaussian_blur(X_test,kernel_size = p_dict['blur_ksize'],\
# 		stddev = p_dict['blur_sigma'])
# 	pca_X_test = do_pca(blurred_X_test,num_components=p_dict['pca_dim'],fit=False,\
# 		fitted_model=pca_model)
# 	return [model.score(pca_X_test,y_test)]

def calculate_score(X,y,sample,indices,p_dict,score_fnc_args):
	"""
	Returns the mean accuracy on the given dataset and labels

	Args:
		X:						X data
		y:						y data (labels)
		p_dict:					dictionary of hyperparameters provided by SmartSearch
		calculate_statistic: 	function that calculates accuracy/F1 score/value in question
		score_fnc_args:    		tuple of arguments to the calculate_statistic function
	"""
	# NOTE: The multinomial sample weights are currently not included for PCA
	pca_model,forest_model = score_fnc_args

	sampled_X = X[indices]
	sampled_y = y[indices]

	blurred_X_sample = gaussian_blur(sampled_X,kernel_size = p_dict['blur_ksize'],\
		stddev = p_dict['blur_sigma'])
	pca_X_sample = do_pca(blurred_X_sample,num_components=p_dict['pca_dim'],\
		fitted_model=pca_model)
	return forest_model.score(pca_X_sample,sampled_y,sample_weight=sample)

# New, BLB scoring function
def scoring_function(p_dict):
	"""
	Executes the MNIST pipeline according to the parameters in p_dict. Returns score
	"""
	
	# Train model
	blurred_X_train = gaussian_blur(X_train,kernel_size = p_dict['blur_ksize'],\
		stddev = p_dict['blur_sigma'])
	pca_model,pca_X_train = do_pca(blurred_X_train,num_components=p_dict['pca_dim'])
	forest_model = random_forest(pca_X_train,y_train,num_estimators=p_dict['num_trees'])

	return [blb(X_test,y_test,p_dict,calculate_score,(pca_model,forest_model))]



def main():
	### Set parameters ###
	parameters = { 'blur_ksize' : ['cat',[3,5]],
				   'blur_sigma' : ['int',[0,1]],
	               'pca_dim' : ['int',[50,300]],
	               'num_trees': ['int',[5,30]]}
	               # 'degree_poly' : ['int',[1,4]],
	               # 'gamma' : ['int',[int(10**(-3)),int(10**(1))]] }
	nugget = 1.e-10
	n_clusters = 1
	cluster_evol ='constant'
	corr_kernel = 'exponential_periodic'
	mapWithNoise= False
	model_noise = None
	sampling_model = 'GCP'
	n_candidates= 100
	n_random_init= 2
	n_iter = 2
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