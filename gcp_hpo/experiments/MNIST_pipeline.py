"""
Running MNIST pipeline entirely online, rather than from csv files like currently done in the experiments folder
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

def import_data(selected_data,size_test_data):
    """
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

X_train,X_test,y_train,y_test = import_data(1.0,0.33)
    

def gaussian_blur(X,kernel_size,stddev):
    output = np.zeros(X.shape)
    for i in range(X.shape[0]):
        output[i,]= np.reshape(GaussianBlur(X[i,],(kernel_size,kernel_size),sigmaX=stddev,sigmaY=stddev),\
                         (X.shape[1]))
    return output

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


# Old, non-BLB scoring function
# def scoring_function(p_dict):
# 	# parameters: blur_ksize,blur_sigma,pca_dim/10,degree,log10(gamma*1000)

# 	# Train model
# 	blurred_X_train = gaussian_blur(X_train,kernel_size = p_dict['blur_ksize'],\
# 		stddev = p_dict['blur_sigma'])
# 	pca_model,pca_X_train = do_pca(blurred_X_train,num_components=p_dict['pca_dim'],fit = True)
# 	model = random_forest(pca_X_train,y_train,num_estimators=20) # TODO change this to pdict['num_trees']

# 	# Test model
# 	blurred_X_test = gaussian_blur(X_test,kernel_size = p_dict['blur_ksize'],\
# 		stddev = p_dict['blur_sigma'])
# 	pca_X_test = do_pca(blurred_X_test,num_components=p_dict['pca_dim'],fit=False,\
# 		fitted_model=pca_model)
# 	return [model.score(pca_X_test,y_test)]

def blb_main(X,y,calculate_statistic,p_dict,pca_model,forest_model):
	#### Define variables ######
	n = X.shape[0] # Size of data
	b = int(n**(0.6)) # subset size
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

def calculate_score(X,y,sample,indices,p_dict,pca_model,forest_model):
	# TODO use the multinomial sample
	sampled_X = X[indices]
	sampled_y = y[indices]

	blurred_X_sample = gaussian_blur(sampled_X,kernel_size = p_dict['blur_ksize'],\
		stddev = p_dict['blur_sigma'])
	# print 'second shape',X_train.shape
	pca_X_sample = do_pca(blurred_X_sample,num_components=p_dict['pca_dim'],fit=False,\
		fitted_model=pca_model)
	return forest_model.score(pca_X_sample,sampled_y)

# New, BLB scoring function
def scoring_function(p_dict):
	# Train model
	blurred_X_train = gaussian_blur(X_train,kernel_size = p_dict['blur_ksize'],\
		stddev = p_dict['blur_sigma'])
	# print 'first shape',blurred_X_train.shape
	pca_model,pca_X_train = do_pca(blurred_X_train,num_components=p_dict['pca_dim'],fit = True)
	forest_model = random_forest(pca_X_train,y_train,num_estimators=20) # TODO make it num_trees

	return [blb_main(X_test,y_test,calculate_score,p_dict,pca_model,forest_model)]



def main():
	# TODO maybe change the arguments to SmartSearch to be the same as for MNIST in the experiments folder

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
	n_iter = 10
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