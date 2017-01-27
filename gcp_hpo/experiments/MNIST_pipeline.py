"""
Running MNIST pipeline entirely online, rather than from csv files like currently done in the experiments folder
"""

from gcp_hpo.smart_search import SmartSearch
import numpy as np
import math

def branin(x, y):
	"""
	The opposite of the Branin function.
	"""
	result = np.square(y - (5.1/(4*np.square(math.pi)))*np.square(x) + \
		(5/math.pi)*x - 6) + 10*(1-(1./(8*math.pi)))*np.cos(x) + 10
	return [-result]


def mnist_pipeline_tmp():
	"""
	Steps:
		1. Import MNIST data (from Kaggle to be consistent with Sebastien)
		2. Blur the training images
		3. Reduce image matrices by principal component analysis
		4. classify using polynomial SVM
	"""

	# Import MNIST data



# def scoring_function(p_dict):
# 	x,y = p_dict['x'], p_dict['y']
# 	x = x -5.
# 	y= y
# 	return branin(x,y)

def import_data():
	train_data = np.loadtxt('/Users/aandersonlaptop/Desktop/MNIST_train.csv',skiprows=1,delimiter=',')
    train_x = train_data[:,1:]
    train_y = train_data[:,0]
    # print 'train Y',train_y
    # print 'train X',train_x
    test_data = np.loadtxt('/Users/aandersonlaptop/Desktop/MNIST_test.csv',skiprows=1,delimiter=',') # way slower than pandas
    test_x = test_data[:,1:]
    test_y = test_data[:,0]
    # print 'test Y',test_y
    # print 'test X',test_x
    return train_x,train_y,test_x,test_y

# TODO set X_train and y_train here (subsets, first 5000, etc)
# X_train =
# y_train =
# X_test =
# y_test = 

def scoring_function(p_dict):
	# parameters: blur_ksize,blur_sigma,pca_dim/10,degree,log10(gamma*1000)

	# Train model
	blurred_X_train = gaussian_blur(X_train,stddev=p_dict['blur_sigma'],k_size=p_dict['blur_ksize'])
	pca_X_train = do_pca(blurred_X_train,num_components=p_dict['pca_dim'])
	model = do_svm(pca_X_train,y_train,degree=p_dict['degree_poly'],gamma_coeff=p_dict['gamma'])

	# Test model
	blurred_X_test = gaussian_blur(X_test,stddev=p_dict['blur_sigma'],k_size=p_dict['blur_ksize'])
	pca_X_test = do_pca(blurred_X_test,num_components=p_dict['pca_dim'])
	return model.score(pca_X_test,y_test)



def main():
	# TODO maybe change the arguments to SmartSearch to be the same as for MNIST in the experiments folder

	### Set parameters ###
	parameters = { 'blur_sigma' : ['int',[0,1]],
	               'blur_ksize' : ['int',[1,1]],
	               'pca_dim' : ['int',[50,300]],
	               'degree_poly' : ['int',[1,4]],
	               'gamma' : ['int',[int(10**(-3)),int(10**(1))]] }
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