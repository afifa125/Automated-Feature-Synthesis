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

def import_data():
	# TODO replace this with new test_train_split thing probably
		# and put in a random sampling option so can test with smaller data sets

    data = np.loadtxt('/Users/aandersonlaptop/Desktop/MNIST_train.csv',skiprows=1,delimiter=',')
    train_x = data[:40000,1:]
    train_y = data[:40000,0]
    # print 'train Y',train_y
    # print 'train X',train_x
    test_x = data[40000:,1:]
    test_y = data[40000:,0]
    
    # print 'test X',test_x
    return train_x,train_y,test_x,test_y
    
    
X_train,y_train,X_test,y_test = import_data()

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


def scoring_function(p_dict):
	# parameters: blur_ksize,blur_sigma,pca_dim/10,degree,log10(gamma*1000)

	# Train model
	blurred_X_train = gaussian_blur(X_train,kernel_size = p_dict['blur_ksize'],\
		stddev = p_dict['blur_sigma'])
	pca_model,pca_X_train = do_pca(blurred_X_train,num_components=p_dict['pca_dim'],fit = True)
	model = random_forest(pca_X_train,y_train,num_estimators=20)

	# Test model
	blurred_X_test = gaussian_blur(X_test,kernel_size = p_dict['blur_ksize'],\
		stddev = p_dict['blur_sigma'])
	pca_X_test = do_pca(blurred_X_test,num_components=p_dict['pca_dim'],fit=False,\
		fitted_model=pca_model)
	return [model.score(pca_X_test,y_test)]



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