# Author: Sebastien Dubois 
#		  for ALFA Group, CSAIL, MIT

# The MIT License (MIT)
# Copyright (c) 2015 Sebastien Dubois

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
import sys

sys.path.append("../../../..")
from DeepMining.gcp_hpo.gcp import GaussianCopulaProcess
from sklearn.gaussian_process import GaussianProcess

### Set parameters ###
t_size = [20,50,80]
nugget = 1.e-10
n_clusters_max = 4
integratedPrediction = False
n_tests = 20
log_likelihood = False
print 'Average on n_tests = 20, log_likelihood = False, nugget = 1.e-10, integratedPrediction = False'

def artificial_f(x):
	x = x[0]
	res = (70-7*np.exp(x/50. - ((x-55.)**2)/500.) + 6*np.sin(x/40.) +3./(1.1+np.cos(x/50.)) - 15./(3.3-3*np.sin((x-70)/25.)))/100.
	return [res]

def branin_f(p_vector):
	x,y = p_vector
	x = x -5.
	y= y
	result = np.square(y - (5.1/(4*np.square(math.pi)))*np.square(x) + \
		(5/math.pi)*x - 6) + 10*(1-(1./(8*math.pi)))*np.cos(x) + 10
	return [-result]

def har6(x):
    """6d Hartmann test function
    constraints:
    0 <= xi <= 1, i = 1..6
    global optimum at (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),
    where har6 = 3.32236"""


    a = np.array([[10.0,   3.0, 17.0,   3.5,  1.7,  8.0],
                [ 0.05, 10.0, 17.0,   0.1,  8.0, 14.0],
                [ 3.0,   3.5,  1.7,  10.0, 17.0,  8.0],
                [17.0,   8.0,  0.05, 10.0,  0.1, 14.0]])
    c = np.array([1.0, 1.2, 3.0, 3.2])
    p = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
    s = 0
    for i in [0,1,2,3]:
        sm = a[i,0]*(x[0]-p[i,0])**2
        sm += a[i,1]*(x[1]-p[i,1])**2
        sm += a[i,2]*(x[2]-p[i,2])**2
        sm += a[i,3]*(x[3]-p[i,3])**2
        sm += a[i,4]*(x[4]-p[i,4])**2
        sm += a[i,5]*(x[5]-p[i,5])**2
        s += c[i]*np.exp(-sm)
    
    return [s]

all_parameter_bounds = {'artificial_f': np.asarray( [[0,400]] ),
						'branin': np.asarray( [[0,15],[0,15]] ),
						'har6' : np.asarray( [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]) }

tests = {'function' : ['artificial_f','branin','har6'],
		'corr_kernel': ['squared_exponential','exponential_periodic'] }

functions = {'artificial_f':artificial_f,'branin':branin_f,'har6':har6}


def run_test(training_size,scoring_function,parameter_bounds,corr_kernel,n_cluster,prior='GCP',log=True):

	x_training = []
	y_training = []
	for i in range(training_size):
		x = [np.random.uniform(parameter_bounds[j][0],parameter_bounds[j][1]) for j in range(parameter_bounds.shape[0])]
		x_training.append(x)
		y_training.append(scoring_function(x)[0])

	if(prior == 'GP'):
		gp = GaussianProcess(theta0=.1 *np.ones(parameter_bounds.shape[0]),
						 thetaL = 0.001 * np.ones(parameter_bounds.shape[0]),
						 thetaU = 10. * np.ones(parameter_bounds.shape[0]),
						 random_start = 5,
						 nugget=nugget)
		gp.fit(x_training,y_training)
		likelihood = gp.reduced_likelihood_function_value_
	else:
		gcp = GaussianCopulaProcess(nugget = nugget,
		                            corr=corr_kernel,
		                            random_start=5,
		                            normalize = True,
		                            coef_latent_mapping = 0.4,
		                            n_clusters=n_clusters)
		gcp.fit(x_training,y_training)
		likelihood = gcp.reduced_likelihood_function_value_

	if not log:
		likelihood = np.exp(likelihood)

	return likelihood


for s in t_size:
	print 'Training size :',s
	for f in tests['function']:
		print ' **  Test function',f,' ** '
		scoring_function = functions[f]
		parameter_bounds = all_parameter_bounds[f]
		likelihood = [run_test(s,scoring_function,parameter_bounds,'',0,prior='GP',log=log_likelihood) for j in range(n_tests)]
		print '\t\t\t\t > GP - Likelihood =',np.mean(likelihood),'\t',np.std(likelihood),'\n'
		for k in tests['corr_kernel']:
			for n_clusters in range(1,n_clusters_max):
				likelihood = [run_test(s,scoring_function,parameter_bounds,k,n_clusters,log=log_likelihood) for j in range(n_tests)]
				print 'corr_kernel:',k,'- n_clusters:',n_clusters,'\tLikelihood =',np.mean(likelihood),'\t',np.std(likelihood)
			print ''
		print ''
	print ''


