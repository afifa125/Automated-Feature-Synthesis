"""
Transform the results of a single experiment to computes quality scores for the parameters 
tested during the search. The quality scores are defined in the paper and aim at giving 
a more accurate feedback than a list of cross-validation outputs. First, it should consider 
only significant differences between the means. Second, one can smooth the results by 
averaging on a parameter's neighborhood.  
Results will be saved in the folder `test_name/exp_results/transformed_t_TTT_a_AAA/expX/` 
depending on the threshold and alpha values, or in the folder 
`test_name/exp_results/transformed_smooth_t_TTT_a_AAA_kKKK_rRRR_bBBB/expX/` if `smoothQ` 
is set to True.
"""

# Author: Sebastien Dubois 
#         for ALFA Group, CSAIL, MIT

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

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import NearestNeighbors


def transformResults(test_name,
                     model_dir, n_exp,
                     threshold,alpha,
                     smoothQ = False,
                     kneighbors_s = 3,
                     sigma_s = 200.,
                     beta = 5.,
                     verbose= False):
    """
    Parameters
    ----------
    `test_name` : name of the test instance, should correspond to a 
        folder name in test/

    `model_dir`: a string corresponding to the model and subsize to use,
        written as a path. Ex: 'GCP/5000'

    `n_exp` : the index number of the experiment to transform

    `threshold` : threshold to use to decide whether the difference between 
        observations' means is significant, based on Welch's t-test

    `alpha` : trade-off parameter to compute the score from the significant
        mean and the standard deviation. score == m - alpha * std

    `smoothQ` : boolean, optional. If True, compute the smooth quality function.
        See the Deep Mining paper for details.

    `kneighbors_s` : if smoothQ == True, number of nearest neighbors to take into 
        account for the smoothing

    `sigma_s` : if smoothQ == True, radius used to compute the coefficients based 
        on a neighbor's return_distance for the smoothing

    `beta` : if smoothQ == True, parameter to balance between the smoothed mean and
        and the original one. m = (smoothed_mean + beta * original_mean) / (1+beta)

    Returns
    -------
    `scores` : quality scores of the parameters tested during the search, in the same
        order as in output file of the experiment
    """

    
    if not (smoothQ):
        folder = test_name + "/exp_results/" + model_dir + \
                 "/transformed_t_" + str(threshold) + "_a_" + str(alpha) 
    else:
        folder = test_name + "/exp_results/" + model_dir + \
                 "/transformed_smooth_t_" + str(threshold) + "_a_" + str(alpha) + \
                 "_k" + str(kneighbors_s) + "_r" + str( int(sigma_s)) + "_b" + str(beta)

    if not os.path.exists(folder):
        os.mkdir(folder)

    if os.path.exists(folder + "/exp" + str(n_exp) + ".csv"):
        score = np.genfromtxt(folder + "/exp" + str(n_exp) +".csv", delimiter = ',')
        return score

    mean_outputs = []
    std_outputs = []
    raw_outputs = []

    p_dir = test_name + "/exp_results/" + model_dir + "/exp"+str(n_exp)+"/param.csv"
    f =open(test_name + "/exp_results/" + model_dir + "/exp"+str(n_exp)+"/output.csv",'r')

    for l in f:
        l = l[1:-2]
        string_l = l.split(',')
        raw_outputs.append([float(i) for i in string_l] )
    f.close()

    c_print = 0
    for o in raw_outputs:
        mean_outputs.append(np.mean(o))
        if(verbose):
            print c_print,np.mean(o)
            c_print += 1
        std_outputs.append(np.std(o))


    parameters = np.genfromtxt(p_dir, delimiter = ',')

    if(smoothQ):
        smooth_raw_output = []
        smooth_output = []
        smoothingKNN = NearestNeighbors()
        smoothingKNN.fit(parameters)
        for i in range(parameters.shape[0]):
            neighbor_dist, neighbor_idx = smoothingKNN.kneighbors(parameters[i,:], kneighbors_s, return_distance = True)
            coefs = np.exp(- (neighbor_dist / sigma_s) **2.)[0]
            neighbor_idx = neighbor_idx[0]
            smooth_o = (np.mean([ coefs[j]* mean_outputs[neighbor_idx[j]] for j in range(kneighbors_s)]) + beta*mean_outputs[i]) \
                        /(1. + beta)
            ro = raw_outputs[i]
            diff = smooth_o - mean_outputs[i]
            smooth_raw_output.append([ (diff + o) for o in ro ])
            smooth_output.append(smooth_o)
            
        mean_outputs = smooth_output
        raw_outputs = smooth_raw_output

        if(verbose):
            c_print = 0
            for o in mean_outputs:
                print c_print, o
                c_print += 1

    sorted_idx = np.argsort(mean_outputs)
    sorted_raw_outputs= [ raw_outputs[i] for i in sorted_idx ]

    # Cluster with Welch's t-test
    clustered_obs = []
    clusters_param_idx = []
    c_count = -1
    for i in range( len( sorted_raw_outputs) ):
        o = sorted_raw_outputs[i]
        if(c_count > -1 ):
            t,p = stats.ttest_ind(clustered_obs[c_count],o,equal_var=False)
            if(p < threshold):
                clustered_obs.append(np.copy(o))
                clusters_param_idx.append( [sorted_idx[i]] )
                c_count += 1
            else:
                clustered_obs[c_count] = np.concatenate((clustered_obs[c_count], np.copy(o)))
                clusters_param_idx[c_count] += [sorted_idx[i]]
        else:
            clustered_obs.append(np.copy(o))
            clusters_param_idx.append([ sorted_idx[i] ] )
            c_count = 0
        
    if(verbose):        
        print '\n', len(clustered_obs),' clusters'
        for i in range(len(clustered_obs)):
            c_o = clustered_obs[i]
            print 'Size',len(c_o),'\t','Mean',np.mean(c_o)
            print 'Params idx', clusters_param_idx[i]
        print '\n'


    # compute score
    params_idx = []
    params_results = []
    for i in range(1,len(clustered_obs)+1):
        mean_on_cluster = np.mean(clustered_obs[-i])
        for idx in clusters_param_idx[-i]:
            params_results.append( [ mean_on_cluster ,np.std(raw_outputs[idx]) ] )
        params_idx += clusters_param_idx[-i]
    params_results = np.asarray(params_results)

    final_score = params_results[:,0] - alpha*params_results[:,1]
                
    final_sorted_idx = np.argsort(final_score)[::-1]
    #print final_score
    rank_idx = np.asarray(params_idx)[final_sorted_idx]
    if(verbose):
        print 'Sorted params idx:\n',rank_idx 
        best_idx = params_idx[final_sorted_idx[0]]
        print '\nBest parameter set', parameters[best_idx,:]
        print 'Next:'
        for i in range(1,5):
            print parameters[params_idx[final_sorted_idx[i]]]

    parameter_score = []
    idx_matching = np.argsort(params_idx)
    #path_params = np.genfromtxt(path_dir,delimiter=',')
    for i in range(parameters.shape[0]):
        parameter_score.append(final_score[idx_matching[i]])

    np.savetxt(folder+"/exp"+str(n_exp)+".csv",parameter_score,delimiter=',')

    return np.asarray(parameter_score)


if __name__ == '__main__':
    n_exp = 1
    test_name = "SentimentAnalysis"
    model_dir = "rand/5000"

    threshold = 0.5
    alpha = 0.5

    smoothQ = False
    kneighbors_s = 3
    sigma_s = 200.
    beta = 5.

    verbose = True

    scores = transformResults(test_name,
                              model_dir, n_exp,
                              threshold, alpha,
                              smoothQ = smoothQ,
                              kneighbors_s = kneighbors_s,
                              sigma_s = sigma_s,
                              beta = beta,
                              verbose=verbose)

    fig = plt.figure(figsize=(15,7))
    plt.plot(range(scores.shape[0]),scores)
    plt.show() 