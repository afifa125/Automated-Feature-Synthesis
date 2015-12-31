# Author: Sebastien Dubois 
#     for ALFA Group, CSAIL, MIT

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

import sys
from gcp_hpo.smart_search import SmartSearch
import numpy as np
import os
import yaml
from sklearn.neighbors import NearestNeighbors

def runExperiment(first_exp,
                  n_exp,
                  dir_,
                  parameters,
                  model = 'GCP',
                  n_random_init = 10,
                  n_total_iter = 30,
                  n_candidates=500,
                  corr_kernel='squared_exponential',
                  acquisition_function = 'UCB',
                  n_clusters = 1,
                  cluster_evol = 'constant',
                  GCP_mapWithNoise=False,
                  GCP_useAllNoisyY=False,
                  model_noise=None):
  
  last_exp = first_exp + n_exp
  print 'Run experiment',first_exp,'to',last_exp

  # Load data
  output = []
  f =open(( dir_ + 'scoring_function/output.csv'),'r')
  for l in f:
      l = l[1:-3]
      string_l = l.split(',')
      output.append( [ float(i) for i in string_l] )
  f.close()
  print 'Loaded output file,',len(output),'rows'

  params = np.genfromtxt((dir_ + 'scoring_function/params.csv'),delimiter=',')
  print 'Loaded parameters file, shape :',params.shape

  KNN = NearestNeighbors()
  KNN.fit(params)
  # KNN.kneighbors(p,1,return_distance=False)[0]

  # function that retrieves a performance evaluation from the stored results
  def get_cv_res(p_dict):
      p = np.zeros(len(parameters))
      for k in p_dict.keys():
        p[int(k)] = p_dict[k]
      idx = KNN.kneighbors(p,1,return_distance=False)[0]
      all_o = output[idx]
      r = np.random.randint(len(all_o)/5)
      return all_o[(5*r):(5*r+5)]


  ###  Run experiment  ### 
  if not os.path.exists(dir_ + 'exp_results'):
    os.mkdir(dir_ + 'exp_results')

  for n_exp in range(first_exp,last_exp):
      print ' ****   Run exp',n_exp,'  ****'
      ### set directory
      if not os.path.exists(dir_ + 'exp_results/exp'+str(n_exp)):
          os.mkdir(dir_ + 'exp_results/exp'+str(n_exp))
      else:
          print('Warning : directory already exists')

      search = SmartSearch(parameters,
                        estimator = get_cv_res,
                        corr_kernel = corr_kernel ,
                        GCP_mapWithNoise=GCP_mapWithNoise,
                        GCP_useAllNoisyY=GCP_useAllNoisyY,
                        model_noise = model_noise,
                        model = model, 
                        n_candidates = n_candidates,
                        n_iter = n_total_iter,
                        n_init = n_random_init, 
                        n_clusters = n_clusters,
                        cluster_evol = cluster_evol,
                        verbose = 2,
                        acquisition_function = acquisition_function,
                        detailed_res = 2)

      all_parameters, all_search_path, all_raw_outputs,all_mean_outputs = search._fit()

      ## save experiment's data
      for i in range(len(all_raw_outputs)):
          f =open((dir_ + 'exp_results/exp'+str(n_exp)+'/output_'+str(i)+'.csv'),'w')
          for line in all_raw_outputs[i]:
              print>>f,line
          f.close()
          np.savetxt((dir_ + 'exp_results/exp'+str(n_exp)+'/param_'+str(i)+'.csv'),all_parameters[i], delimiter=',')
          np.savetxt((dir_ + 'exp_results/exp'+str(n_exp)+'/param_path_'+str(i)+'.csv'),all_search_path[i], delimiter=',')

      print ' ****   End experiment',n_exp,'  ****\n'


if __name__ == '__main__':
  dir_ = sys.argv[1] + '/'
  print 'Saving in subdirectory:', dir_

  config = yaml.safe_load(open(dir_ + 'config.yml'))

  runExperiment(first_exp = config['first_exp'],
              n_exp = config['n_exp'],
              dir_ = dir_,
              model = config['model'],
              parameters = config['parameters'],
              n_random_init = config['n_random_init'],
              n_total_iter = config['n_iter'],
              corr_kernel = config['corr_kernel'],
              acquisition_function = config['acquisition_function'],
              n_clusters = config['n_clusters'],
              cluster_evol = config['cluster_evol'],
              GCP_mapWithNoise = config['GCP_mapWithNoise'],
              GCP_useAllNoisyY = config['GCP_useAllNoisyY'],
              model_noise = config['model_noise'],
              n_candidates = config['n_candidates'])