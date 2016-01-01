"""
The `gcp_hpo.experiments` modules contains data and scripts that can be used to test GCP-based hyper-parameter optimization. Designed for research purposes, 
this module is relevant when one wants to run several hyper-parameter optimization process with different configurations, in order to compare them for example. That is why this 
module relies on two different components: off-line and on-line computations.  

### Off-line computations
First, train and test a pipeline for many parameters, and store their performances in the folder `test_name/scoring_function`. This can be done, for example, by running `SmartSearch` 
but with randomized search. 

### On-line computations
Simulate as many hyper-parameter optimization processes as you want, eventually with different configurations, with the script `run_experiment`. There, instead of training/testing 
a pipeline, the performances of a parameter will actually be a query in the database built from the off-line computations.

### Analyze the results
Run `iterations_needed` to see how many parameters should be tested to reach a given gain, ie how a SmartSearch configuration performs on this test instance. `analyzed_results` is 
here to convert the raw outputs from `run_experiment` into quality scores, to simulate what we would observe in a real case.

### Files and directory structure  
Each test instance follows the same directory structure, and all files are in the folder `experiments/test_name`:
- `config.yml` : a yaml file to set the parameters used to run `SmartSearch`  
- `scoring_function/` : data from off-line computations. `params.csv` contains the parameters tested, and `output.csv` the raw outputs given by the scoring function (all the cross-validation estimations). The files *true_score_t_TTT_a_AAA* contain the quality scores Q computed with a threshold == TTT and alpha == AAA (see *considering only significant differences* in the paper), using all the data available; this is supposed to represent the ground truth about the performance function.
- `exp_results/expXXX/` : data returned by `runExperiment`, where XXX is the number set in the config file.
- `exp_results/transformed_t_TTT_a_AAA/expXXX/` : analyzed results from the trial expXXX, computed by run_result_analysis with a threshold == TTT and alpha == AAA. 
- `exp_results/transformed_smooth_t_TTT_a_AAA_kKKK_rRRR_bBBB/expXXX/` :  analyzed results from the trial expXXX with the *smooth* quality function, computed by `run_result_analysis` with a threshold == TTT, alpha == AAA, using the nearest KKK neighbors, a radius coefficient RRR and beta == BBB. 
- `exp_results/iterations_needed/expXXX_YYY_t_TTT_a_AAA` : the mean, median, first and third quartiles of the iterations needed to reach a given score gain, over experiments XXX to YYY. The score is actually the true score computed with a threshold == TTT and alpha == AAA.
"""