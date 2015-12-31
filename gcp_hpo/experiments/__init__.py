"""
The `gcp_hpo.experiments` modules contains data and scripts that can be used to test GCP-based hyper-parameter optimization.

Each test instance follows the same directory structure, and all files are in the folder experiments/ProblemName :
- config.yml : a yaml file to set the parameters used to run SmartSearch  
- scoring_function/ : the off-line computations stored. params.csv contains the parameters tested, and output.csv the raw outputs given by the scoring function (all the cross-validation estimation). The files *true_score_t_TTT_a_AAA* refer to the Q<sup>1</sup> scores computed with a threshold == TTT and alpha == AAA
- exp_results/expXXX : run_test stores the results in the folder expXXX where XXX is a integer refering to a configuration  
- exp_results/transformed_t_TTT_a_AAA/expXXX : the analyzed results from the trial expXXX, computed by run_result_analysis with a threshold == TTT and alpha == AAA. 
- exp_results/transformed_smooth_t_TTT_a_AAA_kKKK_rRRR_bBBB/expXXX : the analyzed results from the trial expXXX with the *smooth* quality function, computed by run_result_analysis with a threshold == TTT, alpha == AAA, using the nearest KKK neighbors, a radius coefficient RRR and beta == BBB. 
- exp_results/iterations_needed/expXXX_YYY_t_TTT_a_AAA : the mean, median, first and third quartiles of the iterations needed to reach a given score gain, over experiments XXX to YYY. The score is actually the true score computed with a threshold == TTT and alpha == AAA.
"""