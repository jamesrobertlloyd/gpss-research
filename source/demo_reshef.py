"""
Demonstration of stepping through an experiment

@authors: James Robert Lloyd (jrl44@cam.ac.uk)

Created April 2015
"""

import numpy as np

import experiment

# Load details about an experiment

exp_params = experiment.load_experiment_details('../experiments/debug/example_reshef_1.py')
print(experiment.exp_params_to_str(exp_params))

# Some of the more important details are...

#...The specification of the starting model - note that these are strings - but they can also be objects

print(exp_params['mean'])
print(exp_params['kernel'])
print(exp_params['lik'])

#...and a parameter that controls the subset and iteration schedule
print(exp_params['iter_and_subset_schedule'])

# We now load some data

data = experiment.load_data('../experiments/debug/example_reshef_1.py')[0]
X = data['X']
Y = data['Y']

print(X.shape)
print(Y.shape)

# We can now run the kernel search for one step

results = experiment.kernel_search_single_step(X, Y, exp_params)
best_model = results['all_models'][-1]
print('')
print('Best BIC = %f' % best_model.bic)
print('Best model = %s' % best_model.pretty_print())

# Compute residuals and proportion variance explained

Y_hat = best_model.gpy_predict(X, Y, X)['mean']
resid_hat = Y - Y_hat
resid_var = np.var(resid_hat)
total_var = np.var(Y)
var_explained = 1 - resid_var / total_var
print('Variance explained = %02.1f%%' % (var_explained * 100))

# We can update the experiment parameters with this latest result

exp_params['mean'] = best_model.mean
exp_params['kernel'] = best_model.kernel
exp_params['lik'] = best_model.likelihood

# And run another step of the search

results = experiment.kernel_search_single_step(X, Y, exp_params)
best_model = results['all_models'][-1]
print('')
print('Best BIC = %f' % best_model.bic)
print('Best model = %s' % best_model.pretty_print())

# Compute residuals and proportion variance explained

Y_hat = best_model.gpy_predict(X, Y, X)['mean']
resid_hat = Y - Y_hat
resid_var = np.var(resid_hat)
total_var = np.var(Y)
var_explained = 1 - resid_var / total_var
print('Variance explained = %02.1f%%' % (var_explained * 100))

# We can update the experiment parameters with this latest result again

exp_params['mean'] = best_model.mean
exp_params['kernel'] = best_model.kernel
exp_params['lik'] = best_model.likelihood

# And run another step of the search again

results = experiment.kernel_search_single_step(X, Y, exp_params)
best_model = results['all_models'][-1]
print('')
print('Best BIC = %f' % best_model.bic)
print('Best model = %s' % best_model.pretty_print())

# Compute residuals and proportion variance explained

Y_hat = best_model.gpy_predict(X, Y, X)['mean']
resid_hat = Y - Y_hat
resid_var = np.var(resid_hat)
total_var = np.var(Y)
var_explained = 1 - resid_var / total_var
print('Variance explained = %02.1f%%' % (var_explained * 100))

# And we can see that we have no improvement