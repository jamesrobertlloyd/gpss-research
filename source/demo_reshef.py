"""
Demonstration of stepping through an experiment

@authors: James Robert Lloyd (jrl44@cam.ac.uk)

Created April 2015
"""

import experiment

# Load details about an experiment

exp_params = experiment.load_experiment_details('../experiments/debug/example_pedro_1.py')
print(experiment.exp_params_to_str(exp_params))

# Some of the more important details are...

#...The specification of the starting model - note that these are strings - but they can also be objects

print(exp_params['mean'])
print(exp_params['kernel'])
print(exp_params['lik'])

#...and a parameter that controls a subsetting pruning strategy for speed - setting this value larger than the total number
# of data points runs the algorithm without any pruning
print(exp_params['starting_subset'])

# We now load some data

data = experiment.load_data('../experiments/debug/example_pedro_1.py')[0]
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

# And we can see that we have no improvement