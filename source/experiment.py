"""
Main file for setting up experiments, and compiling results.

@authors: David Duvenaud (dkd23@cam.ac.uk)
          James Robert Lloyd (jrl44@cam.ac.uk)
          Roger Grosse (rgrosse@mit.edu)
          
Created Jan 2013          
"""

import numpy as np
import os
import random
import re
import warnings

import flexible_function as ff
from flexible_function import GPModel
import grammar
import utils.latex
import utils.misc

from multiprocessing import Pool
# from multiprocessing import Pipe Process


def optimise_single_model(params):
    model, X, Y, kwargs = params
    return model.gpy_optimize(X=X, Y=Y, **kwargs)


# # A workaround until a memory leak is fixed
# def process_target(pipe_to_parent, model, X, Y, **kwargs):
#     optimised_model = model.gpy_optimize(X=X, Y=Y, **kwargs)
#     pipe_to_parent.send(optimised_model)
#     pipe_to_parent.close()
#
#
# # A workaround until a memory leak is fixed
# def optimize_separate_process(model, X, Y, **kwargs):
#     """A brutal workaround until a memory leak is fixed"""
#     parent_conn, child_conn = Pipe()
#     p = Process(target=process_target, args=(child_conn, model, X, Y), kwargs=kwargs)
#     p.start()
#     optimised_model = parent_conn.recv()
#     p.join()
#     return optimised_model


def perform_kernel_search(X, Y, experiment_data_file_name, results_filename, exp):
    """Search for the best kernel"""
    
    # Initialise random seeds - randomness may be used in e.g. data subsetting

    utils.misc.set_all_random_seeds(exp['random_seed'])
    
    # Create location, scale and minimum period parameters to pass around for parameter initialisations

    data_shape = dict()
    data_shape['x_mean'] = [np.mean(X[:, dim]) for dim in range(X.shape[1])]
    data_shape['y_mean'] = np.mean(Y) # TODO - need to rethink this for non real valued data
    data_shape['x_sd'] = np.log([np.std(X[:, dim]) for dim in range(X.shape[1])])
    data_shape['y_sd'] = np.log(np.std(Y)) # TODO - need to rethink this for non real valued data
    data_shape['y_min'] = np.min(Y)
    data_shape['y_max'] = np.max(Y)
    data_shape['x_min'] = [np.min(X[:, dim]) for dim in range(X.shape[1])]
    data_shape['x_max'] = [np.max(X[:, dim]) for dim in range(X.shape[1])]

    # Initialise period at a multiple of the shortest / average distance between points, to prevent Nyquist problems.
    # This is ultimately a little hacky and is avoiding more fundamental decisions

    if exp['period_heuristic_type'] == 'none':
        data_shape['min_period'] = None
    if exp['period_heuristic_type'] == 'min':
        data_shape['min_period'] = np.log([exp['period_heuristic'] * utils.misc.min_abs_diff(X[:, i])
                                           for i in range(X.shape[1])])
    elif exp['period_heuristic_type'] == 'average':
        data_shape['min_period'] = np.log([exp['period_heuristic'] * np.ptp(X[:, i]) / X.shape[0]
                                           for i in range(X.shape[1])])
    elif exp['period_heuristic_type'] == 'both':
        data_shape['min_period'] = np.log([max(exp['period_heuristic'] * utils.misc.min_abs_diff(X[:, i]),
                                               exp['period_heuristic'] * np.ptp(X[:, i]) / X.shape[0])
                                           for i in range(X.shape[1])])
    else:
        warnings.warn('Unrecognised period heuristic type : using most conservative heuristic')
        data_shape['min_period'] = np.log([max(exp['period_heuristic'] * utils.misc.min_abs_diff(X[:, i]),
                                               exp['period_heuristic'] * np.ptp(X[:, i]) / X.shape[0])
                                           for i in range(X.shape[1])])

    data_shape['max_period'] = [np.log((1.0 / exp['max_period_heuristic']) *
                                       (data_shape['x_max'][i] - data_shape['x_min'][i]))
                                for i in range(X.shape[1])]

    # Initialise mean, kernel and likelihood

    m = eval(exp['mean'])
    k = eval(exp['kernel'])
    l = eval(exp['lik'])
    current_models = [ff.GPModel(mean=m, kernel=k, likelihood=l, ndata=Y.size)]

    print('\n\nStarting search with this model:\n')
    print(current_models[0].pretty_print())
    print('')

    # Perform the initial expansion

    current_models = grammar.expand_models(D=X.shape[1],
                                           models=current_models,
                                           base_kernels=exp['base_kernels'],
                                           rules=exp['search_operators'])

    # Convert to additive form if desired

    if exp['additive_form']:
        current_models = [model.additive_form() for model in current_models]
        current_models = ff.remove_duplicates(current_models)   

    # Setup lists etc to record search and current state
    
    all_results = [] # List of scored kernels
    results_sequence = [] # List of lists of results, indexed by level of expansion.
    nan_sequence = [] # List of list of nan scored results
    oob_sequence = [] # List of list of out of bounds results
    best_models = None
    best_score = np.Inf

    # Setup multiprocessing pool

    processing_pool = Pool(processes=exp['n_processes'], maxtasksperchild=exp['max_tasks_per_process'])

    try:
    
        # Perform search
        for depth in range(exp['max_depth']):

            # If debug reduce number of models for fast evaluation
            if exp['debug']:
                current_models = current_models[0:4]

            # Add random restarts to kernels
            current_models = ff.add_random_restarts(current_models, exp['n_rand'], exp['sd'], data_shape=data_shape)

            # Print result of expansion if debugging
            if exp['debug']:
                print('\nRandomly restarted kernels\n')
                for model in current_models:
                    print(model.pretty_print())

            # Remove any redundancy introduced into kernel expressions
            current_models = [model.simplified() for model in current_models]
            # Print result of simplification
            if exp['debug']:
                print('\nSimplified kernels\n')
                for model in current_models:
                    print(model.pretty_print())

            # Remove duplicate kernels
            current_models = ff.remove_duplicates(current_models)
            # Print result of duplicate removal
            if exp['debug']:
                print('\nDuplicate removed kernels\n')
                for model in current_models:
                    print(model.pretty_print())

            # Add jitter to parameter values (helps sticky optimisers)
            current_models = ff.add_jitter(current_models, exp['jitter_sd'])
            # Print result of jitter
            if exp['debug']:
                print('\nJittered kernels\n')
                for model in current_models:
                    print model.pretty_print()

            # Add the previous best models - in case we just need to optimise more rather than changing structure
            if not best_models is None:
                for a_model in best_models:
                    # noinspection PyUnusedLocal
                    current_models = current_models + [a_model.copy()] +\
                                     ff.add_jitter([a_model.copy() for dummy in range(exp['n_rand'])], exp['jitter_sd'])

            # Randomise the order of the model to distribute computational load evenly if running on cluster
            np.random.shuffle(current_models)

            # Print current models
            if exp['debug']:
                print('\nKernels to be evaluated\n')
                for model in current_models:
                    print(model.pretty_print())

            # Optimise models and score
            # new_results = [model.gpy_optimize(X=X, Y=Y,
            #                                   inference='exact',
            #                                   messages=exp['verbose'],
            #                                   max_iters=exp['iters'])
            #                for model in current_models]

            # A quick separate process hack
            # new_results = [optimize_separate_process(model, X, Y,
            #                                          inference='exact',
            #                                          messages=exp['verbose'],
            #                                          max_iters=exp['iters'])
            #                for model in current_models]

            for subset_percent in [13, 26, 52, 100]:
                subset_n = int(np.floor(X.shape[0] * subset_percent / 100.0))
                X_subset = X[:subset_n]
                Y_subset = Y[:subset_n]
                kwargs = dict(inference='exact',
                              messages=exp['verbose'],
                              max_iters=exp['iters'])
                new_results = processing_pool.map(optimise_single_model,
                                                  ((model, X_subset, Y_subset, kwargs)
                                                   for model in current_models))
                # Remove models that were optimised to be out of bounds (this is similar to a 0-1 prior)
                # TODO - put priors on hyperparameters
                new_results = [a_model for a_model in new_results if not a_model.out_of_bounds(data_shape)]
                oob_results = [a_model for a_model in new_results if a_model.out_of_bounds(data_shape)]
                oob_results = sorted(oob_results, key=lambda a_model: GPModel.score(a_model, exp['score']), reverse=True)
                oob_sequence.append(oob_results)

                # Some of the scores may have failed - remove nans to prevent sorting algorithms messing up
                (new_results, nan_results) = remove_nan_scored_models(new_results, exp['score'])
                nan_sequence.append(nan_results)
                assert(len(new_results) > 0) # FIXME - Need correct control flow if this happens

                # Sort the new results
                new_results = sorted(new_results, key=lambda a_model: GPModel.score(a_model, exp['score']),
                                     reverse=True)

                # Subset hack
                new_results = new_results[int(np.floor(len(new_results) * 9 / 10.0)):]
                current_models = new_results

            # Update user
            print('\nAll new results\n')
            for model in new_results:
                print('BIC=%0.1f' % model.bic,
                      # 'NLL=%0.1f' % model.nll,
                      # 'AIC=%0.1f' % model.aic,
                      # 'PL2=%0.3f' % model.pl2,
                      model.pretty_print())

            all_results = all_results + new_results
            all_results = sorted(all_results, key=lambda a_model: GPModel.score(a_model, exp['score']), reverse=True)

            results_sequence.append(all_results)

            # Extract the best k kernels from the new all_results
            best_results = sorted(new_results, key=lambda a_model: GPModel.score(a_model, exp['score']))[0:exp['k']]

            # Print best kernels if debugging
            if exp['debug']:
                print('\nBest models\n')
                for model in best_results:
                    print model.pretty_print()

            # Expand the best models
            current_models = grammar.expand_models(D=X.shape[1],
                                                   models=best_results,
                                                   base_kernels=exp['base_kernels'],
                                                   rules=exp['search_operators'])

            # Print expansion if debugging
            if exp['debug']:
                print('\nExpanded models\n')
                for model in current_models:
                    print(model.pretty_print())

            # Convert to additive form if desired
            if exp['additive_form']:
                current_models = [model.additive_form() for model in current_models]
                current_models = ff.remove_duplicates(current_models)

                # Print expansion
                if exp['debug']:
                    print('\Converted into additive\n')
                    for model in current_models:
                        print(model.pretty_print())

            # Reduce number of kernels when in debug mode
            if exp['debug']:
                current_models = current_models[0:4]

            # Write all_results to a temporary file at each level.
            all_results = sorted(all_results, key=lambda a_model: GPModel.score(a_model, exp['score']), reverse=True)
            with open(results_filename + '.unfinished', 'w') as outfile:
                outfile.write('Experiment all_results for\n datafile = %s\n\n %s \n\n'
                              % (experiment_data_file_name, exp_params_to_str(exp)))
                for (i, all_results) in enumerate(results_sequence):
                    outfile.write('\n%%%%%%%%%% Level %d %%%%%%%%%%\n\n' % i)
                    if exp['verbose_results']:
                        for result in all_results:
                            print >> outfile, result
                    else:
                        # Only print top k kernels - i.e. those used to seed the next level of the search
                        for result in sorted(all_results,
                                             key=lambda a_model: GPModel.score(a_model, exp['score']))[0:exp['k']]:
                            print >> outfile, result
            # Write nan scored kernels to a log file
            with open(results_filename + '.nans', 'w') as outfile:
                outfile.write('Experiment nan results for\n datafile = %s\n\n %s \n\n'
                              % (experiment_data_file_name, exp_params_to_str(exp)))
                for (i, nan_results) in enumerate(nan_sequence):
                    outfile.write('\n%%%%%%%%%% Level %d %%%%%%%%%%\n\n' % i)
                    for result in nan_results:
                        print >> outfile, result
            # Write oob kernels to a log file
            with open(results_filename + '.oob', 'w') as outfile:
                outfile.write('Experiment oob results for\n datafile = %s\n\n %s \n\n'
                              % (experiment_data_file_name, exp_params_to_str(exp)))
                for (i, nan_results) in enumerate(oob_sequence):
                    outfile.write('\n%%%%%%%%%% Level %d %%%%%%%%%%\n\n' % i)
                    for result in nan_results:
                        print >> outfile, result

            # Have we hit a stopping criterion?
            if 'no_improvement' in exp['stopping_criteria']:
                new_best_score = min(GPModel.score(a_model, exp['score']) for a_model in new_results)
                if new_best_score < best_score - exp['improvement_tolerance']:
                    best_score = new_best_score
                else:
                    # Insufficient improvement
                    print 'Insufficient improvement to score - stopping search'
                    break

        # Rename temporary results file to actual results file
        os.rename(results_filename + '.unfinished', results_filename)

    finally:
        processing_pool.close()
        processing_pool.join()


def run_experiment_file(filename):
    """
    This is intended to be the function that's called to initiate a series of experiments.
    """
    exp_string = open(filename, 'r').read()
    exp = eval(exp_string)
    exp = exp_param_defaults(exp)
    print(exp_params_to_str(exp))

    data_sets = list(gen_all_datasets(exp['data_dir']))

    # Create results directory if it doesn't exist.
    if not os.path.isdir(exp['results_dir']):
        os.makedirs(exp['results_dir'])

    if exp['random_order']:
        random.shuffle(data_sets)

    for r, a_file in data_sets:
        # Check if this experiment has already been done.
        output_file = os.path.join(exp['results_dir'], a_file + "_result.txt")
        if not(exp['skip_complete'] and (os.path.isfile(output_file))):
            print 'Experiment %s' % a_file
            print 'Output to: %s' % output_file
            data_file = os.path.join(r, a_file + ".npz")

            perform_experiment(data_file, output_file, exp)
            print "Finished file %s" % a_file
        else:
            print 'Skipping file %s' % a_file

    # os.system('reset')  # Stop terminal from going invisible - this may no longer be an issue


def perform_experiment(data_file, output_file, exp):

    data = np.load(data_file)
    X = data['X']
    Y = data['Y']

    if len(Y.shape) == 1:
        Y = np.array(Y, ndmin=2).T

    perform_kernel_search(X, Y, data_file, output_file, exp)
    # TODO - best model isn't what we should care about - Bayes model average more interesting
    best_model = parse_results(output_file)
    return best_model

    # os.system('reset')  # Stop terminal from going invisible - this may no longer be an issue


def remove_nan_scored_models(models, score):
    not_nan = [m for m in models if not np.isnan(ff.GPModel.score(m, criterion=score))]
    eq_nan = [m for m in models if np.isnan(ff.GPModel.score(m, criterion=score))]
    return not_nan, eq_nan


def parse_results(results_filenames, max_level=None):
    """
    Returns the best kernel in an experiment output file as a ScoredKernel
    """
    if not isinstance(results_filenames, list):
        # Backward compatibility wth specifying a single file
        results_filenames = [results_filenames]
    # Read relevant lines of file(s)
    result_tuples = []
    score = None
    for results_filename in results_filenames:
        lines = []
        with open(results_filename) as results_file:
            score = None
            for line in results_file:
                if line.startswith('score = '):
                    score = line[8:-2]
                elif line.startswith("GPModel"):
                    lines.append(line)
                elif (not max_level is None) and (len(re.findall('Level [0-9]+', line)) > 0):
                    level = int(line.split(' ')[2])
                    if level > max_level:
                        break
        result_tuples += [ff.repr_to_model(line.strip()) for line in lines]
    if not score is None:
        best_tuple = sorted(result_tuples, key=lambda a_model: GPModel.score(a_model, score))[0]
    else:
        best_tuple = sorted(result_tuples, key=GPModel.score)[0]
    return best_tuple


def exp_param_defaults(exp_params):
    """Sets all missing parameters to their default values"""
    defaults = dict(data_dir=os.path.join('..', 'data', 'debug'),       # Where to find the datasets.
                    results_dir=os.path.join('..', 'results', 'debug'), # Where to write the results.
                    description='Default description',
                    max_depth=4,                  # How deep to run the search.
                    random_order=False,           # Randomize the order of the datasets?
                    k=1,                          # Keep the k best kernels at every iteration. 1 => greedy search.
                    debug=False,                  # Makes search simpler in various ways to keep compute cost down
                    n_rand=9,                     # Number of random restarts.
                    sd=2,                         # Standard deviation of random restarts.
                    jitter_sd=0.1,                # Standard deviation of jitter.
                    max_jobs=500,                 # Maximum number of jobs to run at once on cluster.
                    verbose=True,                 # Talkative?
                    skip_complete=True,           # Whether to re-run already completed experiments.
                    iters=100,                    # How long to optimize hyperparameters for.
                    base_kernels='SE,Noise',      # Base kernels of language
                    additive_form=True,           # Restrict kernels to be in an additive form?
                    mean='ff.MeanZero()',         # Starting mean - zero
                    kernel='ff.NoiseKernel()',    # Starting kernel - noise
                    lik='ff.LikGauss(sf=-np.Inf)',# Starting likelihood - delta likelihood
                    verbose_results=False,        # Whether or not to record all kernels tested
                    random_seed=42,               # Random seed
                    period_heuristic=10,          # The minimum number of data points per period (roughly)
                    max_period_heuristic=5,       # Min number of periods that must be observed to declare periodicity
                    subset=False,                 # Optimise on a subset of the data?
                    subset_size=250,              # Size of data subset
                    full_iters=0,                 # Number of iters to perform on full data after subset optimisation
                    bundle_size=1,                # Number of kernel evaluations per job sent to cluster
                    score='BIC',                  # Search criterion
                    period_heuristic_type='both', # Related to minimum distance between data or something else
                    stopping_criteria=['no_improvement'], # Other reasons to stop the search
                    improvement_tolerance=0.1, # Minimum improvement for no_improvement stopping criterion
                    n_processes=None,             # Number of processes in multiprocessing.pool - None means max
                    max_tasks_per_process=2,      # This is set to one (or a small #) whilst there is a GPy memory leak
                    search_operators=[('A', ('+', 'A', 'B'), {'A': 'kernel', 'B': 'base'}),
                                      ('A', ('*', 'A', 'B'), {'A': 'kernel', 'B': 'base-not-const'}),
                                      #('A', ('*-const', 'A', 'B'), {'A': 'kernel', 'B': 'base-not-const'}),
                                      ('A', 'B', {'A': 'kernel', 'B': 'base'}),
                                      #('A', ('CP', 'd', 'A'), {'A': 'kernel', 'd' : 'dimension'}),
                                      #('A', ('CW', 'd', 'A'), {'A': 'kernel', 'd' : 'dimension'}),
                                      #('A', ('B', 'd', 'A'), {'A': 'kernel', 'd' : 'dimension'}),
                                      #('A', ('BL', 'd', 'A'), {'A': 'kernel', 'd' : 'dimension'}),
                                      ('A', ('None',), {'A': 'kernel'})]
                    )
    # Iterate through default key-value pairs, setting all unset keys
    for key, value in defaults.iteritems():
        if not key in exp_params:
            exp_params[key] = value
    return exp_params


def exp_params_to_str(exp_params):
    result = "Running experiment:\n"
    for key, value in exp_params.iteritems():
        result += "%s = %s,\n" % (key, value)
    return result


def gen_all_datasets(dir_name):
    """Looks through all .npz files in a directory, or just returns that file if it's only one."""
    if dir_name.endswith(".npz"):
        (r, f) = os.path.split(dir_name)
        (f, e) = os.path.splitext(f)
        return [(r, f)]

    file_list = []
    for r, d, f in os.walk(dir_name):
        for files in f:
            if files.endswith(".npz"):
                file_list.append((r, files.split('.')[-2]))
    file_list.sort()
    return file_list
   

def run_debug():
    """This is a quick debugging function."""
    run_experiment_file(os.path.join('..', 'experiments', 'debug', 'debug_example.py'))


if __name__ == "__main__":
    run_debug()
