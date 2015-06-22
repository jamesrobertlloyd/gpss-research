'''
Main file for setting up experiments, and compiling results.

@authors: David Duvenaud (dkd23@cam.ac.uk)
          James Robert Lloyd (jrl44@cam.ac.uk)
          Roger Grosse (rgrosse@mit.edu)
          
Created Jan 2013          
'''

from collections import namedtuple
from itertools import izip
import numpy as np
nax = np.newaxis
import os
import random
import re
import scipy.io

import flexible_function as ff
from flexible_function import GPModel
import grammar
import gpml
import utils.latex
import cblparallel
from cblparallel.util import mkstemp_safe
import job_controller as jc
import utils.misc
 
def remove_nan_scored_models(scored_kernels, score):    
    not_nan = [k for k in scored_kernels if not np.isnan(ff.GPModel.score(k, criterion=score))] 
    eq_nan = [k for k in scored_kernels if np.isnan(ff.GPModel.score(k, criterion=score))] 
    return (not_nan, eq_nan)
    
def perform_kernel_search(X, y, D, experiment_data_file_name, results_filename, exp):
    '''Search for the best kernel, in parallel on fear or local machine.'''
    
    # Initialise random seeds - randomness may be used in e.g. data subsetting

    utils.misc.set_all_random_seeds(exp.random_seed)
    
    # Create location, scale and minimum period parameters to pass around for initialisations

    data_shape = {}
    data_shape['x_mean'] = [np.mean(X[:,dim]) for dim in range(X.shape[1])]
    data_shape['y_mean'] = np.mean(y) #### TODO - should this be modified for non real valued data
    data_shape['x_sd'] = np.log([np.std(X[:,dim]) for dim in range(X.shape[1])])
    data_shape['y_sd'] = np.log(np.std(y)) #### TODO - should this be modified for non real valued data
    data_shape['y_min'] = np.min(y)
    data_shape['y_max'] = np.max(y)
    data_shape['x_min'] = [np.min(X[:,dim]) for dim in range(X.shape[1])]
    data_shape['x_max'] = [np.max(X[:,dim]) for dim in range(X.shape[1])]
    data_shape['x_min_abs_diff'] = np.log([utils.misc.min_abs_diff(X[:,i]) for i in range(X.shape[1])])

    # Initialise period at a multiple of the shortest / average distance between points, to prevent Nyquist problems.

    if exp.period_heuristic_type == 'none':
        data_shape['min_period'] = None
    if exp.period_heuristic_type == 'min':
        data_shape['min_period'] = np.log([exp.period_heuristic * utils.misc.min_abs_diff(X[:,i]) for i in range(X.shape[1])])
    elif exp.period_heuristic_type == 'average':
        data_shape['min_period'] = np.log([exp.period_heuristic * np.ptp(X[:,i]) / X.shape[0] for i in range(X.shape[1])])
    elif exp.period_heuristic_type == 'both':
        data_shape['min_period'] = np.log([max(exp.period_heuristic * utils.misc.min_abs_diff(X[:,i]), exp.period_heuristic * np.ptp(X[:,i]) / X.shape[0]) for i in range(X.shape[1])])
    else:
        warnings.warn('Unrecognised period heuristic type : using most conservative heuristic')
        data_shape['min_period'] = np.log([max(exp.period_heuristic * utils.misc.min_abs_diff(X[:,i]), exp.period_heuristic * np.ptp(X[:,i]) / X.shape[0]) for i in range(X.shape[1])])

    data_shape['max_period'] = [np.log((1.0/exp.max_period_heuristic)*(data_shape['x_max'][i] - data_shape['x_min'][i])) for i in range(X.shape[1])]

    # Initialise mean, kernel and likelihood

    m = eval(exp.mean)
    k = eval(exp.kernel)
    l = eval(exp.lik)
    current_models = [ff.GPModel(mean=m, kernel=k, likelihood=l, ndata=y.size)]

    print '\n\nStarting search with this model:\n'
    print current_models[0].pretty_print()
    print ''

    # Perform the initial expansion

    current_models = grammar.expand_models(D=D, models=current_models, base_kernels=exp.base_kernels, rules=exp.search_operators)

    # Convert to additive form if desired

    if exp.additive_form:
        current_models = [model.additive_form() for model in current_models]
        current_models = ff.remove_duplicates(current_models)   

    # Set up lists to record search
    
    all_results = [] # List of scored kernels
    results_sequence = [] # List of lists of results, indexed by level of expansion.
    nan_sequence = [] # List of list of nan scored results
    oob_sequence = [] # List of list of out of bounds results
    best_models = None

    # Other setup

    best_score = np.Inf
    
    # Perform search
    for depth in range(exp.max_depth):
        
        if exp.debug==True:
            current_models = current_models[0:4]
             
        # Add random restarts to kernels
        current_models = ff.add_random_restarts(current_models, exp.n_rand, exp.sd, data_shape=data_shape)

        # Print result of expansion
        if exp.debug:
            print '\nRandomly restarted kernels\n'
            for model in current_models:
                print model.pretty_print()
        
        # Remove any redundancy introduced into kernel expressions
        current_models = [model.simplified() for model in current_models]
        # Print result of simplification
        if exp.debug:
            print '\nSimplified kernels\n'
            for model in current_models:
                print model.pretty_print()
        current_models = ff.remove_duplicates(current_models)
        # Print result of duplicate removal
        if exp.debug:
            print '\nDuplicate removed kernels\n'
            for model in current_models:
                print model.pretty_print()
        
        # Add jitter to parameter values (empirically discovered to help optimiser)
        current_models = ff.add_jitter(current_models, exp.jitter_sd)
        # Print result of jitter
        if exp.debug:
            print '\nJittered kernels\n'
            for model in current_models:
                print model.pretty_print()
        
        # Add the previous best models - in case we just need to optimise more rather than changing structure
        if not best_models is None:
            for a_model in best_models:
                current_models = current_models + [a_model.copy()] + ff.add_jitter_to_models([a_model.copy() for dummy in range(exp.n_rand)], exp.jitter_sd)
        
        # Randomise the order of the model to distribute computational load evenly
        np.random.shuffle(current_models)

        # Print current models
        if exp.debug:
            print '\nKernels to be evaluated\n'
            for model in current_models:
                print model.pretty_print()
        
        # Optimise parameters of and score the kernels
        new_results = jc.evaluate_models(current_models, X, y, verbose=exp.verbose, local_computation=exp.local_computation,
                                          zip_files=True, max_jobs=exp.max_jobs, iters=exp.iters, random_seed=exp.random_seed,
                                          subset=exp.subset, subset_size=exp.subset_size, full_iters=exp.full_iters, bundle_size=exp.bundle_size)
            
        # Remove models that were optimised to be out of bounds (this is similar to a 0-1 prior)
        new_results = [a_model for a_model in new_results if not a_model.out_of_bounds(data_shape)]
        oob_results = [a_model for a_model in new_results if a_model.out_of_bounds(data_shape)]
        oob_results = sorted(oob_results, key=lambda a_model : GPModel.score(a_model, exp.score), reverse=True)
        oob_sequence.append(oob_results)
        
        # Some of the scores may have failed - remove nans to prevent sorting algorithms messing up
        (new_results, nan_results) = remove_nan_scored_models(new_results, exp.score)
        nan_sequence.append(nan_results)
        assert(len(new_results) > 0) # FIXME - Need correct control flow if this happens

        # Sort the new results
        new_results = sorted(new_results, key=lambda a_model : GPModel.score(a_model, exp.score), reverse=True)

        print '\nAll new results\n'
        for result in new_results:
            print 'NLL=%0.1f' % result.nll, 'BIC=%0.1f' % result.bic, 'AIC=%0.1f' % result.aic, 'PL2=%0.3f' % result.pl2, result.pretty_print()

        all_results = all_results + new_results
        all_results = sorted(all_results, key=lambda a_model : GPModel.score(a_model, exp.score), reverse=True)

        results_sequence.append(all_results)
        
        # Extract the best k kernels from the new all_results
        best_results = sorted(new_results, key=lambda a_model : GPModel.score(a_model, exp.score))[0:exp.k]

        # Print best kernels
        if exp.debug:
            print '\nBest models\n'
            for model in best_results:
                print model.pretty_print()
        
        # Expand the best models
        current_models = grammar.expand_models(D=D, models=best_results, base_kernels=exp.base_kernels, rules=exp.search_operators)

        # Print expansion
        if exp.debug:
            print '\nExpanded models\n'
            for model in current_models:
                print model.pretty_print()
        
        # Convert to additive form if desired
        if exp.additive_form:
            current_models = [model.additive_form() for model in current_models]
            current_models = ff.remove_duplicates(current_models)   

            # Print expansion
            if exp.debug:
                print '\Converted into additive\n'
                for model in current_models:
                    print model.pretty_print()
        
        # Reduce number of kernels when in debug mode
        if exp.debug==True:
            current_models = current_models[0:4]

        # Write all_results to a temporary file at each level.
        all_results = sorted(all_results, key=lambda a_model : GPModel.score(a_model, exp.score), reverse=True)
        with open(results_filename + '.unfinished', 'w') as outfile:
            outfile.write('Experiment all_results for\n datafile = %s\n\n %s \n\n' \
                          % (experiment_data_file_name, experiment_fields_to_str(exp)))
            for (i, all_results) in enumerate(results_sequence):
                outfile.write('\n%%%%%%%%%% Level %d %%%%%%%%%%\n\n' % i)
                if exp.verbose_results:
                    for result in all_results:
                        print >> outfile, result  
                else:
                    # Only print top k kernels - i.e. those used to seed the next level of the search
                    for result in sorted(all_results, key=lambda a_model : GPModel.score(a_model, exp.score))[0:exp.k]:
                        print >> outfile, result 
        # Write nan scored kernels to a log file
        with open(results_filename + '.nans', 'w') as outfile:
            outfile.write('Experiment nan results for\n datafile = %s\n\n %s \n\n' \
                          % (experiment_data_file_name, experiment_fields_to_str(exp)))
            for (i, nan_results) in enumerate(nan_sequence):
                outfile.write('\n%%%%%%%%%% Level %d %%%%%%%%%%\n\n' % i)
                for result in nan_results:
                    print >> outfile, result  
        # Write oob kernels to a log file
        with open(results_filename + '.oob', 'w') as outfile:
            outfile.write('Experiment oob results for\n datafile = %s\n\n %s \n\n' \
                          % (experiment_data_file_name, experiment_fields_to_str(exp)))
            for (i, nan_results) in enumerate(oob_sequence):
                outfile.write('\n%%%%%%%%%% Level %d %%%%%%%%%%\n\n' % i)
                for result in nan_results:
                    print >> outfile, result  

        # Have we hit a stopping criterion?
        if 'no_improvement' in exp.stopping_criteria:
            new_best_score = min(GPModel.score(a_model, exp.score) for a_model in new_results)
            if new_best_score < best_score - exp.improvement_tolerance:
                best_score = new_best_score
            else:
                # Insufficient improvement
                print 'Insufficient improvement to score - stopping search'
                break
    
    # Rename temporary results file to actual results file                
    os.rename(results_filename + '.unfinished', results_filename)

def parse_results(results_filenames, max_level=None):
    '''
    Returns the best kernel in an experiment output file as a ScoredKernel
    '''
    if not isinstance(results_filenames, list):
        # Backward compatibility wth specifying a single file
        results_filenames = [results_filenames]
    # Read relevant lines of file(s)
    result_tuples = []
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
        best_tuple = sorted(result_tuples, key=lambda a_model : GPModel.score(a_model, score))[0]
    else:
        best_tuple = sorted(result_tuples, key=GPModel.score)[0]
    return best_tuple

def gen_all_datasets(dir):
    """Looks through all .mat files in a directory, or just returns that file if it's only one."""
    if dir.endswith(".mat"):
        (r, f) = os.path.split(dir)
        (f, e) = os.path.splitext(f)
        return [(r, f)]
    
    file_list = []
    for r,d,f in os.walk(dir):
        for files in f:
            if files.endswith(".mat"):
                file_list.append((r, files.split('.')[-2]))
    file_list.sort()
    return file_list

# Defines a class that keeps track of all the options for an experiment.
# Maybe more natural as a dictionary to handle defaults - but named tuple looks nicer with . notation
class Experiment(namedtuple("Experiment", 'description, data_dir, max_depth, random_order, k, debug, local_computation, ' + \
                             'n_rand, sd, jitter_sd, max_jobs, verbose, make_predictions, skip_complete, results_dir, ' + \
                             'iters, base_kernels, additive_form, mean, kernel, lik, verbose_results, ' + \
                             'random_seed, period_heuristic, max_period_heuristic, ' + \
                             'subset, subset_size, full_iters, bundle_size, ' + \
                             'search_operators, score, period_heuristic_type, stopping_criteria, improvement_tolerance')):
    def __new__(cls, 
                data_dir,                     # Where to find the datasets.
                results_dir,                  # Where to write the results.
                description = 'no description',
                max_depth = 10,               # How deep to run the search.
                random_order = False,         # Randomize the order of the datasets?
                k = 1,                        # Keep the k best kernels at every iteration.  1 => greedy search.
                debug = False,
                local_computation = True,     # Run experiments locally, or on the cloud.
                n_rand = 9,                   # Number of random restarts.
                sd = 2,                       # Standard deviation of random restarts.
                jitter_sd = 0.1,              # Standard deviation of jitter.
                max_jobs=500,                 # Maximum number of jobs to run at once on cluster.
                verbose=False,
                make_predictions=False,       # Whether or not to forecast on a test set.
                skip_complete=True,           # Whether to re-run already completed experiments.
                iters=100,                    # How long to optimize hyperparameters for.
                base_kernels='SE,Per,Lin,Const',
                additive_form=False,          # Restrict kernels to be in an additive form?
                mean='ff.MeanZero()',      # Starting model
                kernel='ff.NoiseKernel()', # Starting kernel
                lik='ff.LikGauss(sf=-np.Inf)', # Starting likelihood 
                verbose_results=False,        # Whether or not to record all kernels tested
                random_seed=0,
                period_heuristic=10,          # The minimum number of data points per period (roughly)
                max_period_heuristic=5,       # The minimum number of periods that must be observed to declare periodicity
                subset=False,                 # Optimise on a subset of the data?
                subset_size=250,              # Size of data subset
                full_iters=0,                 # Number of iterations to perform on full data after subset optimisation
                bundle_size=1,                # Number of kernel evaluations per job sent to cluster 
                search_operators=None,        # Search operators used in grammar.py
                score='BIC',                  # Search criterion
                period_heuristic_type='both',
                stopping_criteria=[],
                improvement_tolerance=0.1):               
        return super(Experiment, cls).__new__(cls, description, data_dir, max_depth, random_order, k, debug, local_computation, \
                                              n_rand, sd, jitter_sd, max_jobs, verbose, make_predictions, skip_complete, results_dir, \
                                              iters, base_kernels, additive_form, mean, kernel, lik, verbose_results, \
                                              random_seed, period_heuristic, max_period_heuristic, \
                                              subset, subset_size, full_iters, bundle_size, \
                                              search_operators, score, period_heuristic_type, stopping_criteria, improvement_tolerance)

def experiment_fields_to_str(exp):
    str = "Running experiment:\n"
    for field, val in izip(exp._fields, exp):
        str += "%s = %s,\n" % (field, val)
    return str

def run_experiment_file(filename):
    """
    This is intended to be the function that's called to initiate a series of experiments.
    """       
    expstring = open(filename, 'r').read()
    exp = eval(expstring)
    print experiment_fields_to_str(exp)
    
    data_sets = list(gen_all_datasets(exp.data_dir))
    
    # Create results directory if it doesn't exist.
    if not os.path.isdir(exp.results_dir):
        os.makedirs(exp.results_dir)

    if exp.random_order:
        random.shuffle(data_sets)

    for r, file in data_sets:
        # Check if this experiment has already been done.
        output_file = os.path.join(exp.results_dir, file + "_result.txt")
        if not(exp.skip_complete and (os.path.isfile(output_file))):
            print 'Experiment %s' % file
            print 'Output to: %s' % output_file
            data_file = os.path.join(r, file + ".mat")

            perform_experiment(data_file, output_file, exp )
            print "Finished file %s" % file
        else:
            print 'Skipping file %s' % file

    os.system('reset')  # Stop terminal from going invisible.   

def repeat_predictions(filename):
    """
    A convenience function to re run the predictions from an experiment
    """ 

    expstring = open(filename, 'r').read()
    exp = eval(expstring)
    print experiment_fields_to_str(exp)

    if not exp.make_predictions:
        print 'This experiment does not make predictions'
        return None
    
    data_sets = list(gen_all_datasets(exp.data_dir))

    for r, file in data_sets:
        # Check if this experiment has already been done.
        output_file = os.path.join(exp.results_dir, file + "_result.txt")
        if os.path.isfile(output_file):
            print 'Predictions for %s' % file
            data_file = os.path.join(r, file + ".mat")

            X, y, D, Xtest, ytest = gpml.load_mat(data_file)
            prediction_file = os.path.join(exp.results_dir, os.path.splitext(os.path.split(data_file)[-1])[0] + "_predictions.mat")
            best_model = parse_results(output_file)
            predictions = jc.make_predictions(X, y, Xtest, ytest, best_model, local_computation=True,
                                              max_jobs=exp.max_jobs, verbose=exp.verbose, random_seed=exp.random_seed)
            scipy.io.savemat(prediction_file, predictions, appendmat=False)

            print "Finished file %s" % file
        else:
            print 'Results not found for %s' % file
    
def perform_experiment(data_file, output_file, exp):
    
    if exp.make_predictions:        
        X, y, D, Xtest, ytest = gpml.load_mat(data_file)
        prediction_file = os.path.join(exp.results_dir, os.path.splitext(os.path.split(data_file)[-1])[0] + "_predictions.mat")
    else:
        X, y, D = gpml.load_mat(data_file)
        
    perform_kernel_search(X, y, D, data_file, output_file, exp)
    best_model = parse_results(output_file)
    
    if exp.make_predictions:
        print '\nMaking predictions\n'
        predictions = jc.make_predictions(X, y, Xtest, ytest, best_model, local_computation=True,
                                          max_jobs=exp.max_jobs, verbose=exp.verbose, random_seed=exp.random_seed)
        scipy.io.savemat(prediction_file, predictions, appendmat=False)
        
    os.system('reset')  # Stop terminal from going invisible.   
   

def run_debug_kfold():
    """This is a quick debugging function."""
    run_experiment_file('../experiments/debug/debug_example.py')
