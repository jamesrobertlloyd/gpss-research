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

import flexiblekernel as fk
from flexiblekernel import ScoredKernel
import grammar
import gpml
import utils.latex
import cblparallel
from cblparallel.util import mkstemp_safe
import job_controller as jc
import utils.misc
 
def remove_nan_scored_kernels(scored_kernels, score):    
    not_nan = [k for k in scored_kernels if not np.isnan(k.score(criterion=score))] 
    eq_nan = [k for k in scored_kernels if np.isnan(k.score(criterion=score))] 
    return (not_nan, eq_nan)
    
def perform_kernel_search(X, y, D, experiment_data_file_name, results_filename, exp):
    '''Search for the best kernel, in parallel on fear or local machine.'''
    
    # Initialise random seeds - randomness may be used in e.g. data subsetting
    utils.misc.set_all_random_seeds(exp.random_seed)

    if not exp.model_noise:
        # Initialise kernels to be all base kernels along all dimensions.
        current_kernels = list(fk.base_kernels(D, exp.base_kernels))
    else:
        # Initialise to white noise kernel
        #### FIXME - no need in principle for the mask kernel
        current_kernels = [fk.MaskKernel(D,0,fk.NoiseKernelFamily().default())]
        # And then expand as per usual
        current_kernels = grammar.expand_kernels(D, current_kernels, verbose=exp.verbose, debug=exp.debug, base_kernels=exp.base_kernels, rules=exp.search_operators)
        # Convert to additive form if desired
        if exp.additive_form:
            current_kernels = [grammar.additive_form(k) for k in current_kernels]
            # Using regular expansion rules followed by forcing additive results in lots of redundancy
            # TODO - this should happen always when other parts of code fixed
            # Remove any duplicates
            current_kernels = grammar.remove_duplicates(current_kernels)  
    
    # Create location, scale and minimum period parameters to pass around for initialisations
    data_shape = {}
    data_shape['input_location'] = [np.mean(X[:,dim]) for dim in range(X.shape[1])]
    data_shape['output_location'] = np.mean(y)
    data_shape['input_scale'] = np.log([np.std(X[:,dim]) for dim in range(X.shape[1])])
    data_shape['output_scale'] = np.log(np.std(y)) 
    data_shape['output_min'] = np.min(y)
    data_shape['output_max'] = np.max(y)
    ##### FIXME - only works in one dimension
    data_shape['input_min'] = [np.min(X[:,dim]) for dim in range(X.shape[1])][0]
    data_shape['input_max'] = [np.max(X[:,dim]) for dim in range(X.shape[1])][0]
    data_shape['min_integral_lengthscale'] = np.log(data_shape['input_max'] - data_shape['input_min']) - 2.5
    # Initialise period at a multiple of the shortest / average distance between points, to prevent Nyquist problems.
    if exp.use_min_period:
        if exp.period_heuristic_type == 'min':
            data_shape['min_period'] = np.log([exp.period_heuristic * utils.misc.min_abs_diff(X[:,i]) for i in range(X.shape[1])])
        elif exp.period_heuristic_type == 'average':
            data_shape['min_period'] = np.log([exp.period_heuristic * np.ptp(X[:,i]) / X.shape[0] for i in range(X.shape[1])])
        elif exp.period_heuristic_type == 'both':
            data_shape['min_period'] = np.log([max(exp.period_heuristic * utils.misc.min_abs_diff(X[:,i]), exp.period_heuristic * np.ptp(X[:,i]) / X.shape[0]) for i in range(X.shape[1])])
        else:
            warnings.warn('Unrecognised period heuristic type : using most conservative heuristic')
            data_shape['min_period'] = np.log([max(exp.period_heuristic * utils.misc.min_abs_diff(X[:,i]), exp.period_heuristic * np.ptp(X[:,i]) / X.shape[0]) for i in range(X.shape[1])])
    else:
        data_shape['min_period'] = None
    #### TODO - delete these constraints unless you have thought of a reason to keep them - not currently used
    if exp.use_constraints:
        data_shape['min_alpha'] = exp.alpha_heuristic
        data_shape['min_lengthscale'] = exp.lengthscale_heuristic + data_shape['input_scale']
    else:
        data_shape['min_alpha'] = None
        data_shape['min_lengthscale'] = None
    
    all_results = [] # List of scored kernels
    results_sequence = [] # List of lists of results, indexed by level of expansion.
    nan_sequence = [] # List of list of nan scored results
    oob_sequence = [] # List of list of out of bounds results
    
    noise = None # Initially have no guess at noise
    
    best_mae = np.Inf
    best_kernels = None
    #### Explanation : Sometimes, marginal likelihood does not pick a kernel that results in dramatically increased predictive performance
    ####               The below was included to track these kernels - no obvious wins for the search were noted (no crossover between optimising predictions and marginal liklelihood)
    best_predictor_sequence = [] # List of kernels that were good at predicting
    
    # Perform search
    for depth in range(exp.max_depth):
        
        if exp.debug==True:
            current_kernels = current_kernels[0:4]
             
        # Add random restarts to kernels
        current_kernels = fk.add_random_restarts(current_kernels, exp.n_rand, exp.sd, data_shape=data_shape)
        
        # Remove redundancy in kernel expressions - currently only works in additive mode
        if exp.additive_form:
            # Additive mode = True tells it to use dangerous hacky tricks that should be replaced
            current_kernels = [grammar.remove_redundancy(k, additive_mode=True) for k in current_kernels]
            current_kernels = grammar.remove_duplicates(current_kernels)
        
        # Add jitter to parameter values (empirically discovered to help broken optimiser - hopefully prevents excessive const kernel proliferation)
        current_kernels = fk.add_jitter(current_kernels, exp.jitter_sd)
        
        # Add the previous best kernels - in case we just need to optimise more rather than changing structure
        if not best_kernels is None:
            for kernel in best_kernels:
                current_kernels = current_kernels + [kernel.copy()] + fk.add_jitter([kernel.copy() for dummy in range(exp.n_rand)], exp.jitter_sd)
        
        #print 'Trying these kernels'
        #for result in current_kernels:
        #    print result.pretty_print()
        
        # Randomise the order of the kernels to distribute computationaal load evenly
        np.random.shuffle(current_kernels)
        
        # Optimise parameters of and score the kernels
        new_results = jc.evaluate_kernels(current_kernels, X, y, verbose=exp.verbose, noise = noise, local_computation=exp.local_computation,
                                          zip_files=True, max_jobs=exp.max_jobs, iters=exp.iters, zero_mean=exp.zero_mean, random_seed=exp.random_seed,
                                          subset=exp.subset, subset_size=exp.subset_size, full_iters=exp.full_iters, bundle_size=exp.bundle_size,
                                          no_noise=exp.no_noise)
                                          
        #print 'Raw results'
        #for result in new_results:
        #    print result.bic_nle, result.pic_nle, result.mae, result.k_opt.pretty_print()
            
        # Remove kernels that were optimised to be out of bounds (this is similar to a 0-1 prior)
        new_results = [sk for sk in new_results if not sk.k_opt.out_of_bounds(data_shape)]
        oob_results = [sk for sk in new_results if sk.k_opt.out_of_bounds(data_shape)]
        oob_results = sorted(oob_results, key=lambda sk : ScoredKernel.score(sk, exp.score), reverse=True)
        oob_sequence.append(oob_results)
            
        #print 'Removing out of bounds'
        #for result in new_results:
        #    print result.bic_nle, result.pic_nle, result.mae, result.k_opt.pretty_print()
        
        # Some of the scores may have failed - remove nans to prevent sorting algorithms messing up
        #### TODO - this should not fail silently like this
        (new_results, nan_results) = remove_nan_scored_kernels(new_results, exp.score)
        assert(len(new_results) > 0) # FIXME - Need correct control flow if this happens
        # Sort the new results
        new_results = sorted(new_results, key=lambda sk : ScoredKernel.score(sk, exp.score), reverse=True)

        nan_sequence.append(nan_results)
        
        #print 'All new results:'
        #for result in new_results:
        #    #print result.nll, result.laplace_nle, result.bic_nle, result.npll, result.pic_nle, result.k_opt.pretty_print()
        #    print result.bic_nle, result.pic_nle, result.mae, result.k_opt.pretty_print()
            
        #### DEBUG CODE

        #print 'NaNs:'
        #for result in nan_results:
        #    print 'BIC=%0.1f' % result.bic_nle, 'AIC=%0.1f' % result.aic_nle, 'Laplace=%0.1f' % result.laplace_nle, 'MAE=%0.1f' % result.mae, result.k_opt.pretty_print()
            
        #print 'OOBs:'
        #for result in oob_results:
        #    print 'BIC=%0.1f' % result.bic_nle, 'AIC=%0.1f' % result.aic_nle, 'Laplace=%0.1f' % result.laplace_nle, 'MAE=%0.1f' % result.mae, result.k_opt.pretty_print()

        print 'All new results after duplicate removal:'
        for result in new_results:
            #print result.nll, result.laplace_nle, result.bic_nle, result.npll, result.pic_nle, result.k_opt.pretty_print()
            print 'BIC=%0.1f' % result.bic_nle, 'AIC=%0.1f' % result.aic_nle, 'Laplace=%0.1f' % result.laplace_nle, 'MAE=%0.1f' % result.mae, result.k_opt.pretty_print()
            
        #### Explanation : This heuristic was not especially useful when first tried   
        # Remove bad predictors
        #old_best_mae = best_mae
        #best_mae = min(result.mae for result in new_results)
        #if old_best_mae == np.Inf:
        #    cut_off = best_mae * 2
        #elif best_mae > old_best_mae:
        #    cut_off = np.Inf
        #else:
        #    cut_off = best_mae + 0.5 * (old_best_mae - best_mae)
        #new_results = [result for result in new_results if (result.mae < cut_off)]

        #print 'All new results after removing bad predictors:'
        #for result in new_results:
        #    #print result.nll, result.laplace_nle, result.bic_nle, result.npll, result.pic_nle, result.k_opt.pretty_print()
        #    print result.bic_nle, result.pic_nle, result.mae, result.k_opt.pretty_print()

        all_results = all_results + new_results
        all_results = sorted(all_results, key=lambda sk : ScoredKernel.score(sk, exp.score), reverse=True)

        results_sequence.append(all_results)
        #if exp.verbose:
        #    print 'Printing all results'
        #    for result in all_results:
        #        #print result.nll, result.laplace_nle, result.bic_nle, result.npll, result.pic_nle, result.k_opt.pretty_print()
        #        print result.bic_nle, result.pic_nle, result.mae, result.k_opt.pretty_print()
        
        # Extract the best k kernels from the new all_results
        best_results = sorted(new_results, key=lambda sk : ScoredKernel.score(sk, exp.score))[0:exp.k]
        #### Explanation : This would be fixed if kernel objects know their noise - rather than just scored kernels
        ####               Ultimately we have to decide if we really want all kernels to have the form K + sigma^2*I
        ####               The answer is probably - but I don't think noise should be treated in a special way as it is currently
        #### FIXME - this only really works for k = 1 - see comment above
        noise = best_results[0].noise # Remember the best noise #### WARNING - this only really makes sense when k = 1 since other kernels may have different noise levels
        best_kernels = [r.k_opt for r in best_results]
        
        #### Explanation : This heuristic was not especially useful when first tried   
        # Add the best predicting kernel as well - might lead to a better marginal likelihood eventually
        #best_kernels = best_kernels + [sorted(new_results, key=lambda sk : ScoredKernel.score(sk, 'mae'))[0].k_opt]
        
        #### Explanation : Sometimes, marginal likelihood does not pick a kernel that results in dramatically increased predictive performance
        ####               The below was included to track these kernels - no obvious wins for the search were noted (no crossover between optimising predictions and marginal liklelihood) 
        best_predictor_sequence += [[sorted(new_results, key=lambda sk : ScoredKernel.score(sk, 'mae'))[0]]]
        
        # Expand the best kernels
        #### Question : Does the grammar expand kernels or is this really a search object?
        current_kernels = grammar.expand_kernels(D, best_kernels, verbose=exp.verbose, debug=exp.debug, base_kernels=exp.base_kernels, rules=exp.search_operators)
        
        # Convert to additive form if desired
        if exp.additive_form:
            current_kernels = [grammar.additive_form(k) for k in current_kernels]
            # Using regular expansion rules followed by forcing additive results in lots of redundancy
            # TODO - this should happen always when other parts of code fixed
            # Remove any duplicates
            current_kernels = grammar.remove_duplicates(current_kernels)
        
        # Reduce number of kernels when in debug mode
        if exp.debug==True:
            current_kernels = current_kernels[0:4]

        # Write all_results to a temporary file at each level.
        all_results = sorted(all_results, key=lambda sk : ScoredKernel.score(sk, exp.score), reverse=True)
        with open(results_filename + '.unfinished', 'w') as outfile:
            outfile.write('Experiment all_results for\n datafile = %s\n\n %s \n\n' \
                          % (experiment_data_file_name, experiment_fields_to_str(exp)))
            for (i, (best_predictors, all_results)) in enumerate(zip(best_predictor_sequence, results_sequence)):
                outfile.write('\n%%%%%%%%%% Level %d %%%%%%%%%%\n\n' % i)
                if exp.verbose_results:
                    for result in all_results:
                        print >> outfile, result  
                else:
                    # Only print top k kernels - i.e. those used to seed the next level of the search
                    for result in sorted(all_results, key=lambda sk : ScoredKernel.score(sk, exp.score))[0:exp.k]:
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
                elif line.startswith("ScoredKernel"):
                    lines.append(line)
                elif (not max_level is None) and (len(re.findall('Level [0-9]+', line)) > 0):
                    level = int(line.split(' ')[2])
                    if level > max_level:
                        break
        result_tuples += [fk.repr_string_to_kernel(line.strip()) for line in lines]
    if not score is None:
        best_tuple = sorted(result_tuples, key=lambda sk : ScoredKernel.score(sk, score))[0]
    else:
        best_tuple = sorted(result_tuples, key=ScoredKernel.score)[0]
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
                             'iters, base_kernels, additive_form, zero_mean, model_noise, no_noise, verbose_results, ' + \
                             'random_seed, use_min_period, period_heuristic, use_constraints, alpha_heuristic, ' + \
                             'lengthscale_heuristic, subset, subset_size, full_iters, bundle_size, ' + \
                             'search_operators, score, period_heuristic_type')):
    def __new__(cls, 
                data_dir,                     # Where to find the datasets.
                results_dir,                  # Where to write the results.
                description = 'no description',
                max_depth = 10,               # How deep to run the search.
                random_order = False,         # Randomize the order of the datasets?
                k = 1,                        # Keep the k best kernels at every iteration.  1 => greedy search.
                debug = False,
                local_computation = True,     # Run experiments locally, or on the cloud.
                n_rand = 2,                   # Number of random restarts.
                sd = 4,                       # Standard deviation of random restarts.
                jitter_sd = 0.5,              # Standard deviation of jitter.
                max_jobs=500,                 # Maximum number of jobs to run at once on cluster.
                verbose=False,
                make_predictions=False,       # Whether or not to forecast on a test set.
                skip_complete=True,           # Whether to re-run already completed experiments.
                iters=100,                    # How long to optimize hyperparameters for.
                base_kernels='SE,Per,Lin,Const',
                additive_form=False,          # Restrict kernels to be in an additive form?
                zero_mean=True,               # If false, use a constant mean function - cannot be used with the Const kernel
                model_noise=False,            # If true, the noise is included in the kernel and searched over
                no_noise=False,               # Noiseless likelihood - typically = model_noise
                verbose_results=False,        # Whether or not to record all kernels tested
                random_seed=0,
		        use_min_period=True,          # Whether to not let the period in a periodic kernel be smaller than the minimum period.
                period_heuristic=10,          # The minimum number of data points per period (roughly)
		        use_constraints=False,        # Place hard constraints on some parameter values? #### TODO - should be replaced with a prior / more Bayesian analysis
                alpha_heuristic=-2,           # Minimum alpha value for RQ kernel
                lengthscale_heuristic=-4.5,   # Minimum lengthscale 
                subset=False,                 # Optimise on a subset of the data?
                subset_size=250,              # Size of data subset
                full_iters=0,                 # Number of iterations to perform on full data after subset optimisation
                bundle_size=1,                # Number of kernel evaluations per job sent to cluster 
                search_operators=None,        # Search operators used in grammar.py
                score='BIC',                  # Search criterion
                period_heuristic_type='both'):               
        return super(Experiment, cls).__new__(cls, description, data_dir, max_depth, random_order, k, debug, local_computation, \
                                              n_rand, sd, jitter_sd, max_jobs, verbose, make_predictions, skip_complete, results_dir, \
                                              iters, base_kernels, additive_form, zero_mean, model_noise, no_noise, verbose_results, \
                                              random_seed, use_min_period, period_heuristic, use_constraints, alpha_heuristic, \
                                              lengthscale_heuristic, subset, subset_size, full_iters, bundle_size, \
                                              search_operators, score, period_heuristic_type)

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

def generate_model_fits(filename):
    """
    This is intended to be the function that's called to initiate a series of experiments.
    """       
    expstring = open(filename, 'r').read()
    exp = eval(expstring)
    exp = exp._replace(local_computation = True)
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
        if os.path.isfile(output_file):
            print 'Experiment %s' % file
            print 'Output to: %s' % output_file
            data_file = os.path.join(r, file + ".mat")

            calculate_model_fits(data_file, output_file, exp )
            print "Finished file %s" % file
        else:
            print 'Skipping file %s' % file

    os.system('reset')  # Stop terminal from going invisible. 
    
def perform_experiment(data_file, output_file, exp):
    
    if exp.make_predictions:        
        X, y, D, Xtest, ytest = gpml.load_mat(data_file, y_dim=1)
        prediction_file = os.path.join(exp.results_dir, os.path.splitext(os.path.split(data_file)[-1])[0] + "_predictions.mat")
    else:
        X, y, D = gpml.load_mat(data_file, y_dim=1)
        
    perform_kernel_search(X, y, D, data_file, output_file, exp)
    best_scored_kernel = parse_results(output_file)
    
    if exp.make_predictions:
        predictions = jc.make_predictions(X, y, Xtest, ytest, best_scored_kernel, local_computation=True,
                                          max_jobs=exp.max_jobs, verbose=exp.verbose, zero_mean=exp.zero_mean, random_seed=exp.random_seed,
                                          no_noise=exp.no_noise)
        scipy.io.savemat(prediction_file, predictions, appendmat=False)
        
    os.system('reset')  # Stop terminal from going invisible.
    
def calculate_model_fits(data_file, output_file, exp):
         
    prediction_file = os.path.join(exp.results_dir, os.path.splitext(os.path.split(data_file)[-1])[0] + "_predictions.mat")
    X, y, D, = gpml.load_mat(data_file, y_dim=1)
    Xtest = X
    ytest = y
        
    best_scored_kernel = parse_results(output_file)
    
    predictions = jc.make_predictions(X, y, Xtest, ytest, best_scored_kernel, local_computation=exp.local_computation,
                                      max_jobs=exp.max_jobs, verbose=exp.verbose, zero_mean=exp.zero_mean, random_seed=exp.random_seed)
    scipy.io.savemat(prediction_file, predictions, appendmat=False)
        
    os.system('reset')  # Stop terminal from going invisible.
   

def run_debug_kfold():
    """This is a quick debugging function."""
    run_experiment_file('../experiments/debug_example.py')
