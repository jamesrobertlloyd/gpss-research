'''
Main file for dispatching jobs to a cluster, creates remote files, etc.

@authors: David Duvenaud (dkd23@cam.ac.uk)
          James Robert Lloyd (jrl44@cam.ac.uk)
          Roger Grosse (rgrosse@mit.edu)
          
Created Jan 2013
'''

import flexible_function as ff
from flexible_function import GPModel
import grammar
import gpml
import utils.latex
import utils.fear
try:
    import config
except:
    print '\n\nERROR : source/config.py not found\n\nPlease create it following example file as a guide\n\n'
    raise Exception('No config')
from utils import gaussians, psd_matrices

import numpy as np
nax = np.newaxis
# import pylab
import scipy.io
import sys
import os
import tempfile
import subprocess
import time

import cblparallel
from cblparallel.util import mkstemp_safe
import re

import shutil
import random

def evaluate_models(models, X, y, verbose=True, iters=300, local_computation=False, zip_files=False, max_jobs=500, random_seed=0, subset=False, subset_size=250, full_iters=0, bundle_size=1):
   
    # Make data into matrices in case they're unidimensional.
    if X.ndim == 1: X = X[:, nax]
    if y.ndim == 1: y = y[:, nax]
    ndata = y.shape[0]
    
    # Create data file
    if verbose:
        print 'Creating data file locally'
    data_file = cblparallel.create_temp_file('.mat')

    scipy.io.savemat(data_file, {'X': X, 'y': y})
    
    # Move to fear if necessary
    if not local_computation:
        if verbose:
            print 'Moving data file to fear'
        cblparallel.copy_to_remote(data_file)
    
    # Create a list of MATLAB scripts to assess and optimise parameters for each kernel
    if verbose:
        print 'Creating scripts'
    scripts = [None] * len(models)
    for (i, model) in enumerate(models):
        parameters = {'datafile': data_file.split('/')[-1],
                      'writefile': '%(output_file)s', # N.B. cblparallel manages output files
                      'gpml_path': cblparallel.gpml_path(local_computation),
                      'mean_syntax': model.mean.get_gpml_expression(dimensions=X.shape[1]),
                      'mean_params': '[ %s ]' % ' '.join(str(p) for p in model.mean.param_vector),
                      'kernel_syntax': model.kernel.get_gpml_expression(dimensions=X.shape[1]),
                      'kernel_params': '[ %s ]' % ' '.join(str(p) for p in model.kernel.param_vector),
                      'lik_syntax': model.likelihood.get_gpml_expression(dimensions=X.shape[1]),
                      'lik_params': '[ %s ]' % ' '.join(str(p) for p in model.likelihood.param_vector),
                      'inference': model.likelihood.gpml_inference_method,
                      'iters': str(iters),
                      'seed': str(np.random.randint(2**31)),
                      'subset': 'true' if subset else 'false',
                      'subset_size' : str(subset_size),
                      'full_iters' : str(full_iters)}

        scripts[i] = gpml.OPTIMIZE_KERNEL_CODE % parameters
        #### Need to be careful with % signs
        #### For the moment, cblparallel expects no single % signs - FIXME
        scripts[i] = re.sub('% ', '%% ', scripts[i])
    
    # Send to cblparallel and save output_files
    if verbose:
        print 'Sending scripts to cblparallel'
    if local_computation:
        output_files = cblparallel.run_batch_locally(scripts, language='matlab', max_cpu=1.1, job_check_sleep=5, submit_sleep=0.1, max_running_jobs=10, verbose=verbose)  
    else:
        output_files = cblparallel.run_batch_on_fear(scripts, language='matlab', max_jobs=max_jobs, verbose=verbose, zip_files=zip_files, bundle_size=bundle_size)  
    
    # Read in results
    results = [None] * len(models)
    for (i, output_file) in enumerate(output_files):
        if verbose:
            print 'Reading output file %d of %d' % (i + 1, len(models))
        results[i] = GPModel.from_matlab_output(gpml.read_outputs(output_file), models[i], ndata)
    
    # Tidy up local output files
    for (i, output_file) in enumerate(output_files):
        if verbose:
            print 'Removing output file %d of %d' % (i + 1, len(models)) 
        os.remove(output_file)
    # Remove temporary data file (perhaps on the cluster server)
    cblparallel.remove_temp_file(data_file, local_computation)
    
    # Return results i.e. list of ScoredKernel objects
    return results     

   
def make_predictions(X, y, Xtest, ytest, model, local_computation=False, max_jobs=500, verbose=True, random_seed=0, no_noise=False):
    # Make data into matrices in case they're unidimensional.
    if X.ndim == 1: X = X[:, nax]
    if y.ndim == 1: y = y[:, nax]
    ndata = y.shape[0]
    # Save temporary data file in standard temporary directory
    data_file = cblparallel.create_temp_file('.mat')
    scipy.io.savemat(data_file, {'X': X, 'y': y, 'Xtest' : Xtest, 'ytest' : ytest})
    # Copy onto cluster server if necessary
    if not local_computation:
        if verbose:
            print 'Moving data file to fear'
        cblparallel.copy_to_remote(data_file)
    # Create prediction code
    parameters ={'datafile': data_file.split('/')[-1],
                 'writefile': '%(output_file)s',
                 'gpml_path': cblparallel.gpml_path(local_computation),
                 'mean_syntax': model.mean.get_gpml_expression(dimensions=X.shape[1]),
                 'mean_params': '[ %s ]' % ' '.join(str(p) for p in model.mean.param_vector),
                 'kernel_syntax': model.kernel.get_gpml_expression(dimensions=X.shape[1]),
                 'kernel_params': '[ %s ]' % ' '.join(str(p) for p in model.kernel.param_vector),
                 'lik_syntax': model.likelihood.get_gpml_expression(dimensions=X.shape[1]),
                 'lik_params': '[ %s ]' % ' '.join(str(p) for p in model.likelihood.param_vector),
                 'inference': model.likelihood.gpml_inference_method,
                 'iters': str(30),
                 'seed': str(random_seed)}
    code = gpml.PREDICT_AND_SAVE_CODE % parameters
    code = re.sub('% ', '%% ', code) # HACK - cblparallel currently does not like % signs
    # Evaluate code - potentially on cluster
    if local_computation:   
        temp_results_file = cblparallel.run_batch_locally([code], language='matlab', max_cpu=1.1, max_mem=1.1, verbose=verbose)[0]
    else:
        temp_results_file = cblparallel.run_batch_on_fear([code], language='matlab', max_jobs=max_jobs, verbose=verbose)[0]
    results = scipy.io.loadmat(temp_results_file)
    # Remove temporary files (perhaps on the cluster server)
    cblparallel.remove_temp_file(temp_results_file, local_computation)
    cblparallel.remove_temp_file(data_file, local_computation)
    # Return dictionary of MATLAB results
    return results     

   
def make_predictions(X, y, Xtest, ytest, model, local_computation=False, max_jobs=500, verbose=True, random_seed=0, no_noise=False):
    # Make data into matrices in case they're unidimensional.
    if X.ndim == 1: X = X[:, nax]
    if y.ndim == 1: y = y[:, nax]
    ndata = y.shape[0]
    # Save temporary data file in standard temporary directory
    data_file = cblparallel.create_temp_file('.mat')
    scipy.io.savemat(data_file, {'X': X, 'y': y, 'Xtest' : Xtest, 'ytest' : ytest})
    # Copy onto cluster server if necessary
    if not local_computation:
        if verbose:
            print 'Moving data file to fear'
        cblparallel.copy_to_remote(data_file)
    # Create prediction code
    parameters ={'datafile': data_file.split('/')[-1],
                 'writefile': '%(output_file)s',
                 'gpml_path': cblparallel.gpml_path(local_computation),
                 'mean_syntax': model.mean.get_gpml_expression(dimensions=X.shape[1]),
                 'mean_params': '[ %s ]' % ' '.join(str(p) for p in model.mean.param_vector),
                 'kernel_syntax': model.kernel.get_gpml_expression(dimensions=X.shape[1]),
                 'kernel_params': '[ %s ]' % ' '.join(str(p) for p in model.kernel.param_vector),
                 'lik_syntax': model.likelihood.get_gpml_expression(dimensions=X.shape[1]),
                 'lik_params': '[ %s ]' % ' '.join(str(p) for p in model.likelihood.param_vector),
                 'inference': model.likelihood.gpml_inference_method,
                 'iters': str(30),
                 'seed': str(random_seed)}
    code = gpml.PREDICT_AND_SAVE_CODE % parameters
    code = re.sub('% ', '%% ', code) # HACK - cblparallel currently does not like % signs
    # Evaluate code - potentially on cluster
    if local_computation:   
        temp_results_file = cblparallel.run_batch_locally([code], language='matlab', max_cpu=1.1, max_mem=1.1, verbose=verbose)[0]
    else:
        temp_results_file = cblparallel.run_batch_on_fear([code], language='matlab', max_jobs=max_jobs, verbose=verbose)[0]
    results = scipy.io.loadmat(temp_results_file)
    # Remove temporary files (perhaps on the cluster server)
    cblparallel.remove_temp_file(temp_results_file, local_computation)
    cblparallel.remove_temp_file(data_file, local_computation)
    # Return dictionary of MATLAB results
    return results


