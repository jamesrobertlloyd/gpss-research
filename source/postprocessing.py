"""
Contains helper functions to create figures and tables, based on the results of experiments.

@authors: David Duvenaud (dkd23@cam.ac.uk)
          James Robert Lloyd (jrl44@cam.ac.uk)
          Roger Grosse (rgrosse@mit.edu)
          
February 2013          
"""

import numpy as np
nax = np.newaxis
import os
import random
import scipy.io
import subprocess

import config
import experiment as exp
import flexible_function as ff
import gpml
import utils.latex
import re
import grammar
import translation

def compare_mse(folders, data_folder):
    if not isinstance(folders, list):
        folders = [folders] # Backward compatibility with specifying one folder
    data_sets = list(exp.gen_all_datasets(data_folder))
    RMSEs = np.inf * np.ones((len(data_sets), len(folders)))
    for (i, folder) in enumerate(folders):
        print ''
        print folder
        print ''
        # Load predictions file
        for (j, (r, data_file)) in enumerate(data_sets):
            print '%s : ' % data_file,
            results_file = os.path.join(folder, data_file + "_predictions.mat")
            if os.path.isfile(results_file):
                data = scipy.io.loadmat(results_file)
                RMSE = np.sqrt(np.mean(np.power(data['predictions'].ravel() - data['actuals'].ravel(), 2)))
                RMSEs[data_sets.index((r,data_file)), folders.index(folder)] = RMSE
                print '%f' % RMSE
            else:
                print ''
    np.set_printoptions(precision=3)
    standard_RMSEs = RMSEs / np.tile(np.min(RMSEs,1), (RMSEs.shape[1],1)).T # Divide by best algorithm
    print ''
    for folder in folders:
        print folder
    print ''
    print standard_RMSEs
    print ''
    medians = np.median(standard_RMSEs, 0)
    print medians
    return RMSEs

def gen_all_results(folder):
    """Look through all the files in the results directory"""
    file_list = sorted([f for (r,d,f) in os.walk(folder)][0])
    #for r,d,f in os.walk(folder):
    for files in file_list:
        if files.endswith(".txt"):
            results_filename = os.path.join(folder,files)
            best_tuple = exp.parse_results( results_filename )
            yield files.split('.')[-2], best_tuple
                

def make_all_1d_figures(folders, save_folder='../figures/decomposition/', prefix='', rescale=True, data_folder=None, skip_kernel_evaluation=False):
    """Crawls the results directory, and makes decomposition plots for each file.
    
    prefix is an optional string prepended to the output directory
    """    
    
    if not isinstance(folders, list):
        folders = [folders] # Backward compatibility with specifying one folder
    #### Quick fix to axis scaling
    #### TODO - Ultimately this and the shunt below should be removed / made elegant
    if rescale:
        data_sets = list(exp.gen_all_datasets("../data/1d_data_rescaled/"))
    else:
        if data_folder is None:
            data_sets = list(exp.gen_all_datasets("../data/1d_data/"))
        else:
            data_sets = list(exp.gen_all_datasets(data_folder))
    for r, file in data_sets:
        results_files = []
        for folder in folders:
            results_file = os.path.join(folder, file + "_result.txt")
            if os.path.isfile(results_file):
                results_files.append(results_file)
        # Is the experiment complete
        if len(results_files) > 0:
            # Find best kernel and produce plots
            datafile = os.path.join(r,file + ".mat")
            data = gpml.load_mat(datafile)
            X = data[0]
            y = data[1]
            D = data[2]
            if rescale:
                # Load unscaled data to remove scaling later
                unscaled_file = os.path.join('../data/1d_data/', re.sub('-s$', '', file) + '.mat')
                data = gpml.load_mat(unscaled_file)
                (X_unscaled, y_unscaled) = (data[0], data[1])
                (X_mean, X_scale) = (X_unscaled.mean(), X_unscaled.std())
                (y_mean, y_scale) = (y_unscaled.mean(), y_unscaled.std())
            else:
                (X_mean, X_scale, y_mean, y_scale) = (0,1,0,1)
                
            # A shunt to deal with a legacy issue.
            if datafile == '../data/1d_data/01-airline-months.mat':
                # Scaling should turn months starting at zero into years starting at 1949
                print "Special rescaling for airline months data"
                X_mean = X_mean + 1949
                X_scale = 1.0/12.0
                                
            model = exp.parse_results(results_files, max_level=max_level)
            model.kernel = ff.canonical(ff.simplify(model.kernel))
            print model.pretty_print()
            if not max_level is None:
                fig_folder = os.path.join(save_folder, (prefix + file + '_max_level_%d' % max_level))
            else:
                fig_folder = os.path.join(save_folder, (prefix + file))
            if not os.path.exists(fig_folder):
                os.makedirs(fig_folder)
            # Call gpml to plot the decomposition and evaluate the kernels
            (code, kernel_components) = gpml.plot_decomposition(model, X, y, os.path.join(fig_folder, file), X_mean, X_scale, y_mean, y_scale, dont_run_code_hack=skip_kernel_evaluation)
            # Now the kernels have been evaluated we can translate the revelant ones
            evaluation_data = scipy.io.loadmat(os.path.join(fig_folder, '%s_decomp_data.mat' % file))
            component_order = evaluation_data['idx'].ravel() - 1 # MATLAB to python OBOE
            evaluation_data['vars'] = evaluation_data['vars'].ravel()
            evaluation_data['cum_vars'] = evaluation_data['cum_vars'].ravel()
            evaluation_data['cum_resid_vars'] = evaluation_data['cum_resid_vars'].ravel()
            evaluation_data['MAEs'] = evaluation_data['MAEs'].ravel()
            evaluation_data['MAE_reductions'] = evaluation_data['MAE_reductions'].ravel()
            evaluation_data['monotonic'] = evaluation_data['monotonic'].ravel()
            evaluation_data['acf_min_p'] = evaluation_data['acf_min_p'].ravel()
            evaluation_data['acf_min_loc_p'] = evaluation_data['acf_min_loc_p'].ravel()
            evaluation_data['pxx_max_p'] = evaluation_data['pxx_max_p'].ravel()
            evaluation_data['pxx_max_loc_p'] = evaluation_data['pxx_max_loc_p'].ravel()
            evaluation_data['qq_d_max_p'] = evaluation_data['qq_d_max_p'].ravel()
            evaluation_data['qq_d_min_p'] = evaluation_data['qq_d_min_p'].ravel()
            i = 1
            short_descriptions = []
            while os.path.isfile(os.path.join(fig_folder, '%s_%d.fig' % (file, i))):
                # Describe this component
                (summary, sentences, extrap_sentences) = translation.translate_additive_component(kernel_components[component_order[i-1]], X, evaluation_data['monotonic'][i-1], evaluation_data['gradients'][i-1], 'year')
                short_descriptions.append(summary)
                paragraph = '.\n'.join(sentences) + '.'
                extrap_paragraph = '.\n'.join(extrap_sentences) + '.'
                with open(os.path.join(fig_folder, '%s_%d_description.tex' % (file, i)), 'w') as description_file:
                    description_file.write(paragraph)
                with open(os.path.join(fig_folder, '%s_%d_extrap_description.tex' % (file, i)), 'w') as description_file:
                    description_file.write(extrap_paragraph)
                with open(os.path.join(fig_folder, '%s_%d_short_description.tex' % (file, i)), 'w') as description_file:
                    description_file.write(summary + '.')
                i += 1
            # Produce the summary LaTeX document
            latex_summary = translation.produce_summary_document(file, i-1, evaluation_data, short_descriptions)
            with open(os.path.join(save_folder, os.pardir, '%s.tex' % file), 'w') as latex_file:
                latex_file.write(latex_summary)
        else:
            print "Cannnot find results for %s" % file
