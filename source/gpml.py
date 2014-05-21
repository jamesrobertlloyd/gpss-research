'''
Some routines to interface with GPML.

@authors: 
          James Robert Lloyd (jrl44@cam.ac.uk)
          Roger Grosse (rgrosse@mit.edu)
          David Duvenaud (dkd23@cam.ac.uk)
'''

import numpy as np
nax = np.newaxis
import scipy.io
import tempfile, os
import subprocess
try:
    import config
except:
    print '\n\nERROR : source/config.py not found\n\nPlease create it following example file as a guide\n\n'
    raise Exception('No config')
import flexible_function as ff
import grammar

def run_matlab_code(code, verbose=False, jvm=True):
    # Write to a temp script
    (fd1, script_file) = tempfile.mkstemp(suffix='.m')
    (fd2, stdout_file) = tempfile.mkstemp(suffix='.txt')
    (fd3, stderr_file) = tempfile.mkstemp(suffix='.txt')
    
    f = open(script_file, 'w')
    f.write(code)
    f.close()
    
    jvm_string = '-nojvm'
    if jvm: jvm_string = ''
    call = [config.MATLAB_LOCATION, '-nosplash', jvm_string, '-nodisplay']
    print call
    
    stdin = open(script_file)
    stdout = open(stdout_file, 'w')
    stderr = open(stderr_file, 'w')
    subprocess.call(call, stdin=stdin, stdout=stdout, stderr=stderr)
    
    stdin.close()
    stdout.close()
    stderr.close()
    
    f = open(stderr_file)
    err_txt = f.read()
    f.close()
    
    os.close(fd1)
    os.close(fd2)
    os.close(fd3)    
    
    if verbose:
        print
        print 'Script file (%s) contents : ==========================================' % script_file
        print open(script_file, 'r').read()
        print
        print 'Std out : =========================================='        
        print open(stdout_file, 'r').read()   
    
    if err_txt != '':
        #### TODO - need to catch error local to new MLG machines
        print 'Matlab produced the following errors:\n\n%s' % err_txt  
#        raise RuntimeError('Matlab produced the following errors:\n\n%s' % err_txt)
    else:     
        # Only remove temporary files if run was successful    
        os.remove(script_file)
        os.remove(stdout_file)
        os.remove(stderr_file)
    


# Matlab code to optimise hyper-parameters on one file, given one kernel.
OPTIMIZE_KERNEL_CODE = r"""
rand('twister', %(seed)s);
randn('state', %(seed)s);

a='Load the data, it should contain X and y.'
load '%(datafile)s'
X = double(X)
y = double(y)

X_full = X;
y_full = y;

if %(subset)s & (%(subset_size)s < size(X, 1))
    subset = randsample(size(X, 1), %(subset_size)s, false)
    X = X_full(subset,:);
    y = y_full(subset);
end

%% Load GPML
addpath(genpath('%(gpml_path)s'));

%% Set up model.
meanfunc = %(mean_syntax)s
hyp.mean = %(mean_params)s

covfunc = %(kernel_syntax)s
hyp.cov = %(kernel_params)s

likfunc = %(lik_syntax)s
hyp.lik = %(lik_params)s

inference = %(inference)s

%% Optimise on subset
[hyp_opt, nlls] = minimize(hyp, @gp, -int32(%(iters)s * 3 / 3), inference, meanfunc, covfunc, likfunc, X, y);

%% Optimise on full data
if %(full_iters)s > 0
    hyp_opt = minimize(hyp_opt, @gp, -%(full_iters)s, inference, meanfunc, covfunc, likfunc, X_full, y_full);
end

%% Evaluate the nll on the full data
best_nll = gp(hyp_opt, inference, meanfunc, covfunc, likfunc, X_full, y_full)

save( '%(writefile)s', 'hyp_opt', 'best_nll', 'nlls');
%% exit();
"""

class OptimizerOutput:
    def __init__(self, mean_hypers, kernel_hypers, lik_hypers, nll, nlls):
        self.mean_hypers = mean_hypers
        self.kernel_hypers = kernel_hypers
        self.lik_hypers = lik_hypers
        self.nll = nll
        self.nlls = nlls

def read_outputs(write_file):
    gpml_result = scipy.io.loadmat(write_file)
    optimized_hypers = gpml_result['hyp_opt']
    nll = gpml_result['best_nll'][0, 0]
    nlls = gpml_result['nlls'].ravel()

    mean_hypers = optimized_hypers['mean'][0, 0].ravel()
    kernel_hypers = optimized_hypers['cov'][0, 0].ravel()
    lik_hypers = optimized_hypers['lik'][0, 0].ravel()
    
    return OptimizerOutput(mean_hypers, kernel_hypers, lik_hypers, nll, nlls)

# Matlab code to make predictions on a dataset.
PREDICT_AND_SAVE_CODE = r"""
rand('twister', %(seed)s);
randn('state', %(seed)s);

a='Load the data, it should contain X and y.'
load '%(datafile)s'
X = double(X)
y = double(y)
Xtest = double(Xtest)
ytest = double(ytest)

if size(ytest,1)==1
    ytest = ytest';
end

%% Load GPML
addpath(genpath('%(gpml_path)s'));

%% Set up model.

meanfunc = %(mean_syntax)s
hyp.mean = %(mean_params)s

covfunc = %(kernel_syntax)s
hyp.cov = %(kernel_params)s

likfunc = %(lik_syntax)s
hyp.lik = %(lik_params)s

inference = %(inference)s

model.hypers = hyp;

%% Evaluate at test points.
[ymu, ys2, predictions, fs2, loglik] = gp(model.hypers, inference, meanfunc, covfunc, likfunc, X, y, Xtest, ytest)

actuals = ytest;
timestamp = now

'%(writefile)s'

save('%(writefile)s', 'loglik', 'predictions', 'actuals', 'model', 'timestamp', 'ymu', 'ys2', 'Xtest', 'ytest', 'X', 'y');

a='Supposedly finished writing file'

%% exit();
"""

def standardise_and_save_data(X, y):
    if X.ndim == 1: X = X[:, nax]
    if y.ndim == 1: y = y[:, nax]
    data = {'X': X, 'y': y}
    (fd1, temp_data_file) = tempfile.mkstemp(suffix='.mat')
    scipy.io.savemat(temp_data_file, data)
    return (fd1, temp_data_file)

MATLAB_ORDER_BY_MAE_CODE = r"""
load '%(datafile)s'  %% Load the data, it should contain X and y.
X = double(X);
y = double(y);

addpath(genpath('%(gpml_path)s'));
addpath(genpath('%(matlab_script_path)s'));

mean_family = %(mean_syntax)s;
mean_params = %(mean_params)s;
kernel_family = %(kernel_syntax)s;
kernel_params = %(kernel_params)s;
lik_family = %(lik_syntax)s;
lik_params = %(lik_params)s;
kernel_family_list = %(kernel_syntax_list)s;
kernel_params_list = %(kernel_params_list)s;
figname = '%(figname)s';

order_by_mae_reduction(X, y, mean_family, mean_params, kernel_family, kernel_params, kernel_family_list, kernel_params_list, lik_family, lik_params, figname)
exit();"""

def order_by_mae(model, kernel_components, X, y, D, figname, skip_kernel_evaluation=False):
    matlab_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'matlab'))
    figname = os.path.abspath(os.path.join(os.path.dirname(__file__), figname))
    print 'Saving to: %s' % figname
    
    kernel_params_list = ','.join('[ %s ]' % ' '.join(str(p) for p in k.param_vector) for k in kernel_components)
    
    (fd1, temp_data_file) = standardise_and_save_data(X, y)
    
    code = MATLAB_ORDER_BY_MAE_CODE
    code = code % {'datafile': temp_data_file,
        'gpml_path': config.GPML_PATH,
        'matlab_script_path': matlab_dir,
        'mean_syntax': model.mean.get_gpml_expression(dimensions=D),
        'mean_params': '[ %s ]' % ' '.join(str(p) for p in model.mean.param_vector),
        'kernel_syntax': model.kernel.get_gpml_expression(dimensions=D),
        'kernel_params': '[ %s ]' % ' '.join(str(p) for p in model.kernel.param_vector),
        'lik_syntax': model.likelihood.get_gpml_expression(dimensions=D),
        'lik_params': '[ %s ]' % ' '.join(str(p) for p in model.likelihood.param_vector),
        'kernel_syntax_list': '{ %s }' % ','.join(str(k.get_gpml_expression(dimensions=D)) for k in kernel_components),
        'kernel_params_list': '{ %s }' % kernel_params_list,
        'figname': figname}
    
    if not skip_kernel_evaluation:
        run_matlab_code(code, verbose=True, jvm=True)
    os.close(fd1)
    # The below is commented out whilst debugging
    #os.remove(temp_data_file)
    mae_data = scipy.io.loadmat(figname + '_mae_data.mat')
    component_order = mae_data['idx'].ravel() - 1 # MATLAB to python OBOE
    return (component_order, mae_data)

MATLAB_COMPONENT_STATS_CODE = r"""
load '%(datafile)s'  %% Load the data, it should contain X and y.
X = double(X);
y = double(y);

addpath(genpath('%(gpml_path)s'));
addpath(genpath('%(matlab_script_path)s'));

mean_family = %(mean_syntax)s;
mean_params = %(mean_params)s;
kernel_family = %(kernel_syntax)s;
kernel_params = %(kernel_params)s;
lik_family = %(lik_syntax)s;
lik_params = %(lik_params)s;
kernel_family_list = %(kernel_syntax_list)s;
kernel_params_list = %(kernel_params_list)s;
envelope_family_list = %(envelope_syntax_list)s;
envelope_params_list = %(envelope_params_list)s;
figname = '%(figname)s';
idx = %(component_order)s;

component_stats_and_plots(X, y, mean_family, mean_params, kernel_family, kernel_params, kernel_family_list, kernel_params_list, envelope_family_list, envelope_params_list, lik_family, lik_params, figname, idx)
exit();"""

def component_stats(model, kernel_components, X, y, D, figname, component_order, skip_kernel_evaluation=False):
    matlab_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'matlab'))
    figname = os.path.abspath(os.path.join(os.path.dirname(__file__), figname))
    print 'Saving to: %s' % figname
    
    kernel_params_list = ','.join('[ %s ]' % ' '.join(str(p) for p in k.param_vector) for k in kernel_components)
    envelope_params_list = ','.join('[ %s ]' % ' '.join(str(p) for p in k.cp_structure().param_vector) for k in kernel_components)
    
    (fd1, temp_data_file) = standardise_and_save_data(X, y)
    
    code = MATLAB_COMPONENT_STATS_CODE
    code = code % {'datafile': temp_data_file,
        'gpml_path': config.GPML_PATH,
        'matlab_script_path': matlab_dir,
        'mean_syntax': model.mean.get_gpml_expression(dimensions=D),
        'mean_params': '[ %s ]' % ' '.join(str(p) for p in model.mean.param_vector),
        'kernel_syntax': model.kernel.get_gpml_expression(dimensions=D),
        'kernel_params': '[ %s ]' % ' '.join(str(p) for p in model.kernel.param_vector),
        'lik_syntax': model.likelihood.get_gpml_expression(dimensions=D),
        'lik_params': '[ %s ]' % ' '.join(str(p) for p in model.likelihood.param_vector),
        'kernel_syntax_list': '{ %s }' % ','.join(str(k.get_gpml_expression(dimensions=D)) for k in kernel_components),
        'kernel_params_list': '{ %s }' % kernel_params_list,
        'envelope_syntax_list': '{ %s }' % ','.join(str(k.cp_structure().get_gpml_expression(dimensions=D)) for k in kernel_components),
        'envelope_params_list': '{ %s }' % envelope_params_list,
        'figname': figname,
        'component_order': '[ %s ]' % ' '.join(str(comp) for comp in component_order+1)}
    
    if not skip_kernel_evaluation:
        run_matlab_code(code, verbose=True, jvm=True)
    os.close(fd1)
    # The below is commented out whilst debugging
    #os.remove(temp_data_file)
    component_data = scipy.io.loadmat(figname + '_component_data.mat')
    return component_data

MATLAB_CHECKING_STATS_CODE = r"""
load '%(datafile)s'  %% Load the data, it should contain X and y.
X = double(X);
y = double(y);

addpath(genpath('%(gpml_path)s'));
addpath(genpath('%(matlab_script_path)s'));

mean_family = %(mean_syntax)s;
mean_params = %(mean_params)s;
kernel_family = %(kernel_syntax)s;
kernel_params = %(kernel_params)s;
lik_family = %(lik_syntax)s;
lik_params = %(lik_params)s;
kernel_family_list = %(kernel_syntax_list)s;
kernel_params_list = %(kernel_params_list)s;
figname = '%(figname)s';
idx = %(component_order)s;
plot = %(plot)s;

checking_stats(X, y, mean_family, mean_params, kernel_family, kernel_params, kernel_family_list, kernel_params_list, lik_family, lik_params, figname, idx, plot)
exit();"""

def checking_stats(model, kernel_components, X, y, D, figname, component_order, make_plots=False, skip_kernel_evaluation=False):
    matlab_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'matlab'))
    figname = os.path.abspath(os.path.join(os.path.dirname(__file__), figname))
    print 'Saving to: %s' % figname
    
    kernel_params_list = ','.join('[ %s ]' % ' '.join(str(p) for p in k.param_vector) for k in kernel_components)
    
    (fd1, temp_data_file) = standardise_and_save_data(X, y)
    
    code = MATLAB_CHECKING_STATS_CODE
    code = code % {'datafile': temp_data_file,
        'gpml_path': config.GPML_PATH,
        'matlab_script_path': matlab_dir,
        'mean_syntax': model.mean.get_gpml_expression(dimensions=D),
        'mean_params': '[ %s ]' % ' '.join(str(p) for p in model.mean.param_vector),
        'kernel_syntax': model.kernel.get_gpml_expression(dimensions=D),
        'kernel_params': '[ %s ]' % ' '.join(str(p) for p in model.kernel.param_vector),
        'lik_syntax': model.likelihood.get_gpml_expression(dimensions=D),
        'lik_params': '[ %s ]' % ' '.join(str(p) for p in model.likelihood.param_vector),
        'kernel_syntax_list': '{ %s }' % ','.join(str(k.get_gpml_expression(dimensions=D)) for k in kernel_components),
        'kernel_params_list': '{ %s }' % kernel_params_list,
        'figname': figname,
        'component_order': '[ %s ]' % ' '.join(str(comp) for comp in component_order+1),
        'plot': 'true' if make_plots else 'false'}
    
    if not skip_kernel_evaluation:
        run_matlab_code(code, verbose=True, jvm=True)
    os.close(fd1)
    # The below is commented out whilst debugging
    #os.remove(temp_data_file)
    checking_stats = scipy.io.loadmat(figname + '_checking_stats.mat')
    return checking_stats

def load_mat(data_file, y_dim=0):
    '''
    Load a Matlab file containing inputs X and outputs y, output as np.arrays
     - X is (data points) x (input dimensions) array
     - y is (data points) x (output dimensions) array
     - y_dim selects which output dimension is returned (10 indexed)
    Returns tuple (X, y, # data points)
    '''
     
    data = scipy.io.loadmat(data_file)
    #### TODO - this should return a dictionary, not a tuple
    if 'Xtest' in data:
        return data['X'], data['y'][:,y_dim], np.shape(data['X'])[1], data['Xtest'], data['ytest'][:,y_dim]
    else:
        return data['X'], data['y'][:,y_dim], np.shape(data['X'])[1]

