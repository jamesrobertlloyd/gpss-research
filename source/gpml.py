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

%% Optimise...
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

%% Evaluate a test points.
[ymu, ys2, predictions, fs2, loglik] = gp(model.hypers, inference, meanfunc, covfunc, likfunc, X, y, Xtest, ytest)

actuals = ytest;
timestamp = now

'%(writefile)s'

save('%(writefile)s', 'loglik', 'predictions', 'actuals', 'model', 'timestamp');

a='Supposedly finished writing file'

%% exit();
"""

# Matlab code to decompose posterior into additive parts.
MATLAB_PLOT_DECOMP_CALLER_CODE = r"""
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
inference = '%(inference)s';
figname = '%(figname)s';
latex_names = %(latex_names)s;
full_kernel_name = %(full_kernel_name)s;
X_mean = %(X_mean)f;
X_scale = %(X_scale)f;
y_mean = %(y_mean)f;
y_scale = %(y_scale)f;

plot_decomp(X, y, mean_family, mean_params, kernel_family, kernel_params, kernel_family_list, kernel_params_list, lik_family, lik_params, figname, latex_names, full_kernel_name, X_mean, X_scale, y_mean, y_scale)
exit();"""


def plot_decomposition(model, X, y, D, figname, X_mean=0, X_scale=1, y_mean=0, y_scale=1, dont_run_code_hack=False):
    matlab_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'matlab'))
    figname = os.path.abspath(os.path.join(os.path.dirname(__file__), figname))
    print 'Plotting to: %s' % figname
    
    kernel_components = ff.break_kernel_into_summands(model.kernel)
    kernel_components = ff.SumKernel(kernel_components).simplified().canonical().operands
    latex_names = [k.latex.strip() for k in kernel_components]
    kernel_params_list = ','.join('[ %s ]' % ' '.join(str(p) for p in k.param_vector) for k in kernel_components)
    
    if X.ndim == 1: X = X[:, nax]
    if y.ndim == 1: y = y[:, nax]
    data = {'X': X, 'y': y}
    (fd1, temp_data_file) = tempfile.mkstemp(suffix='.mat')
    scipy.io.savemat(temp_data_file, data)
    
    code = MATLAB_PLOT_DECOMP_CALLER_CODE
    code = code % {'datafile': temp_data_file,
        'gpml_path': config.GPML_PATH,
        'matlab_script_path': matlab_dir,
        'mean_syntax': model.mean.get_gpml_expression(dimensions=D),
        'mean_params': '[ %s ]' % ' '.join(str(p) for p in model.mean.param_vector),
        'kernel_syntax': model.kernel.get_gpml_expression(dimensions=D),
        'kernel_params': '[ %s ]' % ' '.join(str(p) for p in model.kernel.param_vector),
        'lik_syntax': model.likelihood.get_gpml_expression(dimensions=D),
        'lik_params': '[ %s ]' % ' '.join(str(p) for p in model.likelihood.param_vector),
        'inference': model.likelihood.gpml_inference_method,
        'kernel_syntax_list': '{ %s }' % ','.join(str(k.get_gpml_expression(dimensions=D)) for k in kernel_components),
        'kernel_params_list': '{ %s }' % kernel_params_list,
        'latex_names': "{ ' %s ' }" % "','".join(latex_names),
        'full_kernel_name': "{ '%s' }" % model.kernel.latex.strip(), 
        'figname': figname,
        'X_mean' : X_mean,
        'X_scale' : X_scale,
        'y_mean' : y_mean,
        'y_scale' : y_scale}
    
    if not dont_run_code_hack:
        run_matlab_code(code, verbose=True, jvm=True)
    os.close(fd1)
    # The below is commented out whilst debugging
    #os.remove(temp_data_file)
    return (code, kernel_components)

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

