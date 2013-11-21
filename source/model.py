'''
Created Nov 2012

@authors: James Robert Lloyd (jrl44@cam.ac.uk)
          David Duvenaud (dkd23@cam.ac.uk)
          Roger Grosse (rgrosse@mit.edu)
'''

from __future__ import division

import itertools
from numpy import nan, inf
import numpy as np
import re

import operator
from utils import psd_matrices
import utils.misc
from utils.misc import colored, format_if_possible
from scipy.special import i0 # 0th order Bessel function of the first kind

##############################################
#                                            #
#               Base classes                 #
#                                            #
##############################################

# Base mean / kernel / likelihood function class

class FunctionWrapper:
            
    def __hash__(self): return hash(self.__repr__())

    # Properties

    @property
    def gpml_function(self): raise RuntimeError('This property must be overriden')
       
    @property    
    def is_operator(self): return False

    @property
    def id(self): raise RuntimeError('This property must be overriden')

    @property
    def effective_params(self):
        if not self.is_operator:
            '''This is true of all base functions, hence definition here'''  
            return len(self.param_vector)
        else:
            raise RuntimeError('Operators must override this property')
    
    @property
    def param_vector(self): raise RuntimeError('This property must be overriden')

    @property
    def latex(self): raise RuntimeError('This property must be overriden') 

    @property
    def depth(self):
        if not self.is_operator:
            return 0 
        else:
            raise RuntimeError('Operators must override this property')
    
    @property
    def num_params(self): return len(self.param_vector)

    @property
    def syntax(self): raise RuntimeError('This property must be overriden') 

    # Methods

    def copy(self): raise RuntimeError('This method must be overriden')

    def initialise_params(self, sd=1, data_shape=None): raise RuntimeError('This method must be overriden')

    def __repr__(self): return 'FunctionWrapper()'
    
    def pretty_print(self): return RuntimeError('This method must be overriden')
        
    def out_of_bounds(self, constraints): return False

    def load_param_vector(self, params): return RuntimeError('This method must be overriden')

    def __cmp__(self, other):
        if cmp(self.__class__, other.__class__):
            return cmp(self.__class__, other.__class__)
        return cmp(list(self.param_vector), list(other.param_vector))  

# Base mean function class with default properties and methods

class MeanFunction(FunctionWrapper):
    # Syntactic sugar e.g. f1 + f2
    def __add__(self, other):
        assert isinstance(other, MeanFunction)
        if isinstance(other, SumFunction):
            if isinstance(self, SumFunction):
                self.operands = self.operands + other.operands
                return self
            else:
                other.operands = [self] + other.operands
                return other
        else:
            return SumFunction([self, other])
    
    # Syntactic sugar e.g. f1 * f2
    def __mul__(self, other):
        assert isinstance(other, MeanFunction)
        if isinstance(other, ProductFunction):
            if isinstance(self, ProductFunction):
                self.operands = self.operands + other.operands
                return self
            else:
                other.operands = [self] + other.operands
                return other
        else:
            return ProductFunction([self, other])

    # Properties
       
    @property    
    def is_thunk(self): return False

    # Methods

    def get_gpml_expression(self, dimensions):
        if not self.is_operator:
            if self.is_thunk or (dimensions == 1):
                return self.gpml_function
            else:
                # Need to screen out dimensions
                assert (self.dimension < dimensions) and (not self.dimension is None)
                dim_vec = np.zeros(dimensions, dtype=int)
                dim_vec[self.dimension] = 1
                dim_vec_str = '[' + ' '.join(map(str, dim_vec)) + ']'
                return '{@meanMask, {%s, %s}}' % (dim_vec_str, self.gpml_function)
        else:
            raise RuntimeError('Operators must override this method')

    def __repr__(self): return 'MeanFunction()'

# Base kernel class with default properties and methods

class Kernel(FunctionWrapper):
    # Syntactic sugar e.g. k1 + k2
    def __add__(self, other):
        assert isinstance(other, Kernel)
        if isinstance(other, SumKernel):
            if isinstance(self, SumKernel):
                self.operands = self.operands + other.operands
                return canonical(self)
            else:
                other.operands = [self] + other.operands
                return canonical(other)
        else:
            return canonical(SumKernel([self, other]))
    
    # Syntactic sugar e.g. k1 * k2
    def __mul__(self, other):
        assert isinstance(other, Kernel)
        if isinstance(other, ProductKernel):
            if isinstance(self, ProductKernel):
                self.operands = self.operands + other.operands
                return canonical(self)
            else:
                other.operands = [self] + other.operands
                return canonical(other)
        else:
            return canonical(ProductKernel([self, other]))

    # Properties
       
    @property    
    def is_stationary(self): return True
       
    @property    
    def is_thunk(self): return False

    # Methods

    def get_gpml_expression(self, dimensions):
        if not self.is_operator:
            if self.is_thunk or (dimensions == 1):
                return self.gpml_function
            else:
                # Need to screen out dimensions
                assert (self.dimension < dimensions) and (not self.dimension is None)
                dim_vec = np.zeros(dimensions, dtype=int)
                dim_vec[self.dimension] = 1
                dim_vec_str = '[' + ' '.join(map(str, dim_vec)) + ']'
                return '{@covMask, {%s, %s}}' % (dim_vec_str, self.gpml_function)
        else:
            raise RuntimeError('Operators must override this method')

    def multiply_by_const(self, sf):
        if not self.is_operator:
            if hasattr(self, 'sf'):
                self.sf += sf
            else:
                raise RuntimeError('Kernels without a scale factor must overide this method')
        else:
            raise RuntimeError('Operators must override this method')

    def __repr__(self): return 'Kernel()'

# Base likelihood function class with default properties and methods

class Likelihood(FunctionWrapper):
    # Syntactic sugar e.g. l1 + l2
    def __add__(self, other):
        assert isinstance(other, Likelihood)
        if isinstance(other, SumLikelihood):
            if isinstance(self, SumLikelihood):
                self.operands = self.operands + other.operands
                return self
            else:
                other.operands = [self] + other.operands
                return other
        else:
            return SumLikelihood([self, other])
    
    # Syntactic sugar e.g. l1 * l2
    def __mul__(self, other):
        assert isinstance(other, Likelihood)
        if isinstance(other, ProductLikelihood):
            if isinstance(self, ProductLikelihood):
                self.operands = self.operands + other.operands
                return self
            else:
                other.operands = [self] + other.operands
                return other
        else:
            return ProductLikelihood([self, other])

    # Methods

    def get_gpml_expression(self, dimensions):
        if not self.is_operator:
            if self.is_thunk or (dimensions == 1):
                return self.gpml_function
            else:
                # Need to screen out dimensions
                assert self.dimension < dimensions
                dim_vec = np.zeros(dimensions, dtype=int)
                dim_vec[self.dimension] = 1
                dim_vec_str = '[' + ' '.join(map(str, dim_vec)) + ']'
                return '{@meanMask, {%s, %s}}' % (dim_vec_str, self.gpml_function)
        else:
            raise RuntimeError('Operators must override this method')

    def __repr__(self): return 'Likelihood()'

# Model class - this will take over from ScoredKernel

class RegressionModel:

    def __init__(self, mean=None, kernel=None, likelihood=None, nll=None, ndata=None):
        assert isinstance(mean, MeanFunction) or (mean is None)
        assert isinstance(kernel, Kernel) or (kernel is None)
        assert isinstance(likelihood, Likelihood) or (likelihood is None)
        self.mean = mean
        self.kernel = kernel
        self.likelihood = likelihood
        self.nll = nll
        self.ndata = ndata
            
    def __hash__(self): return hash(self.__repr__())

    def __repr__(self):
        # Remember all the various scoring criteria
        return 'RegressionModel(mean=%s, kernel=%s, likelihood=%s, nll=%s, ndata=%s)' % \
               (self.mean.__repr__(), self.kernel.__repr__(), self.likelihood.__repr__(), self.nll, self.ndata)

    def __cmp__(self, other):
        if cmp(self.__class__, other.__class__):
            return cmp(self.__class__, other.__class__)
        return cmp([self.mean, self.kernel, self.likelihood], [other.mean, other.kernel, other.likelihood])

    def pretty_print(self):
        return 'RegressionModel(mean=%s, kernel=%s, likelihood=%s)' % \
                (self.mean.pretty_print(), self.kernel.pretty_print(), self.likelihood.pretty_print())

    @property
    def bic(self):
        return 2 * self.nll + self.kernel.effective_params * np.log(self.ndata)

    @property
    def aic(self):
        return 2 * self.nll + self.kernel.effective_params * 2

    @property
    def pl2(self):
        return self.nll / self.ndata + self.kernel.effective_params / (2 * self.ndata)

    def score(self, criterion='bic'):
        return {'bic': self.bic,
                'aic': self.aic,
                'pl2': self.pl2,
                'nll': self.nll
                }[criterion.lower()]
                
    @staticmethod
    def from_printed_outputs(nll=None, ndata=None, noise=None, mean=None, kernel=None, likelihood=None):
        return RegressionModel(mean=mean, kernel=kernel, likelihood=likelihood, nll=nll, ndata=ndata)

    @staticmethod 
    def from_matlab_output(output, mean, kernel, likelihood, ndata):
        mean.load_param_vector(output.mean_hypers)
        kernel.load_param_vector(output.kernel_hypers)
        likelihood.load_param_vector(output.likelihood_hypers)
        return RegressionModel(mean=mean, kernel=kernel, likelihood=likelihood, nll=output.nll, ndata=ndata) 

##############################################
#                                            #
#              Mean functions                #
#                                            #
##############################################

class MeanZero(MeanFunction):
    def __init__(self):
        pass

    # Properties
        
    @property
    def gpml_function(self): return '{@meanZero}'

    @property    
    def is_thunk(self): return True
    
    @property
    def id(self): return 'Zero'
    
    @property
    def param_vector(self): return np.array([])
        
    @property
    def latex(self): return '{\\emptyset}' 
    
    @property
    def syntax(self): return colored('MZ', self.depth)

    # Methods

    def copy(self): return MeanZero()
        
    def initialise_params(self, sd=1, data_shape=None):
        pass
    
    def __repr__(self):
        return 'MeanZero()'
    
    def pretty_print(self):
        return colored('MZ', self.depth)   

    def load_param_vector(self, params):
        assert len(params) == 0

class MeanConst(MeanFunction):
    def __init__(self, c=None):
        self.c = c

    # Properties
        
    @property
    def gpml_function(self): return '{@meanConst}'

    @property    
    def is_thunk(self): return True
    
    @property
    def id(self): return 'Const'
    
    @property
    def param_vector(self): return np.array([self.c])
        
    @property
    def latex(self): return '{\\sc C}' 
    
    @property
    def syntax(self): return colored('C', self.depth)

    # Methods

    def copy(self): return MeanConst(c=self.c)
        
    def initialise_params(self, sd=1, data_shape=None):
        if self.c == None:
            # Set offset with data
            if np.random.rand() < 0.5:
                self.c = np.random.normal(loc=data_shape['y_mean'], scale=sd*data_shape['y_sd'])
            else:
                self.c = np.random.normal(loc=0, scale=sd*data_shape['y_sd'])
    
    def __repr__(self):
        return 'MeanConst(c=%s)' % (self.c)
    
    def pretty_print(self):
        return colored('C(c=%s)' % (format_if_possible('%1.1f', self.c)), self.depth)    

    def load_param_vector(self, params):
        c, = params # N.B. - expects list input
        self.c = c   

##############################################
#                                            #
#             Kernel functions               #
#                                            #
##############################################

# I hope this class can be deleted one day
class NoneKernel(Kernel):
    def __init__(self):
        pass

    def copy(self): return NoneKernel()
    
    def __repr__(self):
        return 'NoneKernel()'

    def multiply_by_const(self, sf):
        pass

class ZeroKernel(Kernel):
    def __init__(self):
        pass

    # Properties
        
    @property
    def gpml_function(self): return '{@covZero}'

    @property    
    def is_thunk(self): return True
    
    @property
    def id(self): return 'Zero'
    
    @property
    def param_vector(self): return np.array([self.sf])
        
    @property
    def latex(self): return '{\\sc Z}' 
    
    @property
    def syntax(self): return colored('Z', self.depth)

    # Methods

    def copy(self): return ZeroKernel()
        
    def initialise_params(self, sd=1, data_shape=None):
        pass
    
    def __repr__(self):
        return 'ZeroKernel()'
    
    def pretty_print(self):
        return colored('Z', self.depth)   

    def load_param_vector(self, params):
        pass

    def multiply_by_const(self, sf):
        pass

class NoiseKernel(Kernel):
    def __init__(self, sf=None):
        self.sf = sf

    # Properties
        
    @property
    def gpml_function(self): return '{@covNoise}'

    @property    
    def is_thunk(self): return True
    
    @property
    def id(self): return 'Noise'
    
    @property
    def param_vector(self): return np.array([self.sf])
        
    @property
    def latex(self): return '{\\sc WN}' 
    
    @property
    def syntax(self): return colored('WN', self.depth)

    # Methods

    def copy(self): return NoiseKernel(sf=self.sf)
        
    def initialise_params(self, sd=1, data_shape=None):
        if self.sf == None:
            # Set scale factor with 1/10 data std or neutrally
            if np.random.rand() < 0.5:
                self.sf = np.random.normal(loc=data_shape['y_sd']-np.log(10), scale=sd)
            else:
                self.sf = np.random.normal(loc=0, scale=sd)
    
    def __repr__(self):
        return 'NoiseKernel(sf=%s)' % (self.sf)
    
    def pretty_print(self):
        return colored('WN(sf=%s)' % (format_if_possible('%1.1f', self.sf)), self.depth)   

    def load_param_vector(self, params):
        sf, = params # N.B. - expects list input
        self.sf = sf  

class ConstKernel(Kernel):
    def __init__(self, sf=None):
        self.sf = sf

    # Properties
        
    @property
    def gpml_function(self): return '{@covConst}'

    @property    
    def is_thunk(self): return True
    
    @property
    def id(self): return 'Const'
    
    @property
    def param_vector(self): return np.array([self.sf])
        
    @property
    def latex(self): return '{\\sc C}' 
    
    @property
    def syntax(self): return colored('C', self.depth)

    # Methods

    def copy(self): return ConstKernel(sf=self.sf)
        
    def initialise_params(self, sd=1, data_shape=None):
        if self.sf == None:
            # Set scale factor with output location, scale or neutrally
            if rand < 1.0 / 3:
                self.sf = np.random.normal(loc=np.log(np.abs(data_shape['y_mean'])), scale=sd)
            elif rand < 2.0 / 3:
                self.sf = np.random.normal(loc=data_shape['y_sd'], scale=sd)
            else:
                self.sf = np.random.normal(loc=0, scale=sd)             
    
    def __repr__(self):
        return 'ConstKernel(sf=%s)' % (self.sf)
    
    def pretty_print(self):
        return colored('C(sf=%s)' % (format_if_possible('%1.1f', self.sf)), self.depth)   

    def load_param_vector(self, params):
        sf, = params # N.B. - expects list input
        self.sf = sf  

class SqExpKernel(Kernel):
    def __init__(self, dimension=None, lengthscale=None, sf=None):
        self.dimension = dimension
        self.lengthscale = lengthscale
        self.sf = sf

    # Properties
        
    @property
    def gpml_function(self): return '{@covSEiso}'
    
    @property
    def id(self): return 'SE'
    
    @property
    def param_vector(self): return np.array([self.lengthscale, self.sf])
        
    @property
    def latex(self): return '{\\sc SE}' 
    
    @property
    def syntax(self): return colored('SE_%s' % self.dimension, self.depth)

    # Methods

    def copy(self): return SqExpKernel(dimension=self.dimension, lengthscale=self.lengthscale, sf=self.sf)
        
    def initialise_params(self, sd=1, data_shape=None):
        if self.lengthscale == None:
            # Set lengthscale with input scale or neutrally
            if np.random.rand() < 0.5:
                self.lengthscale = np.random.normal(loc=data_shape['x_sd'][self.dimension], scale=sd)
            else:
                # Long lengthscale ~ infty = neutral
                self.lengthscale = np.random.normal(loc=np.log(2*(data_shape['x_max'][self.dimension]-data_shape['x_min'][self.dimension])), scale=sd)
        if self.sf == None:
            # Set scale factor with output scale or neutrally
            if np.random.rand() < 0.5:
                self.sf = np.random.normal(loc=data_shape['y_sd'], scale=sd)
            else:
                self.sf = np.random.normal(loc=0, scale=sd)         
    
    def __repr__(self):
        return 'SqExpKernel(dimension=%s, lengthscale=%s, sf=%s)' % \
               (self.dimension, self.lengthscale, self.sf)
    
    def pretty_print(self):
        return colored('SE(dim=%s, ell=%s, sf=%s)' % \
               (self.dimension, \
                format_if_possible('%1.1f', self.lengthscale), \
                format_if_possible('%1.1f', self.sf)), \
               self.depth)   

    def load_param_vector(self, params):
        lengthscale, sf = params # N.B. - expects list input
        self.lengthscale = lengthscale  
        self.sf = sf  

##############################################
#                                            #
#             Kernel operators               #
#                                            #
##############################################

class SumKernel(Kernel):
    def __init__(self, operands=None):
        if operands is None:
            self.operands = []
        else:
            self.operands  = operands

    # Properties

    @property    
    def is_stationary(self):
        return all(o.is_stationary for o in self.operands)

    @property
    def sf(self):
        if self.is_stationary:
            sf = 0
            for o in self.operands:
                sf += np.exp(2*o.sf)
            return 0.5*np.log(sf)
        else:
            raise RuntimeError('Cannot ask for scale factor of non-stationary kernel')
        
    @property
    def arity(self): return 'n'
        
    @property
    def gpml_function(self): return '{@covSum}'
    
    @property
    def id(self): return 'Sum'
    
    @property
    def param_vector(self):
        return np.concatenate([o.param_vector for o in self.operands])
        
    @property
    def latex(self):
        return '\\left( ' + ' + '.join([o.latex for o in self.operands]) + ' \\right)'  
    
    @property
    def syntax(self): 
        op = colored(' + ', self.depth)
        return colored('( ', self.depth) + \
            op.join([o.syntax for o in self.operands]) + \
            colored(' ) ', self.depth)
       
    @property    
    def is_operator(self): return True

    @property
    def effective_params(self):
        return sum([o.effective_params for o in self.operands])

    @property
    def depth(self):
        return max([o.depth for o in self.operands]) + 1

    # Methods

    def copy(self):
        return SumKernel(operands=[o.copy() for o in self.operands])
        
    def initialise_params(self, sd=1, data_shape=None):
        for o in self.operands:
            o.initialise_params(sd=sd, data_shape=data_shape)
    
    def __repr__(self):
        return 'SumKernel(operands=[%s])' % ', '.join(o.__repr__() for o in self.operands)
    
    def pretty_print(self):
        op = colored(' + ', self.depth)
        return colored('( ', self.depth) + \
            op.join([o.pretty_print() for o in self.operands]) + \
            colored(' ) ', self.depth)

    def load_param_vector(self, params):
        start = 0
        for o in self.operands:
            end = start + o.num_params
            o.load_param_vector(params[start:end])
            start = end

    def get_gpml_expression(self, dimensions):
        return '{@covSum, {%s}}' % ', '.join(o.get_gpml_expression(dimensions=dimensions) for o in self.operands)

    def multiply_by_const(self, sf):
        for o in self.operands:
            o.multiply_by_const(sf=sf)

class ProductKernel(Kernel):
    def __init__(self, operands=None):
        if operands is None:
            self.operands = []
        else:
            self.operands  = operands

    # Properties

    @property    
    def is_stationary(self):
        return all(o.is_stationary for o in self.operands)

    @property
    def sf(self):
        if self.is_stationary:
            return sum(o.sf for o in self.operands)
        else:
            raise RuntimeError('Cannot ask for scale factor of non-stationary kernel')
        
    @property
    def arity(self): return 'n'
        
    @property
    def gpml_function(self): return '{@covProd}'
    
    @property
    def id(self): return 'Product'
    
    @property
    def param_vector(self):
        return np.concatenate([o.param_vector for o in self.operands])
        
    @property
    def latex(self):
        return ' \\times '.join([o.latex for o in self.operands])  
    
    @property
    def syntax(self): 
        op = colored(' x ', self.depth)
        return colored('( ', self.depth) + \
            op.join([o.syntax for o in self.operands]) + \
            colored(' ) ', self.depth)
       
    @property    
    def is_operator(self): return True

    @property
    def effective_params(self):
        return sum([o.effective_params for o in self.operands]) - (len(self.operands) - 1)

    @property
    def depth(self):
        return max([o.depth for o in self.operands]) + 1

    # Methods

    def copy(self):
        return ProductKernel(operands=[o.copy() for o in self.operands])
        
    def initialise_params(self, sd=1, data_shape=None):
        for o in self.operands:
            o.initialise_params(sd=sd, data_shape=data_shape)
    
    def __repr__(self):
        return 'ProductKernel(operands=[%s])' % ', '.join(o.__repr__() for o in self.operands)
    
    def pretty_print(self):
        op = colored(' x ', self.depth)
        return colored('( ', self.depth) + \
            op.join([o.pretty_print() for o in self.operands]) + \
            colored(' ) ', self.depth)

    def load_param_vector(self, params):
        start = 0
        for o in self.operands:
            end = start + o.num_params
            o.load_param_vector(params[start:end])
            start = end

    def get_gpml_expression(self, dimensions):
        return '{@covProd, {%s}}' % ', '.join(o.get_gpml_expression(dimensions=dimensions) for o in self.operands)

    def multiply_by_const(self, sf):
        self.operands[0].multiply_by_const(sf=sf)

class ChangePointKernel(Kernel):
    pass

class ChangeBurstKernel(Kernel):
    pass

##############################################
#                                            #
#           Likelihood functions             #
#                                            #
##############################################

class LikGauss(Likelihood):
    def __init__(self, sf=None):
        self.sf = sf

    # Properties
        
    @property
    def gpml_function(self): return '{@likGauss}'

    @property    
    def is_thunk(self): return True
    
    @property
    def id(self): return 'Gauss'
    
    @property
    def param_vector(self): return np.array([self.sf])
        
    @property
    def latex(self): return '{\\sc GS}' 
    
    @property
    def syntax(self): return colored('GS', self.depth)

    @property
    def effective_params(self):
        if self.sf == -np.Inf:
            return 0
        else:
            return 1

    # Methods

    def copy(self): return LikGauss(sf=self.sf)
        
    def initialise_params(self, sd=1, data_shape=None):
        if self.sf == None:
            # Set scale factor with 1/10 data std or neutrally
            if np.random.rand() < 0.5:
                self.sf = np.random.normal(loc=data_shape['y_sd']-np.log(10), scale=sd)
            else:
                self.sf = np.random.normal(loc=0, scale=sd)
    
    def __repr__(self):
        return 'LikGauss(sf=%s)' % (self.sf)
    
    def pretty_print(self):
        return colored('GS(sf=%s)' % (format_if_possible('%1.1f', self.sf)), self.depth)   

    def load_param_vector(self, params):
        sf, = params # N.B. - expects list input
        self.sf = sf   

##############################################
#                                            #
#           Kernel manipulation              #
#                                            #
##############################################

def canonical(k):
    '''Sorts a kernel tree into a canonical form.'''
    if not k.is_operator:
        return k
    elif k.arity == 2:
        for o in k.operands:
            o = canonical(o)
        if isinstance(k.operands[0], NoneKernel) or isinstance(k.operands[1], NoneKernel):
            return NoneKernel()
        else:
            return k
    else:
        new_ops = []
        for op in k.operands:
            op_canon = canonical(op)
            if isinstance(op_canon, k.__class__):
                new_ops += op_canon.operands
            elif not isinstance(op_canon, NoneKernel):
                new_ops.append(op_canon)
        if len(new_ops) == 0:
            return NoneKernel()
        elif len(new_ops) == 1:
            return new_ops[0]
        else:
            k.operands = new_ops
            return k

def simplify(k):
    # TODO - how many times do these need to be done?
    k = collapse_additive_idempotency(k)
    k = collapse_multiplicative_idempotency(k)
    k = collapse_multiplicative_identity(k)
    k = collapse_multiplicative_zero(k)
    k = canonical(k)
    k = collapse_additive_idempotency(k)
    k = collapse_multiplicative_idempotency(k)
    k = collapse_multiplicative_identity(k)
    k = collapse_multiplicative_zero(k)
    return canonical(k)

def collapse_additive_idempotency(k):
    # TODO - abstract this behaviour
    k = canonical(k)
    if not k.is_operator:
        return k
    elif isinstance(k, SumKernel):
        ops = [collapse_additive_idempotency(o) for o in k.operands]
        # Count the number of white noises
        sf = 0
        WN_count = 0
        not_WN_ops = []
        for op in ops:
            if isinstance(op, NoiseKernel):
                WN_count += 1
                sf += np.exp(2*op.sf)
            else:
                not_WN_ops.append(op)
        # Compactify if necessary
        if WN_count > 0:
            ops = not_WN_ops + [NoiseKernel(sf=0.5*np.log(sf))]
        # Now count the number of constants
        sf = 0
        const_count = 0
        not_const_ops = []
        for op in ops:
            if isinstance(op, ConstKernel):
                const_count += 1
                sf += np.exp(2*op.sf)
            else:
                not_const_ops.append(op)
         # Compactify if necessary
        if (const_count > 0):
            ops = not_const_ops + [ConstKernel(sf=0.5*np.log(sf))]
        # Finish
        k.operands = ops
        return canonical(k)
    else:
        for o in k.operands:
            o = collapse_additive_idempotency(o)
        return k

def collapse_multiplicative_idempotency(k):
    # TODO - abstract this behaviour
    k = canonical(k)
    if not k.is_operator:
        return k
    elif isinstance(k, ProductKernel):
        ops = [collapse_multiplicative_idempotency(o) for o in k.operands]
        # Count the number of SEs in different dimensions
        lengthscales = {}
        sfs = {}
        not_SE_ops = []
        for op in ops:
            if isinstance(op, SqExpKernel):
                if not lengthscales.has_key(op.dimension):
                    lengthscales[op.dimension] = np.Inf
                    sfs[op.dimension] = 0
                lengthscales[op.dimension] = -0.5 * np.log(np.exp(-2*lengthscales[op.dimension]) + np.exp(-2*op.lengthscale))
                sfs[op.dimension] += op.sf
            else:
                not_SE_ops.append(op)
        # Compactify if necessary
        ops = not_SE_ops
        for dimension in lengthscales:
            ops += [SqExpKernel(dimension=dimension, lengthscale=lengthscales[dimension], sf=sfs[dimension])]
        # Count the number of white noises
        sf = 0
        WN_count = 0
        not_WN_ops = []
        for op in ops:
            if isinstance(op, NoiseKernel):
                WN_count += 1
                sf += op.sf
            else:
                not_WN_ops.append(op)
        # Compactify if necessary
        if WN_count > 0:
            ops = not_WN_ops + [NoiseKernel(sf=sf)]
        # Now count the number of constants
        sf = 0
        const_count = 0
        not_const_ops = []
        for op in ops:
            if isinstance(op, ConstKernel):
                const_count += 1
                sf += op.sf
            else:
                not_const_ops.append(op)
         # Compactify if necessary
        if const_count > 0:
            ops = not_const_ops + [ConstKernel(sf=sf)]
        # Finish
        k.operands = ops
        return canonical(k)
    else:
        for o in k.operands:
            o = collapse_multiplicative_idempotency(o)
        return k

def collapse_multiplicative_zero(k):
    # TODO - abstract this behaviour
    k = canonical(k)
    if not k.is_operator:
        return k
    elif isinstance(k, ProductKernel):
        ops = [collapse_multiplicative_zero(o) for o in k.operands]
        sf = 0
        WN_count = 0
        not_WN_ops = []
        for op in ops:
            if isinstance(op, NoiseKernel):
                WN_count += 1
                sf += op.sf
            elif op.is_stationary:
                sf += op.sf
            else:
                not_WN_ops.append(op)
        # Compactify if necessary
        if WN_count > 0:
            ops = not_WN_ops + [NoiseKernel(sf=sf)]
        # Finish
        k.operands = ops
        return canonical(k)
    else:
        for o in k.operands:
            o = collapse_multiplicative_zero(o)
        return k

def collapse_multiplicative_identity(k):
    # TODO - abstract this behaviour
    k = canonical(k)
    if not k.is_operator:
        return k
    elif isinstance(k, ProductKernel):
        ops = [collapse_multiplicative_identity(o) for o in k.operands]
        sf = 0
        const_count = 0
        not_const_ops = []
        for op in ops:
            if isinstance(op, ConstKernel):
                const_count += 1
                sf += op.sf
            else:
                not_const_ops.append(op)
        # Compactify if necessary
        if const_count > 0:
            ops = not_const_ops
            ops[0].multiply_by_const(sf=sf)
        # Finish
        k.operands = ops
        return canonical(k)
    else:
        for o in k.operands:
            o = collapse_multiplicative_identity(o)
        return k

def break_kernel_into_summands(k):
    '''Takes a kernel, expands it into a polynomial, and breaks terms up into a list.
    
    Mutually Recursive with distribute_products().
    Always returns a list.
    '''    
    # First, recursively distribute all products within the kernel.
    k_dist = distribute_products(k)
    
    if isinstance(k_dist, SumKernel):
        # Break the summands into a list of kernels.
        return list(k_dist.operands)
    else:
        return [k_dist]

def distribute_products(k):
    """Distributes products to get a polynomial.
    
    Mutually recursive with break_kernel_into_summands().
    Always returns a sumkernel.
    """

    if isinstance(k, ProductKernel):
        # Recursively distribute each of the terms to be multiplied.
        distributed_ops = [break_kernel_into_summands(op) for op in k.operands]
        
        # Now produce a sum of all combinations of terms in the products. Itertools is awesome.
        new_prod_ks = [ProductKernel( operands=prod ) for prod in itertools.product(*distributed_ops)]
        return SumKernel(operands=new_prod_ks)
    
    elif isinstance(k, SumKernel):
        # Recursively distribute each the operands to be summed, then combine them back into a new SumKernel.
        return SumKernel([subop for op in k.operands for subop in break_kernel_into_summands(op)])
    elif k.is_operator:
        if k.arity == 2:
            summands = []
            operands_list = [[op, ZeroKernel()] for op in break_kernel_into_summands(k.operands[0])]
            for ops in operand_list:
                k_new = k.copy()
                k_new.operands = ops
                summands.append(k_new)
            operands_list = [[ZeroKernel(), op] for op in break_kernel_into_summands(k.operands[1])]
            for ops in operand_list:
                k_new = k.copy()
                k_new.operands = ops
                summands.append(k_new)
            return SumKernel(operands=summands)
        else:
            raise RuntimeError('Not sure how to distribute products of this operator')
    else:
        # Base case: A kernel that's just, like, a kernel, man.
        return k

def additive_form(k):
    '''
    Converts a kernel into a sum of products and with changepoints percolating to the top
    Output is always in canonical form
    '''
    #### TODO - currently implemented for a subset of changepoint operators - to be extended or operators to be abstracted
    if isinstance(k, ProductKernel):
        # Convert operands into additive form
        additive_ops = sorted([additive_form(op) for op in k.operands])
        # Initialise the new kernel
        new_kernel = additive_ops[0]
        # Build up the product, iterating over the other components
        for additive_op in additive_ops[1:]:
            if isinstance(new_kernel, ChangePointKernel) or isinstance(new_kernel, ChangeBurstKernel):
                # Changepoints take priority - nest the products within this operator
                new_kernel.operands = [additive_form(canonical(op*additive_op.copy())) for op in new_kernel.operands]
            elif isinstance(additive_op, ChangePointKernel) or isinstance(additive_op, ChangeBurstKernel):
                # Nest within the next operator
                old_kernel = new_kernel.copy()
                new_kernel = additive_op
                new_kernel.operands = [additive_form(canonical(op*old_kernel.copy())) for op in new_kernel.operands]
            elif isinstance(new_kernel, SumKernel):
                # Nest the products within this sum
                new_kernel.operands = [additive_form(canonical(op*additive_op.copy())) for op in new_kernel.operands]
            elif isinstance(additive_op, SumKernel):
                # Nest within the next operator
                old_kernel = new_kernel.copy()
                new_kernel = additive_op
                new_kernel.operands = [additive_form(canonical(op*old_kernel.copy())) for op in new_kernel.operands]
            else:
                # Both base kernels - just multiply
                new_kernel = new_kernel*additive_op
            # Make sure still in canonical form - useful mostly for detecting duplicates
            new_kernel = canonical(new_kernel)
        return new_kernel
    elif k.is_operator:
        # This operator is additive - make all operands additive
        new_kernel = k.copy()
        new_kernel.operands = [additive_form(op) for op in k.operands]
        return canonical(new_kernel)
    else:
        #### TODO - Place a check here that the kernel is not a binary or higher operator
        # Base case - return self
        return canonical(k) # Just to make it clear that the output is always canonical

##############################################
#                                            #
#         Miscellaneous functions            #
#                                            #
##############################################

def repr_to_model(string):
    return eval(string)

def remove_duplicates(kernels):
    # This is possible since kernels are now hashable
    return list(set(kernels))
         
def base_kernels(dimensions=1, base_kernel_names='SE'):
    for kernel in base_kernels_without_dimension(base_kernel_names):
        if kernel.is_thunk:
            yield kernel
        else:
            for dimension in range(dimensions):
                k = kernel.copy()
                k.dimension = dimension
                yield k
 
def base_kernels_without_dimension(base_kernel_names):
    for kernel in [SqExpKernel(), \
                   ConstKernel(), \
                   #PureLinKernelFamily(), \
                   #CosineKernelFamily(), \
                   #SpectralKernelFamily(), \
                   #FourierKernelFamily(), \
                   NoiseKernel()]:
        if kernel.id in base_kernel_names.split(','):
            yield kernel 

def add_random_restarts_single_kernel(kernel, n_rand, sd, data_shape):
    '''Returns a list of kernels with random restarts for default values'''
    kernel_list = []
    for dummy in range(n_rand):
        k = kernel.copy()
        k.initialise_params(sd=sd, data_shape=data_shape)
        kernel_list.append(k)
    return kernel_list

def add_random_restarts(kernels, n_rand=1, sd=4, data_shape=None):    
    '''Augments the list to include random restarts of all default value parameters'''
    return [k_rand for kernel in kernels for k_rand in add_random_restarts_single_kernel(kernel, n_rand, sd, data_shape)]

def add_jitter(kernels, sd=0.1):    
    '''Adds random noise to all parameters - empirically observed to help when optimiser gets stuck'''
    for k in kernels:
        k.load_param_vector(k.param_vector + np.random.normal(loc=0., scale=sd, size=k.param_vector.size))
    return kernels      

##############################################
#                                            #
#     Old kernel functions to be revived     #
#                                            #
##############################################  

# #### TODO - this is a code name for the reparametrised centred periodic
# class FourierKernelFamily(BaseKernelFamily):
#     def from_param_vector(self, params):
#         lengthscale, period, output_variance = params
#         return FourierKernel(lengthscale, period, output_variance)
    
#     def num_params(self):
#         return 3
    
#     def pretty_print(self):
#         return colored('FT', self.depth)
    
#     #### FIXME - Caution - magic numbers!
    
#     @staticmethod#### Explanation : This is centered on about 20 periods
#     def default():
#         return FourierKernel(0., -2.0, 0.)
    
#     def __cmp__(self, other):
#         assert isinstance(other, KernelFamily)
#         if cmp(self.__class__, other.__class__):
#             return cmp(self.__class__, other.__class__)
#         return 0
    
#     def depth(self):
#         return 0
    
#     def id_name(self):
#         return 'Fourier'
    
#     @staticmethod    
#     def description():
#         return "Fourier decomposition"

#     @staticmethod    
#     def params_description():
#         return "lengthscale, period"  
    
# class FourierKernel(BaseKernel):
#     def __init__(self, lengthscale, period, output_variance):
#         self.lengthscale = lengthscale
#         self.period = period
#         self.output_variance = output_variance
        
#     def family(self):
#         return FourierKernelFamily()
        
#     def gpml_kernel_expression(self):
#         return '{@covFourier}'
    
#     def english_name(self):
#         return 'Fourier'
    
#     def id_name(self):
#         return 'Fourier'
    
#     def param_vector(self):
#         # order of args matches GPML
#         return np.array([self.lengthscale, self.period, self.output_variance])
        
#     def default_params_replaced(self, sd=1, data_shape=None):
#         '''Overwrites base method, using min period to prevent Nyquist errors'''
#         result = self.param_vector()
#         if result[0] == 0:
#             # Lengthscale is relative to period so this parameter does not need to scale
#             result[0] = np.random.normal(loc=0, scale=sd)
#         if result[1] == -2:
#             #### FIXME - Caution, magic numbers
#             #### Explanation : This is centered on about 25 periods
#             # Min period represents a minimum sensible scale
#             # Scale with data_scale or data range
#             if np.random.rand() < 0.5:
#                 if data_shape['min_period'] is None:
#                     result[1] = np.random.normal(loc=data_shape['input_scale']-2, scale=sd)
#                 else:
#                     result[1] = utils.misc.sample_truncated_normal(loc=data_shape['input_scale']-2, scale=sd, min_value=data_shape['min_period'])
#             else:
#                 if data_shape['min_period'] is None:
#                     result[1] = np.random.normal(loc=np.log(data_shape['input_max']-data_shape['input_min'])-3.2, scale=sd)
#                 else:
#                     result[1] = utils.misc.sample_truncated_normal(loc=np.log(data_shape['input_max']-data_shape['input_min'])-3.2, scale=sd, min_value=data_shape['min_period'])
#         if result[2] == 0:
#             # Set scale factor with output scale or neutrally
#             if np.random.rand() < 0.5:
#                 result[2] = np.random.normal(loc=data_shape['y_sd'], scale=sd)
#             else:
#                 result[2] = np.random.normal(loc=0, scale=sd)
#         return result

#     def copy(self):
#         return FourierKernel(self.lengthscale, self.period, self.output_variance)
    
#     def __repr__(self):
#         return 'FourierKernel(lengthscale=%f, period=%f, output_variance=%f)' % \
#             (self.lengthscale, self.period, self.output_variance)
    
#     def pretty_print(self):
#         return colored('FT(ell=%1.1f, p=%1.1f, sf=%1.1f)' % (self.lengthscale, self.period, self.output_variance),
#                        self.depth)
        
#     def latex_print(self):
#         # return 'PE(\\ell=%1.1f, p=%1.1f, \\sigma=%1.1f)' % (self.lengthscale, self.period, self.output_variance)
#         #return 'PE(p=%1.1f)' % self.period          
#         return 'Fourier'
    
#     def __cmp__(self, other):
#         assert isinstance(other, Kernel)
#         if cmp(self.__class__, other.__class__):
#             return cmp(self.__class__, other.__class__)
#         differences = [self.lengthscale - other.lengthscale, self.period - other.period, self.output_variance - other.output_variance]
#         differences = map(shrink_below_tolerance, differences)
#         return cmp(differences, [0] * len(differences))
        
#     def depth(self):
#         return 0
            
#     def out_of_bounds(self, constraints):
#         return (self.period < constraints['min_period']) or \
#                (self.period > np.log(0.5*(constraints['input_max'] - constraints['input_min']))) # Need to observe more than 2 periods to declare periodicity
        
# class CosineKernelFamily(BaseKernelFamily):
#     def from_param_vector(self, params):
#         period, output_variance = params
#         return CosineKernel(period, output_variance)
    
#     def num_params(self):
#         return 2
    
#     def pretty_print(self):
#         return colored('Cos', self.depth)
    
#     # FIXME - Caution - magic numbers!
    
#     @staticmethod#### Explanation : This is centered on about 20 periods
#     def default():
#         return CosineKernel(-2.0, 0.)
    
#     def __cmp__(self, other):
#         assert isinstance(other, KernelFamily)
#         if cmp(self.__class__, other.__class__):
#             return cmp(self.__class__, other.__class__)
#         return 0
    
#     def depth(self):
#         return 0
    
#     def id_name(self):
#         return 'Cos'
    
#     @staticmethod    
#     def description():
#         return "Cosine"

#     @staticmethod    
#     def params_description():
#         return "period"  
    
# class CosineKernel(BaseKernel):
#     def __init__(self, period, output_variance):
#         self.period = period
#         self.output_variance = output_variance
        
#     def family(self):
#         return CosineKernelFamily()
        
#     def gpml_kernel_expression(self):
#         return '{@covCos}'
    
#     def english_name(self):
#         return 'Cosine'
    
#     def id_name(self):
#         return 'Cos'
    
#     def param_vector(self):
#         # order of args matches GPML
#         return np.array([self.period, self.output_variance])
        
#     def default_params_replaced(self, sd=1, data_shape=None):
#         '''Overwrites base method, using min period to prevent Nyquist errors'''
#         result = self.param_vector()
#         if result[0] == -2:
#             if np.random.rand() < 0.5:
#                 if data_shape['min_period'] is None:
#                     result[0] = np.random.normal(loc=data_shape['input_scale']-2, scale=sd)
#                 else:
#                     result[0] = utils.misc.sample_truncated_normal(loc=data_shape['input_scale']-2, scale=sd, min_value=data_shape['min_period'])
#             else:
#                 if data_shape['min_period'] is None:
#                     result[0] = np.random.normal(loc=np.log(data_shape['input_max']-data_shape['input_min'])-3.2, scale=sd)
#                 else:
#                     result[0] = utils.misc.sample_truncated_normal(loc=np.log(data_shape['input_max']-data_shape['input_min'])-3.2, scale=sd, min_value=data_shape['min_period'])
#         if result[1] == 0:
#             # Set scale factor with output scale or neutrally
#             if np.random.rand() < 0.5:
#                 result[1] = np.random.normal(loc=data_shape['y_sd'], scale=sd)
#             else:
#                 result[1] = np.random.normal(loc=0, scale=sd)
#         return result

#     def copy(self):
#         return CosineKernel(self.period, self.output_variance)
    
#     def __repr__(self):
#         return 'CosineKernel(period=%f, output_variance=%f)' % \
#             (self.period, self.output_variance)
    
#     def pretty_print(self):
#         return colored('Cos(p=%1.1f, sf=%1.1f)' % (self.period, self.output_variance),
#                        self.depth)
        
#     def latex_print(self):    
#         return 'Cos'
    
#     def __cmp__(self, other):
#         assert isinstance(other, Kernel)
#         if cmp(self.__class__, other.__class__):
#             return cmp(self.__class__, other.__class__)
#         differences = [self.period - other.period, self.output_variance - other.output_variance]
#         differences = map(shrink_below_tolerance, differences)
#         return cmp(differences, [0] * len(differences))
        
#     def depth(self):
#         return 0
            
#     def out_of_bounds(self, constraints):
#         return (self.period < constraints['min_period']) or \
#                (self.period > np.log(0.5*(constraints['input_max'] - constraints['input_min']))) # Need to observe more than 2 periods to declare periodicity
        
# class SpectralKernelFamily(BaseKernelFamily):
#     def from_param_vector(self, params):
#         lengthscale, output_variance, period = params
#         return SpectralKernel(lengthscale, period, output_variance)
    
#     def num_params(self):
#         return 3
    
#     def pretty_print(self):
#         return colored('SP', self.depth)
    
#     # FIXME - Caution - magic numbers!
    
#     @staticmethod#### Explanation : This is centered on about 20 periods
#     def default():
#         return SpectralKernel(0., -2.0, 0.)
    
#     def __cmp__(self, other):
#         assert isinstance(other, KernelFamily)
#         if cmp(self.__class__, other.__class__):
#             return cmp(self.__class__, other.__class__)
#         return 0
    
#     def depth(self):
#         return 0
    
#     def id_name(self):
#         return 'SP'
    
#     @staticmethod    
#     def description():
#         return "Spectral"

#     @staticmethod    
#     def params_description():
#         return "lengthscale, period"  
    
# class SpectralKernel(BaseKernel):
#     def __init__(self, lengthscale, period, output_variance):
#         self.lengthscale = lengthscale
#         self.period = period
#         self.output_variance = output_variance
        
#     def family(self):
#         return SpectralKernelFamily()
        
#     def gpml_kernel_expression(self):
#         return '{@covProd, {@covSEiso, @covCosUnit}}'
    
#     def english_name(self):
#         return 'Spectral'
    
#     def id_name(self):
#         return 'SP'
    
#     def param_vector(self):
#         # order of args matches GPML
#         return np.array([self.lengthscale, self.output_variance, self.period])
        
#     def default_params_replaced(self, sd=1, data_shape=None):
#         '''Overwrites base method, using min period to prevent Nyquist errors'''
#         result = self.param_vector()
#         if result[0] == 0:
#             # Set lengthscale with input scale or neutrally
#             if np.random.rand() < 0.5:
#                 result[0] = np.random.normal(loc=data_shape['input_scale'], scale=sd)
#             else:
#                 # Long lengthscale ~ infty = neutral
#                 result[0] = np.random.normal(loc=np.log(2*(data_shape['input_max']-data_shape['input_min'])), scale=sd)
#         if result[2] == -2:
#             #### FIXME - Caution, magic numbers
#             #### Explanation : This is centered on about 25 periods
#             # Min period represents a minimum sensible scale
#             # Scale with data_scale or data range
#             if np.random.rand() < 0.66:
#                 if np.random.rand() < 0.5:
#                     if data_shape['min_period'] is None:
#                         result[2] = np.random.normal(loc=data_shape['input_scale']-2, scale=sd)
#                     else:
#                         result[2] = utils.misc.sample_truncated_normal(loc=data_shape['input_scale']-2, scale=sd, min_value=data_shape['min_period'])
#                 else:
#                     if data_shape['min_period'] is None:
#                         result[2] = np.random.normal(loc=np.log(data_shape['input_max']-data_shape['input_min'])-3.2, scale=sd)
#                     else:
#                         result[2] = utils.misc.sample_truncated_normal(loc=np.log(data_shape['input_max']-data_shape['input_min'])-3.2, scale=sd, min_value=data_shape['min_period'])
#             else:
#                 # Spectral kernel can also approximate SE with long period
#                 result[2] = np.log(data_shape['input_max']-data_shape['input_min'])
#         if result[1] == 0:
#             # Set scale factor with output scale or neutrally
#             if np.random.rand() < 0.5:
#                 result[1] = np.random.normal(loc=data_shape['y_sd'], scale=sd)
#             else:
#                 result[1] = np.random.normal(loc=0, scale=sd)
#         return result

#     def copy(self):
#         return SpectralKernel(self.lengthscale, self.period, self.output_variance)
    
#     def __repr__(self):
#         return 'SpectralKernel(lengthscale=%f, period=%f, output_variance=%f)' % \
#             (self.lengthscale, self.period, self.output_variance)
    
#     def pretty_print(self):
#         return colored('SP(ell=%1.1f, p=%1.1f, sf=%1.1f)' % (self.lengthscale, self.period, self.output_variance),
#                        self.depth)
        
#     def latex_print(self):         
#         return 'Spec'
    
#     def __cmp__(self, other):
#         assert isinstance(other, Kernel)
#         if cmp(self.__class__, other.__class__):
#             return cmp(self.__class__, other.__class__)
#         differences = [self.lengthscale - other.lengthscale, self.period - other.period, self.output_variance - other.output_variance]
#         differences = map(shrink_below_tolerance, differences)
#         return cmp(differences, [0] * len(differences))
        
#     def depth(self):
#         return 0
            
#     def out_of_bounds(self, constraints):
#         return (self.period < constraints['min_period']) or \
#                (self.lengthscale < constraints['min_lengthscale'])

# class PureLinKernelFamily(BaseKernelFamily):
#     def from_param_vector(self, params):
#         lengthscale, location = params
#         return PureLinKernel(lengthscale=lengthscale, location=location)
    
#     def num_params(self):
#         return 2
    
#     def pretty_print(self):
#         return colored('PLN', self.depth)
    
#     @staticmethod
#     def default():
#         return PureLinKernel(0., 0.)
    
#     def __cmp__(self, other):
#         assert isinstance(other, KernelFamily)
#         if cmp(self.__class__, other.__class__):
#             return cmp(self.__class__, other.__class__)
#         return 0
    
#     def depth(self):
#         return 0
    
#     def id_name(self):
#         return 'PureLin'

#     @staticmethod    
#     def description():
#         return "Pure Linear"

#     @staticmethod    
#     def params_description():
#         return "Lengthscale (inverse scale) and location"
    
# class PureLinKernel(BaseKernel):
#     #### FIXME - lengthscale is actually an inverse scale
#     #### Also - lengthscale is a silly name even if it is used by GPML
#     def __init__(self, lengthscale=0, location=0):
#         self.lengthscale = lengthscale
#         self.location = location
        
#     def family(self):
#         return PureLinKernelFamily()
        
#     def gpml_kernel_expression(self):
#         return '{@covLINscaleshift}'
    
#     def english_name(self):
#         return 'PLN'
    
#     def id_name(self):
#         return 'PureLin'
    
#     def param_vector(self):
#         # order of args matches GPML
#         return np.array([self.lengthscale, self.location])
        
#     def default_params_replaced(self, sd=1, data_shape=None):
#         result = self.param_vector()
#         if result[0] == 0:
#             # Lengthscale scales inversely with ratio of y std and x std (gradient = delta y / delta x)
#             # Or with gradient or a neutral value
#             rand = np.random.rand()
#             if rand < 1.0/3:
#                 result[0] = np.random.normal(loc=-(data_shape['y_sd'] - data_shape['input_scale']), scale=sd)
#             elif rand < 2.0/3:
#                 result[0] = np.random.normal(loc=-np.log(np.abs((data_shape['output_max']-data_shape['output_min'])/(data_shape['input_max']-data_shape['input_min']))), scale=sd)
#             else:
#                 result[0] = np.random.normal(loc=0, scale=sd)
#         if result[1] == 0:
#             # Uniform over 3 x data range
#             result[1] = np.random.uniform(low=2*data_shape['input_min']-data_shape['input_max'], high=2*data_shape['input_max']-data_shape['input_min'])
#         return result
        
#     #def effective_params(self):
#     #    return 2

#     def copy(self):
#         return PureLinKernel(lengthscale=self.lengthscale, location=self.location)
    
#     def __repr__(self):
#         return 'PureLinKernel(lengthscale=%f, location=%f)' % \
#             (self.lengthscale, self.location)
    
#     def pretty_print(self):
#         return colored('PLN(ell=%1.1f, loc=%1.1f)' % (self.lengthscale, self.location),
#                        self.depth)
        
#     def latex_print(self):
#         return 'PureLin'           
    
#     def __cmp__(self, other):
#         assert isinstance(other, Kernel)
#         if cmp(self.__class__, other.__class__):
#             return cmp(self.__class__, other.__class__)
#         differences = [self.lengthscale - other.lengthscale, self.location - other.location]
#         differences = map(shrink_below_tolerance, differences)
#         return cmp(differences, [0] * len(differences))
        
#     def depth(self):
#         return 0  
        
#     @property    
#     def stationary(self):
#         return False
               
# class ChangePointTanhKernelFamily(KernelOperatorFamily):
#     def __init__(self, operands):
#         self.operands = operands
#         assert len(operands) == 2
        
#     def from_param_vector(self, params):
#         location = params[0]
#         steepness = params[1]
#         start = 2
#         ops = []
#         for e in self.operands:
#             end = start + e.num_params()
#             ops.append(e.from_param_vector(params[start:end]))
#             start = end
#         return ChangePointTanhKernel(location, steepness, ops)
    
#     def num_params(self):
#         return 2 + sum([e.num_params() for e in self.operands])
    
#     def pretty_print(self):        
#         return colored('CPT(', self.depth) + \
#             self.operands[0].pretty_print() + \
#             colored(', ', self.depth) + \
#             self.operands[1].pretty_print() + \
#             colored(')', self.depth)

#     def default(self):
#         return ChangePointTanhKernel(0., 0., [op.default() for op in self.operands])
    
#     def __cmp__(self, other):
#         assert isinstance(other, KernelFamily)
#         if cmp(self.__class__, other.__class__):
#             return cmp(self.__class__, other.__class__)
#         return cmp(self.operands, other.operands)
    
#     def depth(self):
#         return max([op.depth for op in self.operands]) + 1

# class ChangePointTanhKernel(KernelOperator):
#     def __init__(self, location, steepness, operands):
#         self.location = location
#         self.steepness = steepness
#         self.operands = operands
        
#     def family(self):
#         return ChangePointTanhKernelFamily([e.family() for e in self.operands])
        
#     def pretty_print(self): 
#         return colored('CPT(loc=%1.1f, steep=%1.1f, ' % (self.location, self.steepness), self.depth) + \
#             self.operands[0].pretty_print() + \
#             colored(', ', self.depth) + \
#             self.operands[1].pretty_print() + \
#             colored(')', self.depth)
            
#     def latex_print(self):
#         return 'CPT\\left( ' + ' , '.join([e.latex_print() for e in self.operands]) + ' \\right)'            
            
#     def __repr__(self):
#         return 'ChangePointTanhKernel(location=%f, steepness=%f, operands=%s)' % \
#             (self.location, self.steepness, '[ ' + ', '.join([o.__repr__() for o in self.operands]) + ' ]')                
    
#     def gpml_kernel_expression(self):
#         return '{@covChangePointTanh, {%s}}' % ', '.join(e.gpml_kernel_expression() for e in self.operands)
    
#     def copy(self):
#         return ChangePointTanhKernel(self.location, self.steepness, [e.copy() for e in self.operands])

#     def param_vector(self):
#         return np.concatenate([np.array([self.location, self.steepness])] + [e.param_vector() for e in self.operands])
        
#     def effective_params(self):
#         return 2 + sum([o.effective_params() for o in self.operands])
        
#     def default_params_replaced(self, sd=1, data_shape=None):
#         '''Returns the parameter vector with any default values replaced with random Gaussian'''
#         result = self.param_vector()[:2]
#         if result[0] == 0:
#             # Location uniform in data range
#             result[0] = np.random.uniform(data_shape['input_min'], data_shape['input_max'])
#         if result[1] == 0:
#             #### FIXME - Caution, magic numbers
#             # Set steepness with inverse input scale
#             result[1] = np.random.normal(loc=3.3-np.log((data_shape['input_max'] - data_shape['input_min'])), scale=1)
#         return np.concatenate([result] + [o.default_params_replaced(sd=sd, data_shape=data_shape) for o in self.operands])
    
#     def __cmp__(self, other):
#         assert isinstance(other, Kernel)
#         if cmp(self.__class__, other.__class__):
#             return cmp(self.__class__, other.__class__)
#         return cmp((self.location, self.steepness, self.operands),
#                    (other.location, other.steepness, other.operands))
    
#     def depth(self):
#         return max([op.depth for op in self.operands]) + 1
            
#     def out_of_bounds(self, constraints):
#         return (self.location < constraints['input_min']) or \
#                (self.location > constraints['input_max']) or \
#                (self.steepness < -np.log((constraints['input_max'] -constraints['input_min'])) + 2.3) or \
#                (any([o.out_of_bounds(constraints) for o in self.operands])) 
               
# class ChangeBurstTanhKernelFamily(KernelOperatorFamily):
#     def __init__(self, operands):
#         self.operands = operands
#         assert len(operands) == 2
        
#     def from_param_vector(self, params):
#         location = params[0]
#         steepness = params[1]
#         width = params[2]
#         start = 3
#         ops = []
#         for e in self.operands:
#             end = start + e.num_params()
#             ops.append(e.from_param_vector(params[start:end]))
#             start = end
#         return ChangeBurstTanhKernel(location, steepness, width, ops)
    
#     def num_params(self):
#         return 3 + sum([e.num_params() for e in self.operands])
    
#     def pretty_print(self):        
#         return colored('CBT(', self.depth) + \
#             self.operands[0].pretty_print() + \
#             colored(', ', self.depth) + \
#             self.operands[1].pretty_print() + \
#             colored(')', self.depth)
    
#     def default(self):
#         return ChangeBurstTanhKernel(0., 0., 0., [op.default() for op in self.operands])
    
#     def __cmp__(self, other):
#         assert isinstance(other, KernelFamily)
#         if cmp(self.__class__, other.__class__):
#             return cmp(self.__class__, other.__class__)
#         return cmp(self.operands, other.operands)
    
#     def depth(self):
#         return max([op.depth for op in self.operands]) + 1

# class ChangeBurstTanhKernel(KernelOperator):
#     def __init__(self, location, steepness, width, operands):
#         self.location = location
#         self.steepness = steepness
#         self.width = width
#         self.operands = operands
        
#     def family(self):
#         return ChangeBurstTanhKernelFamily([e.family() for e in self.operands])
        
#     def pretty_print(self): 
#         return colored('CBT(loc=%1.1f, steep=%1.1f, width=%1.1f, ' % (self.location, self.steepness, self.width), self.depth) + \
#             self.operands[0].pretty_print() + \
#             colored(', ', self.depth) + \
#             self.operands[1].pretty_print() + \
#             colored(')', self.depth)
            
#     def latex_print(self):
#         return 'CBT\\left( ' + ' , '.join([e.latex_print() for e in self.operands]) + ' \\right)'            
            
#     def __repr__(self):
#         return 'ChangeBurstTanhKernel(location=%f, steepness=%f, width=%f, operands=%s)' % \
#             (self.location, self.steepness, self.width, '[ ' + ', '.join([o.__repr__() for o in self.operands]) + ' ]')                
    
#     def gpml_kernel_expression(self):
#         return '{@covChangeBurstTanh, {%s}}' % ', '.join(e.gpml_kernel_expression() for e in self.operands)
    
#     def copy(self):
#         return ChangeBurstTanhKernel(self.location, self.steepness, self.width, [e.copy() for e in self.operands])

#     def param_vector(self):
#         return np.concatenate([np.array([self.location, self.steepness, self.width])] + [e.param_vector() for e in self.operands])
        
#     def effective_params(self):
#         return 3 + sum([o.effective_params() for o in self.operands])
        
#     def default_params_replaced(self, sd=1, data_shape=None):
#         '''Returns the parameter vector with any default values replaced with random Gaussian'''
#         result = self.param_vector()[:3]
#         if result[0] == 0:
#             # Location uniform in data range
#             result[0] = np.random.uniform(data_shape['input_min'], data_shape['input_max'])
#         if result[1] == 0:
#             #### FIXME - Caution, magic numbers
#             # Set steepness with inverse input scale
#             result[1] = np.random.normal(loc=3.3-np.log((data_shape['input_max'] - data_shape['input_min'])), scale=1)
#         if result[2] == 0:
#             # Set width with input scale - but expecting small widths
#             #### FIXME - Caution, magic numbers
#             result[2] = np.random.normal(loc=np.log(0.1*(data_shape['input_max'] - data_shape['input_min'])), scale=1)
#         return np.concatenate([result] + [o.default_params_replaced(sd=sd, data_shape=data_shape) for o in self.operands])
    
#     def __cmp__(self, other):
#         assert isinstance(other, Kernel)
#         if cmp(self.__class__, other.__class__):
#             return cmp(self.__class__, other.__class__)
#         return cmp((self.location, self.steepness, self.width, self.operands),
#                    (other.location, other.steepness, self.width, other.operands))
    
#     def depth(self):
#         return max([op.depth for op in self.operands]) + 1
            
#     def out_of_bounds(self, constraints):
#         return (self.location - np.exp(self.width)/2 < constraints['input_min'] + 0.05 * (constraints['input_max'] -constraints['input_min'])) or \
#                (self.location + np.exp(self.width)/2 > constraints['input_max'] - 0.05 * (constraints['input_max'] -constraints['input_min'])) or \
#                (self.width > np.log(0.25*(constraints['input_max'] - constraints['input_min']))) or \
#                (self.steepness < -np.log((constraints['input_max'] -constraints['input_min'])) + 2.3) or \
#                (any([o.out_of_bounds(constraints) for o in self.operands])) 
