"""
Defines wrappers for mean, covariance and likelihood functions.
Also defines kernel manipulation routines.

Created November 2012

@authors: James Robert Lloyd (jrl44@cam.ac.uk)
          David Duvenaud (dkd23@cam.ac.uk)
          Roger Grosse (rgrosse@mit.edu)
"""

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


class FunctionWrapper:
    """Base class for mean / kernel / likelihood functions."""

    # Properties - these are read-only and immutable

    # e.g. @covSEiso
    @property
    def gpml_function(self): raise RuntimeError('This property must be overriden')
       
    @property    
    def is_operator(self): return False
       
    @property    
    def is_abelian(self):
        if not self.is_operator:
            return None # Not applicable
        else:
            raise RuntimeError('Operators must override this property')

    # Identification used internally
    @property
    def id(self): raise RuntimeError('This property must be overriden')
    
    # Parameters in the order defined by GPML
    @property
    def param_vector(self): raise RuntimeError('This property must be overriden')
    
    @property
    def num_params(self): return len(self.param_vector)

    # Used by information criteria that count optimised parameters
    @property
    def effective_params(self):
        if not self.is_operator:
            '''This is true of all base functions, hence definition here'''  
            return len(self.param_vector)
        else:
            raise RuntimeError('Operators must override this property')

    # LaTeX representation of function
    @property
    def latex(self): raise RuntimeError('This property must be overriden') 

    # Depth up the expression tree - leaves = 0
    @property
    def depth(self):
        if not self.is_operator:
            return 0 
        else:
            raise RuntimeError('Operators must override this property')

    # String representation of function without any parameters
    @property
    def syntax(self): raise RuntimeError('This property must be overriden') 

    # Hidden methods

    def __repr__(self): return 'FunctionWrapper()'
            
    # NOTE : This hash is defined for convenience but must be used with caution since this class is mutable
    def __hash__(self): return hash(self.__repr__())

    def __cmp__(self, other):
        if cmp(self.__class__, other.__class__):
            return cmp(self.__class__, other.__class__)
        else:
            # QUESTION : Is comparing strings very slow?
            # If so this should be overidden for speed
            return cmp(self.__repr__(), other.__repr__())

    # Methods returning objects of the same type as self

    def copy(self): raise RuntimeError('This method must be overriden')

    # Returns a function with any syntactic redundancy removed (e.g. idempotency, zero elements...)
    def simplified(self): return self.copy()

    # Returns the canonical form of an object
    def canonical(self): return self.copy()

    def additive_form(self): return self.copy()

    # Returns a list of summands
    def break_into_summands(self): return [self.copy()]

    # Methods returning different types

    def initialise_params(self, sd=1, data_shape=None): raise RuntimeError('This method must be overriden')
    
    def pretty_print(self): return RuntimeError('This method must be overriden')
        
    def out_of_bounds(self, constraints): return False

    def load_param_vector(self, params): return RuntimeError('This method must be overriden')


class MeanFunction(FunctionWrapper):
    """Base mean function class with default properties and methods."""
    
    # Syntactic sugar e.g. f1 + f2
    # Returns copies of involved functions - ensured by canonical operation
    def __add__(self, other):
        assert isinstance(other, MeanFunction)
        if isinstance(other, SumFunction):
            if isinstance(self, SumFunction):
                new_f = self.copy()
                new_f.operands = self.operands + other.operands
                return new_f.canonical()
            else:
                new_f = self.copy()
                new_f.operands = [self] + other.operands
                return new_f.canonical()
        else:
            return SumFunction([self, other]).canonical()
    
    # Syntactic sugar e.g. f1 * f2
    # Returns copies of involved functions - ensured by canonical operation
    def __mul__(self, other):
        assert isinstance(other, MeanFunction)
        if isinstance(other, ProductFunction):
            if isinstance(self, ProductFunction):
                new_f = self.copy()
                new_f.operands = self.operands + other.operands
                return new_f.canonical()
            else:
                new_f = self.copy()
                new_f.operands = [self] + other.operands
                return new_f.canonical()
        else:
            return ProductFunction([self, other]).canonical()

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


class Kernel(FunctionWrapper):
    """Base kernel class with default properties and methods"""
    
    # Syntactic sugar e.g. k1 + k2
    # Returns copies of involved functions - ensured by canonical operation
    def __add__(self, other):
        assert isinstance(other, Kernel)
        if isinstance(other, SumKernel):
            if isinstance(self, SumKernel):
                new_f = self.copy()
                new_f.operands = self.operands + other.operands
                return new_f.canonical()
            else:
                new_f = self.copy()
                new_f.operands = [self] + other.operands
                return new_f.canonical()
        else:
            return SumKernel([self, other]).canonical()
    
    # Syntactic sugar e.g. k1 * k2
    # Returns copies of involved functions - ensured by canonical operation
    def __mul__(self, other):
        assert isinstance(other, Kernel)
        if isinstance(other, ProductKernel):
            if isinstance(self, ProductKernel):
                new_f = self.copy()
                new_f.operands = self.operands + other.operands
                return new_f.canonical()
            else:
                new_f = self.copy()
                new_f.operands = [self] + other.operands
                return new_f.canonical()
        else:
            return ProductKernel([self, other]).canonical()

    # Properties
       
    @property    
    def is_stationary(self): return True
       
    @property    
    def sf(self): raise RuntimeError('This must be overriden')
       
    #### TODO - this only happens when a kernel is dimensionless?
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

    def simplified(self):
        k = self.copy()
        k_prev = None
        while not k_prev == k:
            k_prev = k.copy()
            k = k.collapse_additive_idempotency()
            k = k.collapse_multiplicative_idempotency()
            k = k.collapse_multiplicative_identity()
            k = k.collapse_multiplicative_zero()
            k = k.canonical()
        return k

    def __repr__(self): return 'Kernel()'

    def canonical(self):
        '''Sorts a kernel tree into a canonical form.'''
        #### TODO - This can be abstracted to mean functions and likelihood functions by defining a None wrapper
        if not self.is_operator:
            return self.copy()
        else:
            new_ops = []
            for op in self.operands:
                op_canon = op.canonical()
                if isinstance(op_canon, self.__class__) and (self.arity=='n'):
                    new_ops += op_canon.operands
                elif not isinstance(op_canon, NoneKernel):
                    new_ops.append(op_canon)
            if len(new_ops) == 0:
                return NoneKernel()
            elif len(new_ops) == 1:
                return new_ops[0]
            else:
                canon = self.copy()
                if self.is_abelian:
                    canon.operands = sorted(new_ops)
                else:
                    canon.operands = new_ops
                return canon

    def additive_form(self):
        '''
        Converts a kernel into a sum of products and with changepoints percolating to the top
        Output is always in canonical form
        '''
        #### TODO - currently implemented for a subset of changepoint operators - to be extended or operators to be abstracted
        k = self.canonical()
        if isinstance(k, ProductKernel):
            # Convert operands into additive form
            additive_ops = sorted([op.additive_form() for op in k.operands])
            # Initialise the new kernel
            new_kernel = additive_ops[0]
            # Build up the product, iterating over the other components
            for additive_op in additive_ops[1:]:
                if isinstance(new_kernel, ChangePointKernel) or isinstance(new_kernel, ChangeWindowKernel):
                    # Changepoints take priority - nest the products within this operator
                    new_kernel.operands = [(op*additive_op.copy()).canonical().additive_form() for op in new_kernel.operands]
                elif isinstance(additive_op, ChangePointKernel) or isinstance(additive_op, ChangeWindowKernel):
                    # Nest within the next operator
                    old_kernel = new_kernel.copy()
                    new_kernel = additive_op
                    new_kernel.operands = [(op*old_kernel.copy()).canonical().additive_form() for op in new_kernel.operands]
                elif isinstance(new_kernel, SumKernel):
                    # Nest the products within this sum
                    new_kernel.operands = [(op*additive_op.copy()).canonical().additive_form() for op in new_kernel.operands]
                elif isinstance(additive_op, SumKernel):
                    # Nest within the next operator
                    old_kernel = new_kernel.copy()
                    new_kernel = additive_op
                    new_kernel.operands = [(op*old_kernel.copy()).canonical().additive_form() for op in new_kernel.operands]
                else:
                    # Both base kernels - just multiply
                    new_kernel = new_kernel*additive_op
                # Make sure still in canonical form - useful mostly for detecting duplicates
                new_kernel = new_kernel.canonical()
            return new_kernel
        elif k.is_operator:
            # This operator is additive - make all operands additive
            new_kernel = k.copy()
            new_kernel.operands = [op.additive_form() for op in k.operands]
            return new_kernel.canonical()
        else:
            #### TODO - Place a check here that the kernel is not a binary or higher operator
            # Base case - return self
            return k.canonical() # Just to make it clear that the output is always canonical

    #### TODO - this can be abstracted to function wrapper level
    def break_into_summands(self):
        '''Takes a kernel, expands it into a polynomial, and breaks terms up into a list.
        
        Mutually Recursive with distribute_products_k().
        Always returns a list.
        '''    
        k = self.copy()
        # First, recursively distribute all products within the kernel.
        k_dist = k.distribute_products()
        
        if isinstance(k_dist, SumKernel):
            # Break the summands into a list of kernels.
            return list(k_dist.operands)
        else:
            return [k_dist]

    def distribute_products(self):
        """Distributes products to get a polynomial.
        
        Mutually recursive with break_kernel_into_summands().
        Always returns a sumkernel.
        """
        k = self.copy()
        if isinstance(k, ProductKernel):
            # Recursively distribute each of the terms to be multiplied.
            distributed_ops = [op.break_into_summands() for op in k.operands]
            
            # Now produce a sum of all combinations of terms in the products. Itertools is awesome.
            new_prod_ks = [ProductKernel( operands=prod ) for prod in itertools.product(*distributed_ops)]
            return SumKernel(operands=new_prod_ks)
        
        elif isinstance(k, SumKernel):
            # Recursively distribute each the operands to be summed, then combine them back into a new SumKernel.
            return SumKernel([subop for op in k.operands for subop in op.break_into_summands()])
        elif k.is_operator:
            if k.arity == 2:
                summands = []
                operands_list = [[op, ZeroKernel()] for op in k.operands[0].break_into_summands()]
                for ops in operands_list:
                    k_new = k.copy()
                    k_new.operands = ops
                    summands.append(k_new)
                operands_list = [[ZeroKernel(), op] for op in k.operands[1].break_into_summands()]
                for ops in operands_list:
                    k_new = k.copy()
                    k_new.operands = ops
                    summands.append(k_new)
                return SumKernel(operands=summands)
            else:
                raise RuntimeError('Not sure how to distribute products of this operator')
        else:
            # Base case: A kernel that's just, like, a kernel, man.
            return k

    def collapse_additive_idempotency(self):
        # TODO - abstract this behaviour
        k = self.copy()
        k = k.canonical()
        if not k.is_operator:
            return k
        elif isinstance(k, SumKernel):
            ops = [o.collapse_additive_idempotency() for o in k.operands]
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
            return k.canonical()
        else:
            new_ops = []
            for o in k.operands:
                new_ops.append(o.collapse_additive_idempotency())
            k.operands = new_ops
            return k

    def collapse_multiplicative_idempotency(self):
        # TODO - abstract this behaviour
        k = self.copy()
        k = k.canonical()
        if not k.is_operator:
            return k
        elif isinstance(k, ProductKernel):
            ops = [o.collapse_multiplicative_idempotency() for o in k.operands]
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
            return k.canonical()
        else:
            new_ops = []
            for o in k.operands:
                new_ops.append(o.collapse_multiplicative_idempotency())
            k.operands = new_ops
            return k

    def collapse_multiplicative_zero(self):
        # TODO - abstract this behaviour
        k = self.copy()
        k = k.canonical()
        if not k.is_operator:
            return k
        elif isinstance(k, ProductKernel):
            ops = [o.collapse_multiplicative_zero() for o in k.operands]
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
            return k.canonical()
        else:
            new_ops = []
            for o in k.operands:
                new_ops.append(o.collapse_multiplicative_zero())
            k.operands = new_ops
            return k

    def collapse_multiplicative_identity(self):
        # TODO - abstract this behaviour
        k = self.copy()
        k = k.canonical()
        if not k.is_operator:
            return k
        elif isinstance(k, ProductKernel):
            ops = [o.collapse_multiplicative_identity() for o in k.operands]
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
            return k.canonical()
        else:
            new_ops = []
            for o in k.operands:
                new_ops.append(o.collapse_multiplicative_identity())
            k.operands = new_ops
            return k

    def cp_structure(self):
        # Replaces most things with constants - useful for understanding structure of changepoints
        k = self.copy()
        if isinstance(k, ZeroKernel) or isinstance(k, NoneKernel): # TODO - can this be abstracted?
            return k
        elif not k.is_operator:
            return ConstKernel(sf=0)
        else:
            k.operands = [op.cp_structure() for op in k.operands]
            return k


class Likelihood(FunctionWrapper):
    """Base likelihood function class with default properties and methods"""
    
    # Syntactic sugar e.g. l1 + l2
    # Returns copies of involved functions - ensured by canonical operation
    def __add__(self, other):
        assert isinstance(other, Likelihood)
        if isinstance(other, SumLikelihood):
            if isinstance(self, SumLikelihood):
                new_f = self.copy()
                new_f.operands = self.operands + other.operands
                return new_f.canonical()
            else:
                new_f = self.copy()
                new_f.operands = [self] + other.operands
                return new_f.canonical()
        else:
            return SumLikelihood([self, other]).canonical()
    
    # Syntactic sugar e.g. l1 * l2
    # Returns copies of involved functions - ensured by canonical operation
    def __mul__(self, other):
        assert isinstance(other, Likelihood)
        if isinstance(other, ProductLikelihood):
            if isinstance(self, ProductLikelihood):
                new_f = self.copy()
                new_f.operands = self.operands + other.operands
                return new_f.canonical()
            else:
                new_f = self.copy()
                new_f.operands = [self] + other.operands
                return new_f.canonical()
        else:
            return ProductLikelihood([self, other]).canonical()

    # Properties

    @property
    def gpml_inference_method(self): return '@infExact'

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


class GPModel:
    """Model class - keeps track of a mean function, kernel, likelihood function,
       and optionally a score."""

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
        return 'GPModel(mean=%s, kernel=%s, likelihood=%s, nll=%s, ndata=%s)' % \
               (self.mean.__repr__(), self.kernel.__repr__(), self.likelihood.__repr__(), self.nll, self.ndata)

    def __cmp__(self, other):
        if cmp(self.__class__, other.__class__):
            return cmp(self.__class__, other.__class__)
        else:
            return cmp(self.__repr__(), other.__repr__())

    def copy(self):
        m = self.mean.copy() if not self.mean is None else None
        k = self.kernel.copy() if not self.kernel is None else None
        l = self.likelihood.copy() if not self.likelihood is None else None
        return GPModel(mean=m, kernel=k, likelihood=l, nll=self.nll, ndata=self.ndata)

    def pretty_print(self):
        return 'GPModel(mean=%s, kernel=%s, likelihood=%s)' % \
                (self.mean.pretty_print(), self.kernel.pretty_print(), self.likelihood.pretty_print())
        
    def out_of_bounds(self, constraints):
        return any([self.mean.out_of_bounds(constraints), \
                    self.kernel.out_of_bounds(constraints), \
                    self.likelihood.out_of_bounds(constraints)])

    @property
    def bic(self):
        return 2 * self.nll + self.kernel.effective_params * np.log(self.ndata)

    @property
    def aic(self):
        return 2 * self.nll + self.kernel.effective_params * 2

    @property
    def pl2(self):
        return self.nll / self.ndata + self.kernel.effective_params / (2 * self.ndata)

    @staticmethod
    def score(self, criterion='bic'):
        return {'bic': self.bic,
                'aic': self.aic,
                'pl2': self.pl2,
                'nll': self.nll
                }[criterion.lower()]
                
    @staticmethod
    def from_printed_outputs(nll=None, ndata=None, noise=None, mean=None, kernel=None, likelihood=None):
        return GPModel(mean=mean, kernel=kernel, likelihood=likelihood, nll=nll, ndata=ndata)

    @staticmethod 
    def from_matlab_output(output, model, ndata):
        model.mean.load_param_vector(output.mean_hypers)
        model.kernel.load_param_vector(output.kernel_hypers)
        model.likelihood.load_param_vector(output.lik_hypers)
        return GPModel(mean=model.mean, kernel=model.kernel, likelihood=model.likelihood, nll=output.nll, ndata=ndata) 

    def simplified(self):
        simple = self.copy()
        simple.mean = simple.mean.simplified()
        simple.kernel = simple.kernel.simplified()
        simple.likelihood = simple.likelihood.simplified()
        return simple

    def canonical(self):
        canon = self.copy()
        canon.mean = canon.mean.canonical()
        canon.kernel = canon.kernel.canonical()
        canon.likelihood = canon.likelihood.canonical()
        return canon

    def additive_form(self):
        # This will need to be more cunning when using compound mean and lik
        additive = self.copy()
        additive.kernel = additive.kernel.additive_form()
        return additive

    def break_into_summands(self):
        mean_list = self.mean.break_into_summands()
        kernel_list = self.kernel.break_into_summands()
        likelihood_list = self.likelihood.break_into_summands()
        model_list = []
        for a_mean in mean_list:
            model_list.append(GPModel(mean=a_mean, kernel=ZeroKernel(), likelihood=LikGauss(sf=-np.Inf)))
        for a_kernel in kernel_list:
            model_list.append(GPModel(mean=MeanZero(), kernel=a_kernel, likelihood=LikGauss(sf=-np.Inf)))
        for a_likelihood in likelihood_list:
            model_list.append(GPModel(mean=MeanZero(), kernel=ZeroKernel(), likelihood=a_likelihood))
        null_model = GPModel(ean=MeanZero(), kernel=ZeroKernel(), likelihood=LikGauss(sf=-np.Inf))
        model_list = [model for model in model_list if not model == null_model]
        return model_list

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
                self.c = np.random.normal(loc=data_shape['y_mean'], scale=sd*np.exp(data_shape['y_sd']))
            else:
                self.c = np.random.normal(loc=0, scale=sd*np.exp(data_shape['y_sd']))
    
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

    @property
    def param_vector(self): return np.array([])

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
    def param_vector(self): return np.array([])
        
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
            if np.random.rand() < 1.0 / 3:
                self.sf = np.random.normal(loc=np.log(np.abs(data_shape['y_mean'])), scale=sd)
            elif np.random.rand() < 2.0 / 3:
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

class RQKernel(Kernel):
    def __init__(self, dimension=None, lengthscale=None, sf=None, alpha=None):
        self.dimension = dimension
        self.lengthscale = lengthscale
        self.sf = sf
        self.alpha = alpha

    # Properties
        
    @property
    def gpml_function(self): return '{@covRQiso}'
    
    @property
    def id(self): return 'RQ'
    
    @property
    def param_vector(self): return np.array([self.lengthscale, self.sf, self.alpha])
        
    @property
    def latex(self): return '{\\sc RQ}' 
    
    @property
    def syntax(self): return colored('RQ_%s' % self.dimension, self.depth)

    # Methods

    def copy(self): return RQKernel(dimension=self.dimension, lengthscale=self.lengthscale, sf=self.sf, alpha=self.alpha)
        
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
        if self.alpha == None:
            # Not sure of good heuristics for alpha value
            self.alpha = np.random.normal(loc=0, scale=2*sd) # Twice sd since heuristic is very basic         
    
    def __repr__(self):
        return 'RQKernel(dimension=%s, lengthscale=%s, sf=%s, alpha=%s)' % \
               (self.dimension, self.lengthscale, self.sf, self.alpha)
    
    def pretty_print(self):
        return colored('RQ(dim=%s, ell=%s, sf=%s, alpha=%s)' % \
               (self.dimension, \
                format_if_possible('%1.1f', self.lengthscale), \
                format_if_possible('%1.1f', self.sf), \
                format_if_possible('%1.1f', self.alpha)), \
               self.depth)   

    def load_param_vector(self, params):
        lengthscale, sf, alpha = params # N.B. - expects list input
        self.lengthscale = lengthscale  
        self.sf = sf  
        self.alpha = alpha  

class PeriodicKernel(Kernel):
    def __init__(self, dimension=None, lengthscale=None, period=None, sf=None):
        self.dimension = dimension
        self.lengthscale = lengthscale
        self.period = period
        self.sf = sf

    # Properties
        
    @property
    def gpml_function(self): return '{@covPeriodicNoDC}'
    
    @property
    def id(self): return 'Per'
    
    @property
    def param_vector(self): return np.array([self.lengthscale, self.period, self.sf])
        
    @property
    def latex(self): return '{\\sc Per}' 
    
    @property
    def syntax(self): return colored('Per_%s' % self.dimension, self.depth)

    # Methods

    def copy(self): return PeriodicKernel(dimension=self.dimension, lengthscale=self.lengthscale, period=self.period, sf=self.sf)
        
    def initialise_params(self, sd=1, data_shape=None):
        if self.lengthscale == None:
            # Lengthscale is relative to period so this parameter does not need to scale
            self.lengthscale = np.random.normal(loc=0, scale=sd)
        if self.period == None:
            #### Explanation : This is centered on about 25 periods
            # Min period represents a minimum sensible scale
            # Scale with data_scale or data range or 25 * distance between data points (this is a quick hack)
            if np.random.rand() < 0.33:
                # if data_shape['min_period'] is None:
                #     self.period = np.random.normal(loc=data_shape['x_sd'][self.dimension]-2, scale=sd)
                # else:
                #     self.period = utils.misc.sample_truncated_normal(loc=data_shape['x_sd'][self.dimension]-2, scale=sd, min_value=data_shape['min_period'][self.dimension])
                self.period = np.random.normal(loc=0, scale=sd)
            elif np.random.rand() < 0.5:
                # if data_shape['min_period'] is None:
                #     self.period = np.random.normal(loc=np.log(data_shape['x_max'][self.dimension]-data_shape['x_min'][self.dimension])-3.2, scale=sd)
                # else:
                #     self.period = utils.misc.sample_truncated_normal(loc=np.log(data_shape['x_max'][self.dimension]-data_shape['x_min'][self.dimension])-3.2, scale=sd, min_value=data_shape['min_period'][self.dimension])
                self.period = np.random.normal(loc=-3.95, scale=sd)
            else:
                # if data_shape['min_period'] is None:
                #     self.period = np.random.normal(loc=np.log(data_shape['x_min_abs_diff'][self.dimension]) + 3.2, scale=sd)
                # else:
                #     self.period = utils.misc.sample_truncated_normal(loc=np.log(data_shape['x_min_abs_diff'][self.dimension]) + 3.2, scale=sd, min_value=data_shape['min_period'][self.dimension])
                self.period = np.random.normal(loc=-5.9, scale=sd)
        if self.sf == None:
            # Set scale factor with output scale or neutrally
            if np.random.rand() < 0.5:
                self.sf = np.random.normal(loc=data_shape['y_sd'], scale=sd)
            else:
                self.sf = np.random.normal(loc=0, scale=sd)         
    
    def __repr__(self):
        return 'PeriodicKernel(dimension=%s, lengthscale=%s, period=%s, sf=%s)' % \
               (self.dimension, self.lengthscale, self.period, self.sf)
    
    def pretty_print(self):
        return colored('Per(dim=%s, ell=%s, per=%s, sf=%s)' % \
               (self.dimension, \
                format_if_possible('%1.1f', self.lengthscale), \
                format_if_possible('%1.1f', self.period), \
                format_if_possible('%1.1f', self.sf)), \
               self.depth)   

    def load_param_vector(self, params):
        lengthscale, period, sf = params # N.B. - expects list input
        self.lengthscale = lengthscale  
        self.period = period 
        self.sf = sf  
            
    def out_of_bounds(self, constraints):
        return (self.period < constraints['min_period'][self.dimension]) or \
               (self.period > constraints['max_period'][self.dimension])

class PeriodicKernelOLD(Kernel):
    def __init__(self, dimension=None, lengthscale=None, period=None, sf=None):
        self.dimension = dimension
        self.lengthscale = lengthscale
        self.period = period
        self.sf = sf

    # Properties
        
    @property
    def gpml_function(self): return '{@covPeriodic}'
    
    @property
    def id(self): return 'PerOLD'
    
    @property
    def param_vector(self): return np.array([self.lengthscale, self.period, self.sf])
        
    @property
    def latex(self): return '{\\sc PerOLD}' 
    
    @property
    def syntax(self): return colored('PerOLD_%s' % self.dimension, self.depth)

    # Methods

    def copy(self): return PeriodicKernelOLD(dimension=self.dimension, lengthscale=self.lengthscale, period=self.period, sf=self.sf)
        
    def initialise_params(self, sd=1, data_shape=None):
        if self.lengthscale == None:
            # Lengthscale is relative to period so this parameter does not need to scale
            self.lengthscale = np.random.normal(loc=0, scale=sd)
        if self.period == None:
            #### Explanation : This is centered on about 25 periods
            # Min period represents a minimum sensible scale
            # Scale with data_scale or data range
            if np.random.rand() < 0.5:
                if data_shape['min_period'] is None:
                    self.period = np.random.normal(loc=data_shape['x_sd'][self.dimension]-2, scale=sd)
                else:
                    self.period = utils.misc.sample_truncated_normal(loc=data_shape['x_sd'][self.dimension]-2, scale=sd, min_value=data_shape['min_period'][self.dimension])
            else:
                if data_shape['min_period'] is None:
                    self.period = np.random.normal(loc=np.log(data_shape['x_max'][self.dimension]-data_shape['x_min'][self.dimension])-3.2, scale=sd)
                else:
                    self.period = utils.misc.sample_truncated_normal(loc=np.log(data_shape['x_max'][self.dimension]-data_shape['x_min'][self.dimension])-3.2, scale=sd, min_value=data_shape['min_period'][self.dimension])
        if self.sf == None:
            # Set scale factor with output scale or neutrally
            if np.random.rand() < 0.5:
                self.sf = np.random.normal(loc=data_shape['y_sd'], scale=sd)
            else:
                self.sf = np.random.normal(loc=0, scale=sd)         
    
    def __repr__(self):
        return 'PeriodicKernelOLD(dimension=%s, lengthscale=%s, period=%s, sf=%s)' % \
               (self.dimension, self.lengthscale, self.period, self.sf)
    
    def pretty_print(self):
        return colored('PerOLD(dim=%s, ell=%s, per=%s, sf=%s)' % \
               (self.dimension, \
                format_if_possible('%1.1f', self.lengthscale), \
                format_if_possible('%1.1f', self.period), \
                format_if_possible('%1.1f', self.sf)), \
               self.depth)   

    def load_param_vector(self, params):
        lengthscale, period, sf = params # N.B. - expects list input
        self.lengthscale = lengthscale  
        self.period = period 
        self.sf = sf  
            
    def out_of_bounds(self, constraints):
        return (self.period < constraints['min_period'][self.dimension]) or \
               (self.period > constraints['max_period'][self.dimension])

class LinearKernel(Kernel):
    def __init__(self, dimension=None, location=None, sf=None):
        self.dimension = dimension
        self.location = location
        self.sf = sf

    # Properties
        
    @property
    def gpml_function(self): return '{@covLinear}'
    
    @property
    def id(self): return 'Lin'
    
    @property
    def is_stationary(self): return False
    
    @property
    def param_vector(self): return np.array([self.sf, self.location])
        
    @property
    def latex(self): return '{\\sc Lin}' 
    
    @property
    def syntax(self): return colored('Lin_%s' % self.dimension, self.depth)

    # Methods

    def copy(self): return LinearKernel(dimension=self.dimension, location=self.location, sf=self.sf)
        
    def initialise_params(self, sd=1, data_shape=None):
        if self.sf == None:
            # Scale factor scales with ratio of y std and x std (gradient = delta y / delta x)
            # Or with gradient or a neutral value
            rand = np.random.rand()
            if rand < 1.0/3:
                self.sf = np.random.normal(loc=(data_shape['y_sd'] - data_shape['x_sd'][self.dimension]), scale=sd)
            elif rand < 2.0/3:
                self.sf = np.random.normal(loc=np.log(np.abs((data_shape['y_max']-data_shape['y_min'])/(data_shape['x_max'][self.dimension]-data_shape['x_min'][self.dimension]))), scale=sd)
            else:
                self.sf = np.random.normal(loc=0, scale=sd)
        if self.location == None:
            # Uniform over 3 x data range
            self.location = np.random.uniform(low=2*data_shape['x_min'][self.dimension]-data_shape['x_max'][self.dimension], high=2*data_shape['x_max'][self.dimension]-data_shape['x_min'][self.dimension])      
    
    def __repr__(self):
        return 'LinearKernel(dimension=%s, location=%s, sf=%s)' % \
               (self.dimension, self.location, self.sf)
    
    def pretty_print(self):
        return colored('Lin(dim=%s, loc=%s, sf=%s)' % \
               (self.dimension, \
                format_if_possible('%1.1f', self.location), \
                format_if_possible('%1.1f', self.sf)), \
               self.depth)   

    def load_param_vector(self, params):
        sf, location = params # N.B. - expects list input
        self.location = location 
        self.sf = sf  

class LinearKernelOLD(Kernel):
    def __init__(self, dimension=None, location=None, invsf=None, offset=None):
        self.dimension = dimension
        self.location = location
        self.invsf = invsf
        self.offset = offset

    # Properties
        
    @property
    def gpml_function(self): return '{@covSum, {@covConst, @covLINscaleshift}}'
    
    @property
    def id(self): return 'LinOLD'
    
    @property
    def is_stationary(self): return False
    
    @property
    def param_vector(self): return np.array([self.offset, self.invsf, self.location])
        
    @property
    def latex(self): return '{\\sc LinOLD}' 
    
    @property
    def syntax(self): return colored('LinOLD_%s' % self.dimension, self.depth)

    # Methods

    def copy(self): return LinearKernelOLD(dimension=self.dimension, location=self.location, invsf=self.invsf, offset=self.offset)
        
    def initialise_params(self, sd=1, data_shape=None):
        if self.invsf == None:
            # Inverse scale factor scales with inverse ratio of y std and x std (gradient = delta y / delta x)
            # Or with gradient or a neutral value
            rand = np.random.rand()
            if rand < 1.0/3:
                self.invsf = -np.random.normal(loc=(data_shape['y_sd'] - data_shape['x_sd'][self.dimension]), scale=sd)
            elif rand < 2.0/3:
                self.invsf = -np.random.normal(loc=np.log(np.abs((data_shape['y_max']-data_shape['y_min'])/(data_shape['x_max'][self.dimension]-data_shape['x_min'][self.dimension]))), scale=sd)
            else:
                self.invsf = np.random.normal(loc=0, scale=sd)
        if self.location == None:
            # Uniform over 3 x data range
            self.location = np.random.uniform(low=2*data_shape['x_min'][self.dimension]-data_shape['x_max'][self.dimension], high=2*data_shape['x_max'][self.dimension]-data_shape['x_min'][self.dimension])   
        if self.offset == None:
            # Not sure of good heuristics for offset value
            self.offset = np.random.normal(loc=0, scale=2*sd) # Twice sd since heuristic is very basic       
    
    def __repr__(self):
        return 'LinearKernelOLD(dimension=%s, location=%s, invsf=%s, offset=%s)' % \
               (self.dimension, self.location, self.invsf, self.offset)
    
    def pretty_print(self):
        return colored('LinOLD(dim=%s, loc=%s, invsf=%s, off=%s)' % \
               (self.dimension, \
                format_if_possible('%1.1f', self.location), \
                format_if_possible('%1.1f', self.invsf), \
                format_if_possible('%1.1f', self.offset)), \
               self.depth)   

    def load_param_vector(self, params):
        offset, invsf, location = params # N.B. - expects list input
        self.location = location 
        self.invsf = invsf
        self.offset = offset  

class SpectralKernel(Kernel):
    def __init__(self, dimension=None, lengthscale=None, period=None, sf=None):
        self.dimension = dimension
        self.lengthscale = lengthscale
        self.period = period
        self.sf = sf

    # Properties
        
    @property
    def gpml_function(self): return '{@covProd, {@covSEiso, @covCosUnit}}'
    
    @property
    def id(self): return 'SP'
    
    @property
    def param_vector(self): return np.array([self.lengthscale, self.sf, self.period])
        
    @property
    def latex(self): return '{\\sc SP}' 
    
    @property
    def syntax(self): return colored('SP_%s' % self.dimension, self.depth)

    # Methods

    def copy(self): return SpectralKernel(dimension=self.dimension, lengthscale=self.lengthscale, period=self.period, sf=self.sf)
        
    def initialise_params(self, sd=1, data_shape=None):
        if self.lengthscale == None:
            # Set lengthscale with input scale or neutrally
            if np.random.rand() < 0.5:
                self.lengthscale = np.random.normal(loc=data_shape['x_sd'][self.dimension], scale=sd)
            else:
                # Long lengthscale ~ infty = neutral
                self.lengthscale = np.random.normal(loc=np.log(2*(data_shape['x_max'][self.dimension]-data_shape['x_min'][self.dimension])), scale=sd)
        if self.period == None:
            #### Explanation : This is centered on about 25 periods
            # Min period represents a minimum sensible scale
            # Scale with data_scale or data range
            if np.random.rand() < 0.66:
                if np.random.rand() < 0.5:
                    if data_shape['min_period'] is None:
                        self.period = np.random.normal(loc=data_shape['x_sd'][self.dimension]-2, scale=sd)
                    else:
                        self.period = utils.misc.sample_truncated_normal(loc=data_shape['x_sd'][self.dimension]-2, scale=sd, min_value=data_shape['min_period'][self.dimension])
                else:
                    if data_shape['min_period'] is None:
                        self.period = np.random.normal(loc=np.log(data_shape['x_max'][self.dimension]-data_shape['x_min'][self.dimension])-3.2, scale=sd)
                    else:
                        self.period = utils.misc.sample_truncated_normal(loc=np.log(data_shape['x_max'][self.dimension]-data_shape['x_min'][self.dimension])-3.2, scale=sd, min_value=data_shape['min_period'][self.dimension])
            else:
                # Spectral kernel can also approximate SE with long period
                self.period = np.log(data_shape['x_max'][self.dimension]-data_shape['x_min'][self.dimension])
        if self.sf == None:
            # Set scale factor with output scale or neutrally
            if np.random.rand() < 0.5:
                self.sf = np.random.normal(loc=data_shape['y_sd'], scale=sd)
            else:
                self.sf = np.random.normal(loc=0, scale=sd)         
    
    def __repr__(self):
        return 'SpectralKernel(dimension=%s, lengthscale=%s, period=%s, sf=%s)' % \
               (self.dimension, self.lengthscale, self.period, self.sf)
    
    def pretty_print(self):
        return colored('SP(dim=%s, ell=%s, per=%s, sf=%s)' % \
               (self.dimension, \
                format_if_possible('%1.1f', self.lengthscale), \
                format_if_possible('%1.1f', self.period), \
                format_if_possible('%1.1f', self.sf)), \
               self.depth)   

    def load_param_vector(self, params):
        lengthscale, sf, period = params # N.B. - expects list input
        self.lengthscale = lengthscale  
        self.period = period 
        self.sf = sf  
            
    def out_of_bounds(self, constraints):
        return (self.period < constraints['min_period'][self.dimension])

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
    def is_abelian(self): return True

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

    def out_of_bounds(self, constraints):
        return any([o.out_of_bounds(constraints=constraints) for o in self.operands])

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
    def is_abelian(self): return True

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

    def out_of_bounds(self, constraints):
        return any([o.out_of_bounds(constraints=constraints) for o in self.operands])

class ChangePointKernel(Kernel):
    def __init__(self, dimension=None, location=None, steepness=None, operands=None):
        assert len(operands) == 2
        self.dimension = dimension
        self.location = location
        self.steepness = steepness
        if operands is None:
            self.operands = []
        else:
            self.operands  = operands

    # Properties

    @property    
    def is_stationary(self): return False

    @property
    def sf(self):
        raise RuntimeError('Cannot ask for scale factor of non-stationary kernel')
        
    @property
    def arity(self): return 2
        
    @property
    def gpml_function(self): return '{@covChangePointMultiD}'
    
    @property
    def id(self): return 'CP'
    
    @property
    def param_vector(self):
        return np.concatenate([np.array([self.location, self.steepness])] + [o.param_vector for o in self.operands])
        
    @property
    def latex(self):
        return '{\\sc CP}\\left( ' + ' , '.join([o.latex for o in self.operands]) + ' \\right)'  
    
    @property
    def syntax(self): 
        return colored('CP( ', self.depth) + \
                self.operands[0].syntax + \
                colored(', ', self.depth) + \
                self.operands[1].syntax + \
                colored(' )', self.depth)
       
    @property    
    def is_operator(self): return True
       
    @property    
    def is_abelian(self): return False

    @property
    def effective_params(self):
        return 2 + sum([o.effective_params for o in self.operands])

    @property
    def depth(self):
        return max([o.depth for o in self.operands]) + 1

    # Methods

    def copy(self):
        return ChangePointKernel(dimension=self.dimension, location=self.location, steepness=self.steepness, operands=[o.copy() for o in self.operands])
        
    def initialise_params(self, sd=1, data_shape=None):
        if self.location is None:
            # Location uniform in data range
            self.location = np.random.uniform(data_shape['x_min'][self.dimension], data_shape['x_max'][self.dimension])
        if self.steepness is None:
            # Set steepness with inverse input scale
            self.steepness = np.random.normal(loc=3.3-np.log((data_shape['x_max'][self.dimension] - data_shape['x_min'][self.dimension])), scale=1)
        for o in self.operands:
            o.initialise_params(sd=sd, data_shape=data_shape)
    
    def __repr__(self):
        return 'ChangePointKernel(dimension=%s, location=%s, steepness=%s, operands=%s)' % \
                (self.dimension, self.location, self.steepness, '[ ' + ', '.join([o.__repr__() for o in self.operands]) + ' ]')   
    
    def pretty_print(self):
        return colored('CP(dim=%s, loc=%s, steep=%s, ' % \
               (self.dimension, format_if_possible('%1.1f', self.location), format_if_possible('%1.1f', self.steepness)), self.depth) + \
                self.operands[0].pretty_print() + \
                colored(', ', self.depth) + \
                self.operands[1].pretty_print() + \
                colored(')', self.depth)

    def load_param_vector(self, params):
        self.location = params[0]
        self.steepness = params[1]
        start = 2
        for o in self.operands:
            end = start + o.num_params
            o.load_param_vector(params[start:end])
            start = end

    def get_gpml_expression(self, dimensions):
        #return '{@covChangePointMultiD, %s, {%s}}' % (self.dimension + 1, ', '.join(o.get_gpml_expression(dimensions=dimensions) for o in self.operands))
        return '{@covChangePointMultiD, {%s, %s}}' % (self.dimension + 1, ', '.join(o.get_gpml_expression(dimensions=dimensions) for o in self.operands))

    def multiply_by_const(self, sf):
        for o in self.operands:
            o.multiply_by_const(sf=sf)                    
            
    def out_of_bounds(self, constraints):
        return (self.location < constraints['x_min'][self.dimension]) or \
               (self.location > constraints['x_max'][self.dimension]) or \
               (self.steepness < -np.log((constraints['x_max'][self.dimension] -constraints['x_min'][self.dimension])) + 2.3) or \
               (any([o.out_of_bounds(constraints) for o in self.operands])) 

class ChangeWindowKernel(Kernel):
    def __init__(self, dimension=None, location=None, steepness=None, width=None, operands=None):
        assert len(operands) == 2
        self.dimension = dimension
        self.location = location
        self.steepness = steepness
        self.width = width
        if operands is None:
            self.operands = []
        else:
            self.operands  = operands

    # Properties

    @property    
    def is_stationary(self): return False

    @property
    def sf(self):
        raise RuntimeError('Cannot ask for scale factor of change window kernel')
        
    @property
    def arity(self): return 2
        
    @property
    def gpml_function(self): return '{@covChangeWindowMultiD}'
    
    @property
    def id(self): return 'CW'
    
    @property
    def param_vector(self):
        return np.concatenate([np.array([self.location, self.steepness, self.width])] + [o.param_vector for o in self.operands])
        
    @property
    def latex(self):
        return '{\\sc CW}\\left( ' + ' , '.join([o.latex for o in self.operands]) + ' \\right)'  
    
    @property
    def syntax(self): 
        return colored('CW( ', self.depth) + \
                self.operands[0].syntax + \
                colored(', ', self.depth) + \
                self.operands[1].syntax + \
                colored(' )', self.depth)
       
    @property    
    def is_operator(self): return True
       
    @property    
    def is_abelian(self): return False

    @property
    def effective_params(self):
        return 3 + sum([o.effective_params for o in self.operands])

    @property
    def depth(self):
        return max([o.depth for o in self.operands]) + 1

    # Methods

    def copy(self):
        return ChangeWindowKernel(dimension=self.dimension, location=self.location, steepness=self.steepness, width=self.width, operands=[o.copy() for o in self.operands])
        
    def initialise_params(self, sd=1, data_shape=None):
        if self.location is None:
            # Location uniform in data range
            self.location = np.random.uniform(data_shape['x_min'][self.dimension], data_shape['x_max'][self.dimension])
        if self.steepness is None:
            # Set steepness with inverse input scale
            self.steepness = np.random.normal(loc=3.3-np.log((data_shape['x_max'][self.dimension] - data_shape['x_min'][self.dimension])), scale=1)
        if self.width is None:
            # Set width with input scale - but expecting small widths
            self.width = np.random.normal(loc=np.log(0.1*(data_shape['x_max'][self.dimension] - data_shape['x_min'][self.dimension])), scale=1)
        for o in self.operands:
            o.initialise_params(sd=sd, data_shape=data_shape)
    
    def __repr__(self):
        return 'ChangeWindowKernel(dimension=%s, location=%s, steepness=%s, width=%s, operands=%s)' % \
                (self.dimension, self.location, self.steepness, self.width, '[ ' + ', '.join([o.__repr__() for o in self.operands]) + ' ]')   
    
    def pretty_print(self):
        return colored('CW(dim=%s, loc=%s, steep=%s, width=%s, ' % \
               (self.dimension, format_if_possible('%1.1f', self.location), format_if_possible('%1.1f', self.steepness), format_if_possible('%1.1f', self.width)), self.depth) + \
                self.operands[0].pretty_print() + \
                colored(', ', self.depth) + \
                self.operands[1].pretty_print() + \
                colored(')', self.depth)

    def load_param_vector(self, params):
        self.location = params[0]
        self.steepness = params[1]
        self.width = params[2]
        start = 3
        for o in self.operands:
            end = start + o.num_params
            o.load_param_vector(params[start:end])
            start = end

    def get_gpml_expression(self, dimensions):
        #return '{@covChangeWindowMultiD, %s, {%s}}' % (self.dimension + 1, ', '.join(o.get_gpml_expression(dimensions=dimensions) for o in self.operands))
        return '{@covChangeWindowMultiD, {%s, %s}}' % (self.dimension + 1, ', '.join(o.get_gpml_expression(dimensions=dimensions) for o in self.operands))
        #return '{@covChangeBurstTanh, {%s}}' % (', '.join(o.get_gpml_expression(dimensions=dimensions) for o in self.operands))

    def multiply_by_const(self, sf):
        for o in self.operands:
            o.multiply_by_const(sf=sf)                    
            
    def out_of_bounds(self, constraints):
        return (self.location - np.exp(self.width)/2 < constraints['x_min'][self.dimension] + 0.05 * (constraints['x_max'][self.dimension] - constraints['x_min'][self.dimension])) or \
               (self.location + np.exp(self.width)/2 > constraints['x_max'][self.dimension] - 0.05 * (constraints['x_max'][self.dimension] - constraints['x_min'][self.dimension])) or \
               (self.width > np.log(0.25*(constraints['x_max'][self.dimension] - constraints['x_min'][self.dimension]))) or \
               (self.steepness < -np.log((constraints['x_max'][self.dimension] - constraints['x_min'][self.dimension])) + 2.3) or \
               (any([o.out_of_bounds(constraints) for o in self.operands])) 

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
    def gpml_function(self):
        # TODO - once GPML infExact fixed, always return likGauss
        if self.sf > -np.Inf:
            return '{@likGauss}'
        else:
            return '{@likDelta}'

    @property    
    def is_thunk(self): return True
    
    @property
    def id(self): return 'Gauss'
    
    @property
    def param_vector(self):
        if self.sf > -np.Inf:
            return np.array([self.sf])
        else:
            return np.array([])
        
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

    @property
    def gpml_inference_method(self):
        if self.sf > -np.Inf:
            return '@infExact'
        else:
            return '@infDelta'

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
        if len(params) == 0:
            self.sf = -np.Inf
        else:
            sf, = params # N.B. - expects list input
            self.sf = sf   

class LikErf(Likelihood):
    def __init__(self, inference='EP'):
        self.inference = inference

    # Properties
        
    @property
    def gpml_function(self):
        return '{@likErf}'

    @property    
    def is_thunk(self): return True
    
    @property
    def id(self): return 'Erf'
    
    @property
    def param_vector(self):
        return np.array([])
        
    @property
    def latex(self): return '{\\sc ERF}' 
    
    @property
    def syntax(self): return colored('Erf', self.depth)

    @property
    def effective_params(self):
        return 0

    @property
    def gpml_inference_method(self):
        if self.inference == 0:
            return '@infEP'
        elif self.inference == 1:
            return '@infLaplace'
        else:
            raise RuntimeError('Unrecognised inference method code: %s' % self.inference)

    # Methods

    def copy(self): return LikErf(inference=self.inference)
        
    def initialise_params(self, sd=1, data_shape=None):
        pass
    
    def __repr__(self):
        return 'LikErf(inference=%s)' % (self.inference)
    
    def pretty_print(self):
        return colored('Erf(inf=%s)' % format_if_possible('%d', self.inference), self.depth)   

    def load_param_vector(self, params):
        pass 

##############################################
#                                            #
#         Miscellaneous functions            #
#                                            #
##############################################

def repr_to_model(string):
    return eval(string)

def remove_duplicates(things):
    # This is possible since things are hashable
    return list(set(things))

#### TODO - these should be extended to models
         
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
                   LinearKernel(), \
                   PeriodicKernel(), \
                   #CosineKernelFamily(), \
                   SpectralKernel(), \
                   RQKernel(), \
                   PeriodicKernelOLD(), \
                   LinearKernelOLD(), \
                   NoiseKernel()]:
        if kernel.id in base_kernel_names.split(','):
            yield kernel 

#### TODO - these should be added to model and kernel classes

def add_random_restarts_single_k(kernel, n_rand, sd, data_shape):
    '''Returns a list of kernels with random restarts for default values'''
    kernel_list = []
    for dummy in range(n_rand):
        k = kernel.copy()
        k.initialise_params(sd=sd, data_shape=data_shape)
        kernel_list.append(k)
    return kernel_list

def add_random_restarts_single_l(lik, n_rand, sd, data_shape):
    '''Returns a list of likelihoods with random restarts for default values'''
    lik_list = []
    for dummy in range(n_rand):
        l = lik.copy()
        l.initialise_params(sd=sd, data_shape=data_shape)
        lik_list.append(l)
    return lik_list

def add_random_restarts_single_m(a_mean, n_rand, sd, data_shape):
    '''Returns a list of means with random restarts for default values'''
    mean_list = []
    for dummy in range(n_rand):
        m = a_mean.copy()
        m.initialise_params(sd=sd, data_shape=data_shape)
        mean_list.append(m)
    return mean_list

def add_random_restarts_k(kernels, n_rand=1, sd=4, data_shape=None):    
    '''Augments the list to include random restarts of all default value parameters'''
    return [k_rand for kernel in kernels for k_rand in add_random_restarts_single_k(kernel, n_rand, sd, data_shape)] 

def add_random_restarts(models, n_rand=1, sd=4, data_shape=None):
    new_models = []
    for a_model in models:
        for (kernel, likelihood, mean) in zip(add_random_restarts_single_k(a_model.kernel, n_rand=n_rand, sd=sd, data_shape=data_shape), \
                                              add_random_restarts_single_l(a_model.likelihood, n_rand=n_rand, sd=sd, data_shape=data_shape), \
                                              add_random_restarts_single_m(a_model.mean, n_rand=n_rand, sd=sd, data_shape=data_shape)):
            new_model = a_model.copy()
            new_model.kernel = kernel
            new_model.likelihood = likelihood
            new_model.mean = mean
            new_models.append(new_model)
    return new_models 

def add_jitter_k(kernels, sd=0.1):    
    '''Adds random noise to all parameters - empirically observed to help when optimiser gets stuck'''
    for k in kernels:
        k.load_param_vector(k.param_vector + np.random.normal(loc=0., scale=sd, size=k.param_vector.size))
    return kernels     

def add_jitter(models, sd=0.1):
    for a_model in models:
        a_model.kernel = add_jitter_k([a_model.kernel], sd=sd)[0]
    return models 

##############################################
#                                            #
#     Old kernel functions to be revived     #
#                                            #
##############################################  
        
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