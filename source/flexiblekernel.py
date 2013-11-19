'''
Created Nov 2012

@authors: James Robert Lloyd (jrl44@cam.ac.uk)
          David Duvenaud (dkd23@cam.ac.uk)
          Roger Grosse (rgrosse@mit.edu)
'''

import itertools
import numpy as np
inf = np.inf
try:
    import termcolor
    has_termcolor = True
except:
    has_termcolor = False

try:
    import config
    color_scheme = config.COLOR_SCHEME
except:
    color_scheme = 'dark'

import operator
from utils import psd_matrices
import utils.misc
import re
from scipy.special import i0 # 0th order Bessel function of the first kind

# Pretty printing - move this to utils

def paren_colors():
    if color_scheme == 'dark':
        return ['red', 'green', 'cyan', 'magenta', 'yellow']
    elif color_scheme == 'light':
        return ['blue', 'red', 'magenta', 'green', 'cyan']
    else:
        raise RuntimeError('Unknown color scheme: %s' % color_scheme)

def colored(text, depth):
    if has_termcolor:
        colors = paren_colors()
        color = colors[depth % len(colors)]
        return termcolor.colored(text, color, attrs=['bold'])
    else:
        return text

def format_if_possible(format, value):
    try:
        return format % value
    except:
        return '%s' % value


# Base kernel class with default properties and methods - many need to be overwritten

class Kernel:
    # Syntactic sugar e.g. k1 + k2
    def __add__(self, other):
        assert isinstance(other, Kernel)
        if isinstance(other, SumKernel):
            return SumKernel([self] + other.operands).copy()
        else:
            return SumKernel([self, other]).copy()
    
    # Syntactic sugar e.g. k1 * k2
    def __mul__(self, other):
        assert isinstance(other, Kernel)
        if isinstance(other, ProductKernel):
            return ProductKernel([self] + other.operands).copy()
        else:
            return ProductKernel([self, other])
            
    def __hash__(self):
        return hash(self.__repr__())

    # Properties

    @property
    def gpml_function(self):
        raise RuntimeError('This property must be overriden')
       
    @property    
    def is_stationary(self):
        return True
       
    @property    
    def is_operator(self):
        return False
       
    @property    
    def is_thunk(self):
        return False
    
    @property
    def id(self):
        raise RuntimeError('This property must be overriden')

    @property
    def effective_params(self):
        if not self.is_operator:
            '''This is true of all base kernels, hence definition here'''  
            return len(self.param_vector())
        else:
            raise RuntimeError('Operators must override this property')
    
    @property
    def param_vector(self):
        raise RuntimeError('This property must be overriden')

    @property
    def latex(self):
        raise RuntimeError('This property must be overriden') 

    @property
    def depth(self):
        if not self.is_operator:
            return 0 
        else:
            raise RuntimeError('Operators must override this property')
    
    @property
    def num_params(self):
        raise RuntimeError('This property must be overriden') 

    @property
    def syntax(self):
        raise RuntimeError('This property must be overriden') 

    # Methods

    def copy(self):
        raise RuntimeError('This method must be overriden')

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
                return '{@covMask, {%s, %s}}' % (dim_vec_str, self.gpml_function)
        else:
            raise RuntimeError('Operators must override this method')

    def initialise_params(self, sd=1, data_shape=None):
        raise RuntimeError('This method must be overriden')

    def __repr__(self):
        return 'Kernel()'
    
    def pretty_print(self):
        return RuntimeError('This method must be overriden')
        
    def out_of_bounds(self, constraints):
        '''Most kernels are allowed to have any parameter value'''
        return False

    def load_param_vector(self, params):
        return RuntimeError('This method must be overriden')

class NoiseKernel(Kernel):
    def __init__(self, sf=None):
        self.sf = sf

    # Properties
        
    @property
    def gpml_function(self):
        return '{@covNoise}'

    @property    
    def is_thunk(self):
        return True
    
    @property
    def id(self):
        return 'Noise'
    
    @property
    def param_vector(self):
        # order of args matches GPML
        return np.array([self.sf])
        
    @property
    def latex(self):
        return '{\\sc WN}' 
    
    @property
    def num_params(self):
        return 1
    
    @property
    def syntax(self):
        return colored('WN', self.depth)

    # Methods

    def copy(self):
        return NoiseKernel(self.sf)
        
    def initialise_params(self, sd=1, data_shape=None):
        if self.sf == None:
            # Set scale factor with 1/10 data std or neutrally
            if np.random.rand() < 0.5:
                self.sf = np.random.normal(loc=data_shape['output_scale']-np.log(10), scale=sd)
            else:
                self.sf = np.random.normal(loc=0, scale=sd)
    
    def __repr__(self):
        return 'NoiseKernel(sf=%s)' % (self.sf)
    
    def pretty_print(self):
        return colored('WN(sf=%s)' % (format_if_possible('%1.1f', self.sf)), self.depth)   
    
    def __cmp__(self, other):
        assert isinstance(other, Kernel)
        if cmp(self.__class__, other.__class__):
            return cmp(self.__class__, other.__class__)
        return cmp(self.sf, other.sf)  

    def load_param_vector(self, params):
        sf, = params # N.B. - expects list input
        self.sf = sf   

# class SqExpKernelFamily(BaseKernelFamily):
#     def from_param_vector(self, params):
#         lengthscale, output_variance = params
#         return SqExpKernel(lengthscale=lengthscale, output_variance=output_variance)
    
#     def num_params(self):
#         return 2
    
#     def pretty_print(self):
#         return colored('SqExp', self.depth())
    
#     @staticmethod
#     def default():
#         return SqExpKernel(0., 0.)
    
#     def __cmp__(self, other):
#         assert isinstance(other, KernelFamily)
#         if cmp(self.__class__, other.__class__):
#             return cmp(self.__class__, other.__class__)
#         return 0
    
#     def depth(self):
#         return 0
    
#     def id_name(self):
#         return 'SE'
    
#     @staticmethod    
#     def description():
#         return "Squared-exponential"

#     @staticmethod    
#     def params_description():
#         return "lengthscale"    

# class SqExpKernel(BaseKernel):
#     def __init__(self, lengthscale, output_variance):
#         self.lengthscale = lengthscale
#         self.output_variance = output_variance
        
#     def family(self):
#         return SqExpKernelFamily()
        
#     def gpml_kernel_expression(self):
#         return '{@covSEiso}'
    
#     def english_name(self):
#         return 'SqExp'
    
#     def id_name(self):
#         return 'SE'
    
#     def param_vector(self):
#         # order of args matches GPML
#         return np.array([self.lengthscale, self.output_variance])
        
#     def default_params_replaced(self, sd=1, data_shape=None):
#         result = self.param_vector()
#         if result[0] == 0:
#             # Set lengthscale with input scale or neutrally
#             if np.random.rand() < 0.5:
#                 result[0] = np.random.normal(loc=data_shape['input_scale'], scale=sd)
#             else:
#                 # Long lengthscale ~ infty = neutral
#                 result[0] = np.random.normal(loc=np.log(2*(data_shape['input_max']-data_shape['input_min'])), scale=sd)
#         if result[1] == 0:
#             # Set scale factor with output scale or neutrally
#             if np.random.rand() < 0.5:
#                 result[1] = np.random.normal(loc=data_shape['output_scale'], scale=sd)
#             else:
#                 result[1] = np.random.normal(loc=0, scale=sd)
#         return result

#     def copy(self):
#         return SqExpKernel(self.lengthscale, self.output_variance)
    
#     def __repr__(self):
#         return 'SqExpKernel(lengthscale=%f, output_variance=%f)' % (self.lengthscale, self.output_variance)
    
#     def pretty_print(self):
#         return colored('SE(ell=%1.1f, sf=%1.1f)' % (self.lengthscale, self.output_variance), self.depth())
    
#     def latex_print(self):
#         #return 'SE(\\ell=%1.1f, \\sigma=%1.1f)' % (self.lengthscale, self.output_variance)    
#         #return 'SE(\\ell=%1.1f)' % self.lengthscale
#         return 'SE'
        
#     def __cmp__(self, other):
#         assert isinstance(other, Kernel)
#         if cmp(self.__class__, other.__class__):
#             return cmp(self.__class__, other.__class__)
#         differences = [self.lengthscale - other.lengthscale, self.output_variance - other.output_variance]
#         differences = map(shrink_below_tolerance, differences)
#         return cmp(differences, [0] * len(differences))
    
#     def depth(self):
#         return 0
            
#     def out_of_bounds(self, constraints):
#         return self.lengthscale < constraints['min_lengthscale']
    
#     def english(self):
#         return lengthscale_description(self.lengthscale)          

# #### TODO - this is a code name for the reparametrised centred periodic
# class FourierKernelFamily(BaseKernelFamily):
#     def from_param_vector(self, params):
#         lengthscale, period, output_variance = params
#         return FourierKernel(lengthscale, period, output_variance)
    
#     def num_params(self):
#         return 3
    
#     def pretty_print(self):
#         return colored('FT', self.depth())
    
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
#                 result[2] = np.random.normal(loc=data_shape['output_scale'], scale=sd)
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
#                        self.depth())
        
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
#         return colored('Cos', self.depth())
    
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
#                 result[1] = np.random.normal(loc=data_shape['output_scale'], scale=sd)
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
#                        self.depth())
        
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
#         return colored('SP', self.depth())
    
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
#                 result[1] = np.random.normal(loc=data_shape['output_scale'], scale=sd)
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
#                        self.depth())
        
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
    
# class ConstKernelFamily(BaseKernelFamily):
#     def from_param_vector(self, params):
#         output_variance, = params # N.B. - expects list input
#         return ConstKernel(output_variance)
    
#     def num_params(self):
#         return 1
    
#     def pretty_print(self):
#         return colored('CS', self.depth())
    
#     @staticmethod
#     def default():
#         return ConstKernel(0.)
    
#     def __cmp__(self, other):
#         assert isinstance(other, KernelFamily)
#         if cmp(self.__class__, other.__class__):
#             return cmp(self.__class__, other.__class__)
#         return 0
    
#     def depth(self):
#         return 0
    
#     def id_name(self):
#         return 'Const'
    
#     @staticmethod    
#     def description():
#         return "Constant"

#     @staticmethod    
#     def params_description():
#         return "Output variance"        
    
# class ConstKernel(BaseKernel):
#     def __init__(self, output_variance):
#         self.output_variance = output_variance
        
#     def family(self):
#         return ConstKernelFamily()
        
#     def gpml_kernel_expression(self):
#         return '{@covConst}'
    
#     def english_name(self):
#         return 'CS'
    
#     def id_name(self):
#         return 'Const'
    
#     def param_vector(self):
#         # order of args matches GPML
#         return np.array([self.output_variance])

#     def copy(self):
#         return ConstKernel(self.output_variance)
        
#     def default_params_replaced(self, sd=1, data_shape=None):
#         result = self.param_vector()
#         if result[0] == 0:
#             # Set scale factor with output location, scale or neutrally
#             rand = np.random.rand()
#             if rand < 1.0 / 3:
#                 result[0] = np.random.normal(loc=np.log(np.abs(data_shape['output_location'])), scale=sd)
#             elif rand < 2.0 / 3:
#                 result[0] = np.random.normal(loc=data_shape['output_scale'], scale=sd)
#             else:
#                 result[0] = np.random.normal(loc=0, scale=sd)
#         return result
    
#     def __repr__(self):
#         return 'ConstKernel(output_variance=%f)' % \
#             (self.output_variance)
    
#     def pretty_print(self):
#         return colored('CS(sf=%1.1f)' % (self.output_variance),
#                        self.depth())
        
#     def latex_print(self):
#         return 'CS'    
    
#     def id_name(self):
#         return 'Const'       
    
#     def __cmp__(self, other):
#         assert isinstance(other, Kernel)
#         if cmp(self.__class__, other.__class__):
#             return cmp(self.__class__, other.__class__)
#         differences = [self.output_variance - other.output_variance]
#         differences = map(shrink_below_tolerance, differences)
#         return cmp(differences, [0] * len(differences))
        
#     def depth(self):
#         return 0    
        
# class ZeroKernelFamily(BaseKernelFamily):
#     def from_param_vector(self, params):
#         #### Note - expects list input
#         assert params == []
#         return ZeroKernel()
    
#     def num_params(self):
#         return 0
    
#     def pretty_print(self):
#         return colored('NIL', self.depth())
    
#     @staticmethod
#     def default():
#         return ZeroKernel()
    
#     def __cmp__(self, other):
#         assert isinstance(other, KernelFamily)
#         if cmp(self.__class__, other.__class__):
#             return cmp(self.__class__, other.__class__)
#         return 0
    
#     def depth(self):
#         return 0
    
#     def id_name(self):
#         return 'Zero'
    
#     @staticmethod    
#     def description():
#         return "Zero"

#     @staticmethod    
#     def params_description():
#         return "None"        
    
# class ZeroKernel(BaseKernel):
#     def __init__(self):
#         pass
        
#     def family(self):
#         return ZeroKernelFamily()
        
#     def gpml_kernel_expression(self):
#         return '{@covZero}'
    
#     def english_name(self):
#         return 'NIL'
    
#     def id_name(self):
#         return 'Zero'
    
#     def param_vector(self):
#         return np.array([])

#     def copy(self):
#         return ZeroKernel()
        
#     def default_params_replaced(self, sd=1, data_shape=None):
#         return self.param_vector()
    
#     def __repr__(self):
#         return 'ZeroKernel()'
    
#     def pretty_print(self):
#         return colored('NIL', self.depth())
        
#     def latex_print(self):
#         return 'NIL'       
    
#     def __cmp__(self, other):
#         assert isinstance(other, Kernel)
#         return cmp(self.__class__, other.__class__)
        
#     def depth(self):
#         return 0  

# class PureLinKernelFamily(BaseKernelFamily):
#     def from_param_vector(self, params):
#         lengthscale, location = params
#         return PureLinKernel(lengthscale=lengthscale, location=location)
    
#     def num_params(self):
#         return 2
    
#     def pretty_print(self):
#         return colored('PLN', self.depth())
    
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
#                 result[0] = np.random.normal(loc=-(data_shape['output_scale'] - data_shape['input_scale']), scale=sd)
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
#                        self.depth())
        
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
#         return colored('CPT(', self.depth()) + \
#             self.operands[0].pretty_print() + \
#             colored(', ', self.depth()) + \
#             self.operands[1].pretty_print() + \
#             colored(')', self.depth())

#     def default(self):
#         return ChangePointTanhKernel(0., 0., [op.default() for op in self.operands])
    
#     def __cmp__(self, other):
#         assert isinstance(other, KernelFamily)
#         if cmp(self.__class__, other.__class__):
#             return cmp(self.__class__, other.__class__)
#         return cmp(self.operands, other.operands)
    
#     def depth(self):
#         return max([op.depth() for op in self.operands]) + 1

# class ChangePointTanhKernel(KernelOperator):
#     def __init__(self, location, steepness, operands):
#         self.location = location
#         self.steepness = steepness
#         self.operands = operands
        
#     def family(self):
#         return ChangePointTanhKernelFamily([e.family() for e in self.operands])
        
#     def pretty_print(self): 
#         return colored('CPT(loc=%1.1f, steep=%1.1f, ' % (self.location, self.steepness), self.depth()) + \
#             self.operands[0].pretty_print() + \
#             colored(', ', self.depth()) + \
#             self.operands[1].pretty_print() + \
#             colored(')', self.depth())
            
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
#         return max([op.depth() for op in self.operands]) + 1
            
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
#         return colored('CBT(', self.depth()) + \
#             self.operands[0].pretty_print() + \
#             colored(', ', self.depth()) + \
#             self.operands[1].pretty_print() + \
#             colored(')', self.depth())
    
#     def default(self):
#         return ChangeBurstTanhKernel(0., 0., 0., [op.default() for op in self.operands])
    
#     def __cmp__(self, other):
#         assert isinstance(other, KernelFamily)
#         if cmp(self.__class__, other.__class__):
#             return cmp(self.__class__, other.__class__)
#         return cmp(self.operands, other.operands)
    
#     def depth(self):
#         return max([op.depth() for op in self.operands]) + 1

# class ChangeBurstTanhKernel(KernelOperator):
#     def __init__(self, location, steepness, width, operands):
#         self.location = location
#         self.steepness = steepness
#         self.width = width
#         self.operands = operands
        
#     def family(self):
#         return ChangeBurstTanhKernelFamily([e.family() for e in self.operands])
        
#     def pretty_print(self): 
#         return colored('CBT(loc=%1.1f, steep=%1.1f, width=%1.1f, ' % (self.location, self.steepness, self.width), self.depth()) + \
#             self.operands[0].pretty_print() + \
#             colored(', ', self.depth()) + \
#             self.operands[1].pretty_print() + \
#             colored(')', self.depth())
            
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
#         return max([op.depth() for op in self.operands]) + 1
            
#     def out_of_bounds(self, constraints):
#         return (self.location - np.exp(self.width)/2 < constraints['input_min'] + 0.05 * (constraints['input_max'] -constraints['input_min'])) or \
#                (self.location + np.exp(self.width)/2 > constraints['input_max'] - 0.05 * (constraints['input_max'] -constraints['input_min'])) or \
#                (self.width > np.log(0.25*(constraints['input_max'] - constraints['input_min']))) or \
#                (self.steepness < -np.log((constraints['input_max'] -constraints['input_min'])) + 2.3) or \
#                (any([o.out_of_bounds(constraints) for o in self.operands])) 
        
# class SumKernelFamily(KernelOperatorFamily):
#     def __init__(self, operands):
#         self.operands = operands
        
#     def from_param_vector(self, params):
#         start = 0
#         ops = []
#         for e in self.operands:
#             end = start + e.num_params()
#             ops.append(e.from_param_vector(params[start:end]))
#             start = end
#         return SumKernel(ops)
    
#     def num_params(self):
#         return sum([e.num_params() for e in self.operands])
    
#     def pretty_print(self):
#         op = colored(' + ', self.depth())
#         return colored('( ', self.depth()) + \
#             op.join([e.pretty_print() for e in self.operands]) + \
#             colored(' ) ', self.depth())
    
#     def default(self):
#         return SumKernel([op.default() for op in self.operands])
    
#     def __cmp__(self, other):
#         assert isinstance(other, KernelFamily)
#         if cmp(self.__class__, other.__class__):
#             return cmp(self.__class__, other.__class__)
#         return cmp(self.operands, other.operands)
    
#     def depth(self):
#         return max([op.depth() for op in self.operands]) + 1

# class SumKernel(KernelOperator):
#     def __init__(self, operands):
#         self.operands = operands
        
#     def family(self):
#         return SumKernelFamily([e.family() for e in self.operands])
        
#     def pretty_print(self):
#         #### TODO - Should this call the family method?
#         op = colored(' + ', self.depth())
#         return colored('( ', self.depth()) + \
#             op.join([e.pretty_print() for e in self.operands]) + \
#             colored(' ) ', self.depth())
            
#     def latex_print(self):
#         return '\\left( ' + ' + '.join([e.latex_print() for e in self.operands]) + ' \\right)'            
            
#     def __repr__(self):
#         return 'SumKernel(%s)' % \
#             ('[ ' + ', '.join([o.__repr__() for o in self.operands]) + ' ]')                
    
#     def gpml_kernel_expression(self):
#         return '{@covSum, {%s}}' % ', '.join(e.gpml_kernel_expression() for e in self.operands)
    
#     def copy(self):
#         return SumKernel([e.copy() for e in self.operands])

#     def param_vector(self):
#         return np.concatenate([e.param_vector() for e in self.operands])
        
#     def effective_params(self):
#         return sum([o.effective_params() for o in self.operands])
        
#     def default_params_replaced(self, sd=1, data_shape=None):
#         '''Returns the parameter vector with any default values replaced with random Gaussian'''
#         return np.concatenate([o.default_params_replaced(sd=sd, data_shape=data_shape) for o in self.operands])
    
#     def __cmp__(self, other):
#         assert isinstance(other, Kernel)
#         if cmp(self.__class__, other.__class__):
#             return cmp(self.__class__, other.__class__)
#         return cmp(self.operands, other.operands)
    
#     def depth(self):
#         return max([op.depth() for op in self.operands]) + 1
    
#     def __add__(self, other):
#         assert isinstance(other, Kernel)
#         if isinstance(other, SumKernel):
#             return SumKernel(self.operands + other.operands).copy()
#         else:
#             return SumKernel(self.operands + [other]).copy()
            
#     def out_of_bounds(self, constraints):
#         return any([o.out_of_bounds(constraints) for o in self.operands]) 
    
# class ProductKernelFamily(KernelOperatorFamily):
#     def __init__(self, operands):
#         self.operands = operands
        
#     def from_param_vector(self, params):
#         start = 0
#         ops = []
#         for o in self.operands:
#             end = start + o.num_params()
#             ops.append(o.from_param_vector(params[start:end]))
#             start = end
#         return ProductKernel(ops)
    
#     def num_params(self):
#         return sum([e.num_params() for e in self.operands])
    
#     def pretty_print(self):
#         op = colored(' x ', self.depth())
#         return colored('( ', self.depth()) + \
#             op.join([e.pretty_print() for e in self.operands]) + \
#             colored(' ) ', self.depth())
    
#     def default(self):
#         return ProductKernel([op.default() for op in self.operands])
    
#     def __cmp__(self, other):
#         assert isinstance(other, KernelFamily)
#         if cmp(self.__class__, other.__class__):
#             return cmp(self.__class__, other.__class__)
#         return cmp(self.operands, other.operands)
    
#     def depth(self):
#         return max([op.depth() for op in self.operands]) + 1
        
        
# class ProductKernel(KernelOperator):
#     def __init__(self, operands):
#         self.operands = operands
        
#     def family(self):
#         return ProductKernelFamily([e.family() for e in self.operands])
        
#     def pretty_print(self):
#         #### TODO - Should this call the family method?
#         op = colored(' x ', self.depth())
#         return colored('( ', self.depth()) + \
#             op.join([e.pretty_print() for e in self.operands]) + \
#             colored(' ) ', self.depth())

#     def latex_print(self):
#         return ' \\times '.join([e.latex_print() for e in self.operands])
            
#     def __repr__(self):
#         return 'ProductKernel(%s)' % \
#             ('[ ' + ', '.join([o.__repr__() for o in self.operands]) + ' ]')              
    
#     def gpml_kernel_expression(self):
#         return '{@covProd, {%s}}' % ', '.join(e.gpml_kernel_expression() for e in self.operands)
    
#     def copy(self):
#         return ProductKernel([e.copy() for e in self.operands])

#     def param_vector(self):
#         return np.concatenate([e.param_vector() for e in self.operands])
        
#     def effective_params(self):
#         '''The scale of a product of kernels is over parametrised'''
#         return sum([o.effective_params() for o in self.operands]) - (len(self.operands) - 1)
        
#     def default_params_replaced(self, sd=1, data_shape=None):
#         '''Returns the parameter vector with any default values replaced with random Gaussian'''
#         return np.concatenate([o.default_params_replaced(sd=sd, data_shape=data_shape) for o in self.operands])
    
#     def __cmp__(self, other):
#         assert isinstance(other, Kernel)
#         if cmp(self.__class__, other.__class__):
#             return cmp(self.__class__, other.__class__)
#         return cmp(self.operands, other.operands)
    
#     def depth(self):
#         return max([op.depth() for op in self.operands]) + 1
    
#     def __mul__(self, other):
#         assert isinstance(other, Kernel)
#         if isinstance(other, ProductKernel):
#             return ProductKernel(self.operands + other.operands).copy()
#         else:
#             return ProductKernel(self.operands + [other]).copy()
            
#     def out_of_bounds(self, constraints):
#         return any([o.out_of_bounds(constraints) for o in self.operands])
    
#     @property
#     def output_variance(self):
#         return sum([e.output_variance for e in self.operands])  

# #### FIXME - Sort out the naming of the two functions below            
# def base_kernels(ndim=1, base_kernel_names='SE'):
#     '''
#     Generator of all base kernels for a certain dimensionality of data
#     '''
#     for dim in range(ndim):
#         for fam in base_kernel_families(base_kernel_names):
#             yield MaskKernel(ndim, dim, fam.default())
 
# def base_kernel_families(base_kernel_names):
#     '''
#     Generator of all base kernel families.
#     '''
#     for family in [SqExpKernelFamily(), \
#                    ConstKernelFamily(), \
#                    PureLinKernelFamily(), \
#                    CosineKernelFamily(), \
#                    SpectralKernelFamily(), \
#                    FourierKernelFamily(), \
#                    NoiseKernelFamily()]:
#         if family.id_name() in base_kernel_names.split(','):
#             yield family
   
# #### FIXME - Do the two functions below get called ever?        
# def test_kernels(ndim=1):
#     '''
#     Generator of a subset of base kernels for testing
#     '''
#     for dim in range(ndim):
#         for k in test_kernel_families():
#             yield MaskKernel(ndim, dim, k) 
         
# def test_kernel_families():
#     '''
#     Generator of all base kernel families
#     '''
#     yield SqExpKernelFamily().default()
#     #yield SqExpPeriodicKernelFamily().default() 
#     #yield RQKernelFamily().default()       


# # This removes additively idempotent redundancy        
# def collapse_const_sums(kernel):
#     '''Replaces sums of constants with a single constant'''
#     #### FIXME - This is a bit of a shunt for the periodic kernel - probably somehow fits with the grammar.canonical
#     if isinstance(kernel, BaseKernel):
#         return kernel.copy()
#     elif isinstance(kernel, MaskKernel):
#         return MaskKernel(kernel.ndim, kernel.active_dimension, collapse_const_sums(kernel.base_kernel))
#     elif isinstance(kernel, ChangePointTanhKernel):
#         canop = [collapse_const_sums(o) for o in kernel.operands]
#         return ChangePointTanhKernel(kernel.location, kernel.steepness, canop)
#     elif isinstance(kernel, ChangeBurstTanhKernel):
#         canop = [collapse_const_sums(o) for o in kernel.operands]
#         return ChangeBurstTanhKernel(kernel.location, kernel.steepness, kernel.width, kernel.sf, canop)
#     elif isinstance(kernel, SumKernel):
#         new_ops = []
#         for op in kernel.operands:
#             op_canon = collapse_const_sums(op)
#             if isinstance(op_canon, SumKernel):
#                 new_ops += op_canon.operands
#             elif not isinstance(op_canon, NoneKernel):
#                 new_ops.append(op_canon)
#         # Check for multiple const kernels
#         new_ops_wo_multi_const = []
#         sf = 0
#         for op in new_ops:
#             if isinstance(op, MaskKernel) and isinstance(op.base_kernel, ConstKernel):
#                 sf += np.exp(2*op.base_kernel.output_variance)
#             elif isinstance(op, ConstKernel):
#                 sf += np.exp(2*op.output_variance)
#             else:
#                 new_ops_wo_multi_const.append(op)
#         if sf > 0:
#             new_ops_wo_multi_const.append(ConstKernel(output_variance=np.log(sf)*0.5))
#         new_ops = new_ops_wo_multi_const
#         if len(new_ops) == 0:
#             return NoneKernel()
#         elif len(new_ops) == 1:
#             return new_ops[0]
#         else:
#             return SumKernel(sorted(new_ops))
#     elif isinstance(kernel, ProductKernel):
#         new_ops = []
#         for op in kernel.operands:
#             op_canon = collapse_const_sums(op)
#             if isinstance(op_canon, ProductKernel):
#                 new_ops += op_canon.operands
#             elif not isinstance(op_canon, NoneKernel):
#                 new_ops.append(op_canon)
#         if len(new_ops) == 0:
#             return NoneKernel()
#         elif len(new_ops) == 1:
#             return new_ops[0]
#         else:
#             return ProductKernel(sorted(new_ops))
#     else:
#         raise RuntimeError('Unknown kernel class:', kernel.__class__)

# def break_kernel_into_summands(k):
#     '''Takes a kernel, expands it into a polynomial, and breaks terms up into a list.
    
#     Mutually Recursive with distribute_products().
#     Always returns a list.
#     '''    
#     # First, recursively distribute all products within the kernel.
#     k_dist = distribute_products(k)
    
#     if isinstance(k_dist, SumKernel):
#         # Break the summands into a list of kernels.
#         return list(k_dist.operands)
#     else:
#         return [k_dist]

# def distribute_products(k):
#     """Distributes products to get a polynomial.
    
#     Mutually recursive with break_kernel_into_summands().
#     Always returns a sumkernel.
#     """

#     if isinstance(k, ProductKernel):
#         # Recursively distribute each of the terms to be multiplied.
#         distributed_ops = [break_kernel_into_summands(op) for op in k.operands]
        
#         # Now produce a sum of all combinations of terms in the products. Itertools is awesome.
#         new_prod_ks = [ProductKernel( prod ) for prod in itertools.product(*distributed_ops)]
#         return SumKernel(new_prod_ks)
    
#     elif isinstance(k, SumKernel):
#         # Recursively distribute each the operands to be summed, then combine them back into a new SumKernel.
#         return SumKernel([subop for op in k.operands for subop in break_kernel_into_summands(op)])
#     elif isinstance(k, ChangePointTanhKernel):
#         return SumKernel([ChangePointTanhKernel(location=k.location, steepness=k.steepness, operands=[op, ZeroKernel()]) for op in break_kernel_into_summands(k.operands[0])] + \
#                          [ChangePointTanhKernel(location=k.location, steepness=k.steepness, operands=[ZeroKernel(), op]) for op in break_kernel_into_summands(k.operands[1])])
#     elif isinstance(k, ChangeBurstTanhKernel):
#         return SumKernel([ChangeBurstTanhKernel(location=k.location, steepness=k.steepness, width=k.width, operands=[op, ZeroKernel()]) for op in break_kernel_into_summands(k.operands[0])] + \
#                          [ChangeBurstTanhKernel(location=k.location, steepness=k.steepness, width=k.width, operands=[ZeroKernel(), op]) for op in break_kernel_into_summands(k.operands[1])])
#     else:
#         # Base case: A kernel that's just, like, a kernel, man.
#         return k
        
# from numpy import nan

# def repr_string_to_kernel(string):
#     """This is defined in this module so that all the kernel class names
#     don't have to have the module name in front of them."""
#     return eval(string)

# class ScoredKernel:
#     '''
#     Wrapper around a kernel with various scores and noise parameter
#     '''
#     def __init__(self, k_opt, nll=nan, laplace_nle=nan, bic_nle=nan, aic_nle=nan, pl2=nan, npll=nan, pic_nle=nan, mae=nan, std_ratio=nan, noise=nan):
#         self.k_opt = k_opt
#         self.nll = nll
#         self.laplace_nle = laplace_nle
#         self.bic_nle = bic_nle
#         self.aic_nle = aic_nle
#         self.pl2 = pl2
#         self.npll = npll
#         self.pic_nle = pic_nle
#         self.mae = mae
#         self.std_ratio = std_ratio
#         self.noise = noise
        
#     #### CAUTION - the default keeps on changing!
#     def score(self, criterion='bic'):
#         return {'bic': self.bic_nle,
#                 'aic': self.aic_nle,
#                 'pl2': self.pl2,
#                 'nll': self.nll,
#                 'laplace': self.laplace_nle,
#                 'npll': self.npll,
#                 'pic': self.pic_nle,
#                 'mae': self.mae
#                 }[criterion.lower()]
                
#     @staticmethod
#     def from_printed_outputs(nll, laplace, BIC, AIC, PL2, npll, PIC, mae, std_ratio, noise=None, kernel=None):
#         return ScoredKernel(kernel, nll, laplace, BIC, AIC, PL2, npll, PIC, mae, std_ratio, noise)
    
#     def __repr__(self):
#         return 'ScoredKernel(k_opt=%s, nll=%f, laplace_nle=%f, bic_nle=%f, aic_nle=%f, pl2=%f, npll=%f, pic_nle=%f, mae=%f, std_ratio=%f, noise=%s)' % \
#             (self.k_opt, self.nll, self.laplace_nle, self.bic_nle, self.aic_nle, self.pl2, self.npll, self.pic_nle, self.mae, self.std_ratio, self.noise)

#     def pretty_print(self):
#         return self.k_opt.pretty_print()

#     def latex_print(self):
#         return self.k_opt.latex_print()

#     @staticmethod	
#     def from_matlab_output(output, kernel_family, ndata):
#         '''Computes Laplace marginal lik approx and BIC - returns scored Kernel'''
#         #### TODO - this check should be within the psd_matrices code
#         if np.any(np.isnan(output.hessian)):
#             laplace_nle = np.nan
#         else:
#             laplace_nle, problems = psd_matrices.laplace_approx_stable_no_prior(output.nll, output.hessian)
#         k_opt = kernel_family.from_param_vector(output.kernel_hypers)
#         BIC = 2 * output.nll + k_opt.effective_params() * np.log(ndata)
#         PIC = 2 * output.npll + k_opt.effective_params() * np.log(ndata)
#         AIC = 2 * output.nll + k_opt.effective_params() * 2
#         PL2 = output.nll / ndata + k_opt.effective_params() / (2 * ndata)
#         return ScoredKernel(k_opt, output.nll, laplace_nle, BIC, AIC, PL2, output.npll, PIC, output.mae, output.std_ratio, output.noise_hyp)	

# def add_random_restarts_single_kernel(kernel, n_rand, sd, data_shape):
#     '''Returns a list of kernels with random restarts for default values'''
#     return [kernel] + list(map(lambda unused : kernel.family().from_param_vector(kernel.default_params_replaced(sd=sd, data_shape=data_shape)), [None] * n_rand))

# def add_random_restarts(kernels, n_rand=1, sd=4, data_shape=None):    
#     '''Augments the list to include random restarts of all default value parameters'''
#     return [k_rand for kernel in kernels for k_rand in add_random_restarts_single_kernel(kernel, n_rand, sd, data_shape)]

# def add_jitter(kernels, sd=0.1, data_shape=None):    
#     '''Adds random noise to all parameters - empirically observed to help when optimiser gets stuck'''
#     #### FIXME - this is ok for log transformed parameters - for other parameters the scale of jitter might be completely off
#     return [k.family().from_param_vector(k.param_vector() + np.random.normal(loc=0., scale=sd, size=k.param_vector().size)) for k in kernels]
