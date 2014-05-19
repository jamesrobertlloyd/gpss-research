"""
Translates kernel expressions into natural language descriptions
The main assumptions made by this module at the moment are
 - 1d input
 - A limited number of possible base kernels and operators
 - Probably others

@authors: James Robert Lloyd (jrl44@cam.ac.uk)
          
September 2013          
"""

import numpy as np
import scipy.stats
import time
import warnings

import flexible_function as ff

to_19 = ( 'zero',  'one',   'two',  'three', 'four',   'five',   'six',
          'seven', 'eight', 'nine', 'ten',   'eleven', 'twelve', 'thirteen',
          'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen' )

tens  = ( 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety')

denom = ( '',
          'thousand',     'million',         'billion',       'trillion',       'quadrillion',
          'quintillion',  'sextillion',      'septillion',    'octillion',      'nonillion',
          'decillion',    'undecillion',     'duodecillion',  'tredecillion',   'quattuordecillion',
          'sexdecillion', 'septendecillion', 'octodecillion', 'novemdecillion', 'vigintillion' )

def _convert_nn(val):
    if val < 20:
        return to_19[val]
    for (dcap, dval) in ((k, 20 + (10 * v)) for (v, k) in enumerate(tens)):
        if dval + 10 > val:
            if val % 10:
                return dcap + '-' + to_19[val % 10]
            return dcap

def _convert_nnn(val):
    word = ''
    (mod, rem) = (val % 100, val // 100)
    if rem > 0:
        word = to_19[rem] + ' hundred'
    if mod > 0:
        word = word + ' and '
    if mod > 0:
        word = word + _convert_nn(mod)
    return word

def english_number(val):
    if val < 100:
        return _convert_nn(val)
    if val < 1000:
        return _convert_nnn(val)
    for (didx, dval) in ((v - 1, 1000 ** v) for v in range(len(denom))):
        if dval > val:
            mod = 1000 ** didx
            l = val // mod
            r = val - (l * mod)
            ret = _convert_nnn(l) + ' ' + denom[didx]
            if r > 0:
                ret = ret + ', ' + english_number(r)
            return ret.strip()
            
poly_names = ['linear', 'quadratic', 'cubic', 'quartic', 'quintic']
ordinal_numbers = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eigth', 'ninth', 'tenth']

def english_length(val, unit):
    if unit == '':
        return '%f' % val
    elif unit == 'year':
        if val > 0.75:
            return '%0.1f years' % val
        elif val > 2.0 / 12:
            return '%0.1f months' % (val * 12)
        elif val > 2.0 / 52:
            return '%0.1f weeks' % (val * 52)
        elif val > 2.0 / 365:
            return '%0.1f days' % (val * 365)
        elif val > 2.0 / (24 * 365):
            return '%0.1f hours' % (val * (24 * 365))
        elif val > 2.0 / (60 * 24 * 365):
            return '%0.1f minutes' % (val * (60 * 24 * 365))
        else: 
            return '%0.1f seconds' % (val * (60 * 60 * 24 * 365))
    else:
        warnings.warn('I do not know about this unit of measurement : %s' % unit)
        return 'Unrecognised format'
        
english_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

def english_point(val, unit, X):
    #### TODO - be clever about different dimensions?
    unit_range = np.max(X) - np.min(X)
    if unit == '':
        return '%f' % val
    elif unit == 'year':
        time_val = time.gmtime((val - 1970)*365*24*60*60)
        if unit_range > 20:
            return '%4d' % time_val.tm_year
        elif unit_range > 2:
            return '%s %4d' % (english_months[time_val.tm_mon-1], time_val.tm_year)
        elif unit_range > 2.0 / 12:
            return '%02d %s %4d' % (time_val.tm_mday, english_months[time_val.tm_mon-1], time_val.tm_year)
        else:
            return '%02d:%02d:%02d %02d %s %4d' % (time_val.tm_hour, time_val.tm_min, time_val.tm_sec, time_val.tm_mday, english_months[time_val.tm_mon-1], time_val.tm_year)
    else:
        warnings.warn('I do not know about this unit of measurement : %s' % unit)
        return 'Unrecognised format'

def interval_intersection(int1, int2):
    '''Finds the intersection of two intervals'''
    if ((int1[0] <= int2[0]) and (int1[1] >= int2[0])) or ((int1[1] >= int2[1]) and (int1[0] <= int2[1])):
        return (np.max([int1[0], int2[0]]), np.min([int1[1], int2[1]]))
    else:
        return none

def find_region_of_influence(k, intervals=[(-np.Inf, np.Inf)]):
    '''
    Strips changepoint type operators to understand what region of input the base kernel acts upon
    Returns (intervals, base_kernel) where intervals is a list of tuple intervals
    '''
    #### TODO - what extensions will this need in multi-d?
    if isinstance(k, ff.ChangePointKernel):
        if isinstance(k.operands[0], ff.ZeroKernel) or isinstance(k.operands[0], ff.NoneKernel):
            local_intervals = [(k.location, np.Inf)]
            base_kernel = k.operands[1]
        else:
            local_intervals = [(-np.Inf, k.location)]
            base_kernel = k.operands[0]
    elif isinstance(k, ff.ChangeWindowKernel):
        if isinstance(k.operands[0], ff.ZeroKernel) or isinstance(k.operands[0], ff.NoneKernel):
            local_intervals = [(k.location-np.exp(k.width)/2, k.location+np.exp(k.width)/2)]
            base_kernel = k.operands[1]
        else:
            local_intervals = [(-np.Inf, k.location-np.exp(k.width)/2), (k.location+np.exp(k.width)/2, np.Inf)]
            base_kernel = k.operands[0]
    elif isinstance(k, ff.ProductKernel) and any(op.is_operator for op in k.operands):
        # Product contains an operator - find it
        for (i, op) in enumerate(k.operands):
            if op.is_operator:
                # Found an operator - place all other kernels within it - resulting in the operator at the top of the kernel
                other_kernels = k.operands[:i] + k.operands[(i+1):]
                if isinstance(op.operands[0], ff.ZeroKernel) or isinstance(op.operands[0], ff.NoneKernel):
                    for other_k in other_kernels:
                        op.operands[1] *= other_k
                    op.operands[1] = op.operands[1].canonical()
                else:
                    for other_k in other_kernels:
                        op.operands[0] *= other_k
                    op.operands[0] = op.operands[0].canonical()
                return find_region_of_influence(op, intervals)
    elif isinstance(k, ff.Kernel):
        return (intervals, k)
    else:
        raise RuntimeError('I''m not intelligent enough to find the region of influence of kernel', k.__class__)
    # If the function has got this far then k was a changepoint operator - need to intervals with local_intervals
    new_intervals = []
    for int1 in intervals:
        for int2 in local_intervals:
            if not ((int1[1] < int2[0]) or (int1[0] > int2[1])):
                # Intersection non-empty
                new_intervals.append((np.max([int1[0], int2[0]]), np.min([int1[1], int2[1]])))
    # Note that new_intervals may now be empty but we should recurse to return a kernel in a standard form
    return find_region_of_influence(base_kernel, new_intervals)
    
def translate_parametric_window(X, unit='', lin_count=0, exp_count=0, lin_location=None, exp_rate=None, quantity='standard deviation', component='function', qualifier=''):
    '''
    Translates the effect on standard deviation/amplitude/... of parametric terms (at the time of writing this is just Lin and Exp)
    '''
    summary = ''
    description = ''
    extrap_description = ''
    if (lin_count > 0) and (exp_count == 0):
        description += 'The %s of the %s ' % (quantity, component)
        extrap_description += 'The %s of the %s is assumed to continue to ' % (quantity, component)
        if lin_count == 1:
            if lin_location < np.min(X):
                summary += 'with %slinearly increasing %s' % (qualifier, quantity)
                description += 'increases %slinearly' % qualifier
                extrap_description += 'increase %slinearly' % qualifier
            elif lin_location > np.max(X):
                summary += 'with %slinearly decreasing %s' % (qualifier, quantity)
                description += 'decreases %slinearly' % qualifier
                extrap_description += 'decrease %slinearly until %s after which the %s of the %s is assumed to start increasing %slinearly' % (qualifier, english_point(lin_location, unit, X), quantity, component, qualifier)
            else:
                summary += 'with %s increasing %slinearly away from %s' % (quantity, qualifier, english_point(lin_location, unit, X))
                description += 'increases %slinearly away from %s' % (qualifier, english_point(lin_location, unit, X))
                extrap_description += 'increase %slinearly' % qualifier
        elif lin_count <= len(poly_names):
            summary += 'with %s%sly varying %s' % (qualifier, poly_names[lin_count-1], quantity)
            description += 'varies %s%sly' % (qualifier, poly_names[lin_count-1])
            extrap_description += 'vary %s%sly' % (qualifier, poly_names[lin_count-1])
        else:
            summary += 'with %s given %sby a polynomial of degree %d' % (qualifier, quantity, lin_count)
            description += 'is given %sby a polynomial of degree %d' % (qualifier, lin_count)
            extrap_description += '%s vary according to a polynomial of degree %d' % (qualifier, lin_count)
    elif (exp_count > 0) and (lin_count == 0):
        description += 'The %s of the %s ' % (quantity, component)
        extrap_description += 'The %s of the %s is assumed to continue to ' % (quantity, component)
        if exp_rate > 0:
            summary = 'with exponentially %sincreasing %s' % (qualifier, quantity)
            description += 'increases %sexponentially' % qualifier
            extrap_description += 'increase %sexponentially' % qualifier
        else:
            summary = 'with exponentially %sdecreasing %s' % (qualifier, quantity)
            description += 'decreases %sexponentially' % (qualifier)
            extrap_description += 'decrease %sexponentially' % (qualifier)
    else:
        #### TODO - this is the product of lin and exp - explanantions can be made nicer by looking for turning points
        if exp_rate > 0:
            summary += 'with %s given %sby a product of a polynomial of degree %d and an increasing exponential function' % (quantity, qualifier, lin_count)
            description += 'The %s of the %s is given %sby the product of a polynomial of degree %d and an increasing exponential function' % (quantity, component, qualifier, lin_count)
            extrap_description += 'The %s of the %s is assumed to continue to be given %sby the product of a polynomial of degree %d and an increasing exponential function' % (quantity, component, qualifier, lin_count)
        else:
            summary += 'with %s given %sby a product of a polynomial of degree %d and a decreasing exponential function' % (quantity, qualifier, lin_count)
            description += 'The %s of the %s is given %sby the product of a polynomial of degree %d and a decreasing exponential function' % (quantity, component, qualifier, lin_count)
            extrap_description += 'The %s of the %s is assumed to continue to be given %sby the product of a polynomial of degree %d and a decreasing exponential function' % (quantity, component, qualifier, lin_count)
    return (summary, description, extrap_description)
                    
def translate_product(prod, X, monotonic, gradient, unit=''):
    '''
    Translates a product of base kernels
    '''
    #### TODO
    # - assumes 1d
    # - IMTLin not dealt with
    # - Step function not dealt with
    # Strip masks and produce list of base kernels in product
    prod = prod.canonical()
    if not prod.is_operator:
        kernels = [prod]
    else:
        kernels = prod.operands
    # Initialise
    descriptions = []
    extrap_descriptions = []
    lengthscale = np.Inf
    exp_rate = 0
    los_count = 0 # Local smooth components
    lin_count = 0
    per_count = 0
    cos_count = 0
    exp_count = 0
    noi_count = 0 # Noise components
    imt_count = 0 # Integrated Matern components
    unk_count = 0 # 'Unknown' kernel function
    per_kernels = []
    cos_kernels = []
    min_period = np.Inf
    lin_location = None
    # Count kernels and calculate a variety of summary quantities
    for k in kernels:
        if isinstance(k, ff.SqExpKernel):
            los_count += 1
            lengthscale = -0.5 * np.log(np.exp(-2*lengthscale) + np.exp(-2*k.lengthscale))
        elif isinstance(k, ff.LinearKernel):
            lin_count += 1
            lin_location = k.location
        elif isinstance(k, ff.PeriodicKernel):
            per_count += 1
            per_kernels.append(k)
            min_period = np.min([np.exp(k.period), min_period])
        #elif isinstance(k, ff.CosineKernel):
        #    cos_count += 1
        #    cos_kernels.append(k)
        #    min_period = np.min([np.exp(k.period), min_period])
        #elif isinstance(k, ff.ExpKernel):
        #    exp_count += 1
        #    exp_rate += k.rate
        elif isinstance(k, ff.NoiseKernel):
            noi_count += 1
        elif not isinstance(k, ff.ConstKernel):
            # Cannot deal with whatever type of kernel this is
            unk_count +=1
    lengthscale = np.exp(lengthscale)
    domain_range = np.max(X) - np.min(X)
    # Now describe the properties of this product of kernels
    if (unk_count > 0):
        summary = 'This simple AI is not capable of describing the component who''s python representation is %s' % prod.__repr__()
        descriptions.append('This simple AI is not capable of describing the component who''s python representation is %s' % prod.__repr__())
        extrap_descriptions.append('This simple AI is not capable of describing the component who''s python representation is %s' % prod.__repr__())
        warnings.warn('I''m not intelligent enough to describe this kernel in natural language : %s' % prod.__repr__())
    elif (noi_count > 0):
        summary = 'Uncorrelated noise'
        descriptions.append('This component models uncorrelated noise')  
        extrap_descriptions.append('This component assumes the uncorrelated noise will continue indefinitely')  
        if lin_count + exp_count > 0:
            (var_summary, var_description, var_extrap_description) = translate_parametric_window(X, unit=unit, lin_count=lin_count, exp_count=exp_count, lin_location=lin_location, exp_rate=exp_rate, quantity='standard deviation', component='noise')
            descriptions.append(var_description)
            extrap_descriptions.append(var_extrap_description)
            summary += ' %s' % var_summary
    elif (los_count == 0) and (lin_count == 0) and (per_count == 0) and (cos_count == 0) and (imt_count == 0):
        summary = 'A constant'
        descriptions.append('This component is constant')   
        extrap_descriptions.append('This component is assumed to stay constant')      
    elif (los_count > 0) and (per_count == 0) and (cos_count == 0) and (imt_count == 0):
        # This is a pure smooth and local component (possibly with parametric variance)
        if lengthscale > 0.5 * domain_range:
            if monotonic == 1:
                summary = 'A very smooth monotonically increasing function'
                descriptions.append('This component is a very smooth and monotonically increasing function')
            elif monotonic == -1:
                summary = 'A very smooth monotonically decreasing function'
                descriptions.append('This component is a very smooth and monotonically decreasing function')
            else:
                summary = 'A very smooth function'
                descriptions.append('This component is a very smooth function')
            extrap_descriptions.append('This component is assumed to continue very smoothly but is also assumed to be stationary so its distribution will eventually return to the prior')
        elif lengthscale < domain_range * 0.005:
            summary = 'A rapidly varying smooth function'
            descriptions.append('This component is a rapidly varying but smooth function with a typical lengthscale of %s' % english_length(lengthscale, unit))
            extrap_descriptions.append('This component is assumed to continue smoothly but its distribution is assumed to quickly return to the prior')
        else:
            if monotonic == 1:
                summary = 'A smooth monotonically increasing function'
                descriptions.append('This component is a smooth and monotonically increasing function with a typical lengthscale of %s' % english_length(lengthscale, unit))
            elif monotonic == -1:
                summary = 'A smooth monotonically decreasing function'
                descriptions.append('This component is a smooth and monotonically decreasing function with a typical lengthscale of %s' % english_length(lengthscale, unit))
            else:
                summary = 'A smooth function'
                descriptions.append('This component is a smooth function with a typical lengthscale of %s' % english_length(lengthscale, unit))
            extrap_descriptions.append('This component is assumed to continue smoothly but is also assumed to be stationary so its distribution will return to the prior')
        extrap_descriptions.append('The prior distribution places mass on smooth functions with a marginal mean of zero and a typical lengthscale of %s' % english_length(lengthscale, unit))
        extrap_descriptions.append('[This is a placeholder for a description of how quickly the posterior will start to resemble the prior]')
        if lin_count + exp_count > 0:
            # Parametric variance
            (var_summary, var_description, var_extrap_description) = translate_parametric_window(X, unit=unit, lin_count=lin_count, exp_count=exp_count, lin_location=lin_location, exp_rate=exp_rate, quantity='marginal standard deviation', component='function')
            descriptions.append(var_description)
            extrap_descriptions.append(var_extrap_description)
            summary += ' %s' % var_summary
    elif (los_count == 0) and (lin_count > 0) and (per_count == 0) and (cos_count == 0) and (imt_count == 0):
        # This is a pure polynomial component
        if lin_count == 1:
            if gradient > 0:
                summary = 'A linearly increasing function'
                descriptions.append('This component is linearly increasing')
                extrap_descriptions.append('This component is assumed to continue to increase linearly')
            else:
                summary = 'A linearly decreasing function'
                descriptions.append('This component is linearly decreasing')
                extrap_descriptions.append('This component is assumed to continue to decrease linearly')
        elif lin_count <= len(poly_names):
            # I know a special name for this type of polynomial
            summary = 'A %s polynomial' % poly_names[lin_count-1]
            descriptions.append('This component is a %s polynomial' % poly_names[lin_count-1])
            extrap_descriptions.append('This component is assumed to conitnue as a %s polynomial' % poly_names[lin_count-1])
        else:
            summary = 'A polynomial of degree %d' % lin_count
            descriptions.append('This component is a polynomial of degree %d' % lin_count)
            extrap_descriptions.append('This component is assumed to continue as a polynomial of degree %d' % lin_count)
    elif (per_count > 0) or (cos_count > 0) and (imt_count == 0):
        if ((per_count == 1) and (cos_count == 0)) or ((per_count == 0) and (cos_count == 1)):
            k = per_kernels[0] if per_count == 1 else cos_kernels[0]
            if (lin_count + exp_count == 0) and (los_count == 0):
                # Pure periodic
                summary = 'A '
                main_description = 'This component is '
                extrap_description = 'This component is assumed to continue '
                if per_count == 1:
                    summary += 'periodic function '
                    main_description += 'periodic '
                    extrap_description += 'periodically '
                else:
                    summary += 'sinusoidal function '
                    main_description += 'sinusoidal '
                    extrap_description += 'sinusoidally '
                summary += 'with a period of %s' % english_length(np.exp(k.period), unit)
                main_description += 'with a period of %s' % english_length(np.exp(k.period), unit)
                extrap_description += 'with a period of %s' % english_length(np.exp(k.period), unit)
                descriptions.append(main_description)
                extrap_descriptions.append(extrap_description)
                if per_count == 1:
                    #### FIXME - this correspondence is only approximate
                    per_lengthscale = 0.5*np.exp(k.lengthscale + k.period)/np.pi # This definition of lengthscale fits better with local smooth kernels
                    if k.lengthscale > 2:
                        descriptions.append('The shape of this function within each period is very smooth and resembles a sinusoid')
                    else:
                        descriptions.append('The shape of this function within each period has a typical lengthscale of %s' % english_length(per_lengthscale, unit))
            elif (lin_count + exp_count == 0) and (los_count > 0):
                # Approx periodic
                lower_per = 1.0 / (np.exp(-k.period) + scipy.stats.norm.isf(0.25) / lengthscale)
                upper_per = 1.0 / (np.exp(-k.period) - scipy.stats.norm.isf(0.25) / lengthscale)
                extrap_descriptions.append('This component is assumed to continue to be approximately periodic')
                if upper_per < 0:
                    # This will probably look more like noise
                    summary = 'A very approximately '
                    main_description = 'This component is very approximately '
                    if per_count == 1:
                        summary += 'periodic function '
                        main_description += 'periodic '
                    else:
                        summary += 'sinusoidal function '
                        main_description += 'sinusoidal '
                    summary += 'with a period of %s' % english_length(np.exp(k.period), unit)
                    main_description += 'with a period of %s' % english_length(np.exp(k.period), unit)
                    descriptions.append(main_description)
                    descriptions.append('Across periods the shape of this function varies smoothly with a typical lengthscale of %s' % english_length(lengthscale, unit))
                    descriptions.append('Since this lengthscale is small relative to the period this component may more closely resemble a non-periodic smooth function')
                    extrap_descriptions.append('The shape of the function is assumed to vary smoothly between periods but will quickly return to the prior')
                else:
                    summary = 'An approximately '
                    main_description = 'This component is approximately '
                    if per_count == 1:
                        summary += 'periodic function '
                        main_description += 'periodic '
                    else:
                        summary += 'sinusoidal function '
                        main_description += 'sinusoidal '
                    summary += 'with a period of %s' % english_length(np.exp(k.period), unit)
                    main_description += 'with a period of %s' % english_length(np.exp(k.period), unit)
                    descriptions.append(main_description)
                    if lengthscale > 0.5 * domain_range:
                        descriptions.append('Across periods the shape of this function varies very smoothly')
                        extrap_descriptions.append('The shape of the function is assumed to vary very smoothly between periods but will eventually return to the prior')
                    else:
                        descriptions.append('Across periods the shape of this function varies smoothly with a typical lengthscale of %s' % english_length(lengthscale, unit))  
                        extrap_descriptions.append('The shape of the function is assumed to vary smoothly between periods but will return to the prior') 
                    if per_count == 1:
                        per_lengthscale = 0.5*np.exp(k.lengthscale + k.period)/np.pi # This definition of lengthscale fits better with local smooth kernels
                        if k.lengthscale > 2:
                            descriptions.append('The shape of this function within each period is very smooth and resembles a sinusoid')
                        else:
                            descriptions.append('The shape of this function within each period has a typical lengthscale of %s' % english_length(per_lengthscale, unit))
                extrap_descriptions.append('The prior is entirely uncertain about the phase of the periodic function')
                extrap_descriptions.append('Consequently the pointwise posterior will appear to lose its periodicity, but this merely reflects the uncertainty in the shape and phase of the function')
                extrap_descriptions.append('[This is a placeholder for a description of how quickly the posterior will start to resemble the prior]')
            elif (lin_count + exp_count > 0) and (los_count == 0):
                # Pure periodic but with changing amplitude
                summary = 'A '
                main_description = 'This component is '
                extrap_description = 'This component is assumed to continue '
                if per_count == 1:
                    summary += 'periodic function '
                    main_description += 'periodic '
                    extrap_description += 'periodically '
                else:
                    summary += 'sinusoidal function '
                    main_description += 'sinusoidal '
                    extrap_description += 'sinusoidally '
                summary += 'with a period of %s' % english_length(np.exp(k.period), unit)
                main_description += 'with a period of %s but with varying amplitude' % english_length(np.exp(k.period), unit)
                extrap_description += 'with a period of %s but with varying amplitude' % english_length(np.exp(k.period), unit)
                descriptions.append(main_description)
                extrap_descriptions.append(extrap_description)
                (var_summary, var_description, var_extrap_description) = translate_parametric_window(X, unit=unit, lin_count=lin_count, exp_count=exp_count, lin_location=lin_location, exp_rate=exp_rate, quantity='amplitude', component='function')
                descriptions.append(var_description)
                extrap_descriptions.append(var_extrap_description)
                summary += ' but %s' % var_summary
                if per_count == 1:
                    per_lengthscale = 0.5*np.exp(k.lengthscale + k.period)/np.pi # This definition of lengthscale fits better with local smooth kernels
                    if k.lengthscale > 2:
                        descriptions.append('The shape of this function within each period is very smooth and resembles a sinusoid')
                    else:
                        descriptions.append('The shape of this function within each period has a typical lengthscale of %s' % english_length(per_lengthscale, unit))
            else:
                lower_per = 1.0 / (np.exp(-k.period) + scipy.stats.norm.isf(0.25) / lengthscale)
                upper_per = 1.0 / (np.exp(-k.period) - scipy.stats.norm.isf(0.25) / lengthscale)
                extrap_descriptions.append('This component is assumed to continue to be approximately periodic')
                if upper_per < 0:
                    # This will probably look more like noise
                    summary = 'A very approximately '
                    main_description = 'This component is very approximately '
                    if per_count == 1:
                        summary += 'periodic function '
                        main_description += 'periodic '
                    else:
                        summary += 'sinusoidal function '
                        main_description += 'sinusoidal '
                    summary += 'with a period of %s' % english_length(np.exp(k.period), unit)
                    main_description += 'with a period of %s and varying marginal standard deviation' % english_length(np.exp(k.period), unit)
                    descriptions.append(main_description)
                    descriptions.append('Across periods the shape of this function varies smoothly with a typical lengthscale of %s' % english_length(lengthscale, unit))
                    descriptions.append('Since this lengthscale is small relative to the period this component may more closely resemble a non-periodic smooth function')
                    extrap_descriptions.append('The shape of the function is assumed to vary smoothly between periods but will quickly return to the prior')
                    (var_summary, var_description, var_extrap_description) = translate_parametric_window(X, unit=unit, lin_count=lin_count, exp_count=exp_count, lin_location=lin_location, exp_rate=exp_rate, quantity='marginal standard deviation', component='function')
                    descriptions.append(var_description)
                    extrap_descriptions.append(var_extrap_description)
                    summary += ' and %s' % var_summary
                else:
                    summary = 'An approximately '
                    main_description = 'This component is approximately '
                    if per_count == 1:
                        summary += 'periodic function '
                        main_description += 'periodic '
                    else:
                        summary += 'sinusoidal function '
                        main_description += 'sinusoidal '
                    summary += 'with a period of %s' % english_length(np.exp(k.period), unit)
                    main_description += 'with a period of %s and varying amplitude' % english_length(np.exp(k.period), unit)
                    descriptions.append(main_description)
                    if lengthscale > 0.5 * domain_range:
                        descriptions.append('Across periods the shape of this function varies very smoothly')
                        extrap_descriptions.append('The shape of the function is assumed to vary very smoothly between periods but will eventually return to the prior')
                    else:
                        descriptions.append('Across periods the shape of this function varies smoothly with a typical lengthscale of %s' % english_length(lengthscale, unit))   
                        extrap_descriptions.append('The shape of the function is assumed to vary smoothly between periods but will return to the prior') 
                    (var_summary, var_description, var_extrap_description) = translate_parametric_window(X, unit=unit, lin_count=lin_count, exp_count=exp_count, lin_location=lin_location, exp_rate=exp_rate, quantity='amplitude', component='function', qualifier='')#approximately ')
                    descriptions.append(var_description)
                    extrap_descriptions.append(var_extrap_description)
                    summary += ' and %s' % var_summary
                    if per_count == 1:
                        per_lengthscale = 0.5*np.exp(k.lengthscale + k.period)/np.pi # This definition of lengthscale fits better with local smooth kernels
                        if k.lengthscale > 2:
                            descriptions.append('The shape of this function within each period is very smooth and resembles a sinusoid')
                        else:
                            descriptions.append('The shape of this function within each period has a typical lengthscale of %s' % english_length(per_lengthscale, unit))
                extrap_descriptions.append('The prior is entirely uncertain about the phase of the periodic function')
                extrap_descriptions.append('Consequently the pointwise posterior will appear to lose its periodicity, but this merely reflects the uncertainty in the shape and phase of the function')
                extrap_descriptions.append('[This is a placeholder for a description of how quickly the posterior will start to resemble the prior]')
        else: # Several periodic components
            if los_count > 0:
                summary = 'An approximate product of'
            else:
                summary = 'A product of'
            main_description = 'This component is a product of'
            extrap_description = 'This component is assumed to continue as a product of'
            #if los_count > 0:
            #    main_description += 'approximately '
            #main_description += 'like the product of'
            if per_count == 1:
                summary += ' a periodic function'
                main_description += ' a periodic function'
                extrap_description += ' a periodic function'
            elif per_count > 1:
                summary += ' several periodic functions'
                main_description += ' several periodic functions'
                extrap_description += ' several periodic functions'
            if (per_count > 0) and (cos_count > 0):
                summary += ' and'
                main_description += ' and'
                extrap_description += ' and'
            if cos_count == 1:
                summary += ' a sinusoid'
                main_description += ' a sinusoid'
                extrap_description += ' a sinusoid'
            elif cos_count > 1:
                summary += ' several sinusoids'
                main_description += ' several sinusoids'
                extrap_description += ' several sinusoids'
            descriptions.append(main_description)
            extrap_descriptions.append(main_description)
            if los_count > 0:
                descriptions.append('Across periods the shape of this function varies smoothly with a typical lengthscale of %s' % english_length(lengthscale, unit))
                extrap_descriptions.append('Across periods the shape of this function is assumed to continue to vary smoothly but will return to the prior')
                extrap_descriptions.append('The prior is entirely uncertain about the phase of the periodic functions')
                extrap_descriptions.append('Consequently the pointwise posterior will appear to lose its periodicity, but this merely reflects the uncertainty in the shape and phase of the functions')
                extrap_descriptions.append('[This is a placeholder for a description of how quickly the posterior will start to resemble the prior]')
            if lin_count + exp_count > 0:
                qualifier = ''#approximately ' if (los_count > 0) else ''
                (var_summary, var_description, var_extrap_description) = translate_parametric_window(X, unit=unit, lin_count=lin_count, exp_count=exp_count, lin_location=lin_location, exp_rate=exp_rate, quantity='amplitude', component='function', qualifier=qualifier)
                descriptions.append(var_description)
                extrap_descriptions.append(var_extrap_description)
                summary += ' and %s' % var_summary
            for (i, k) in enumerate(per_kernels):
                description = 'The '
                if i <= len(ordinal_numbers):
                    if len(per_kernels) > 1:
                        description += '%s ' % ordinal_numbers[i]
                else:
                    description += '%dth ' % (i+1)
                description += 'periodic function has a period of %s' % english_length(np.exp(k.period), unit)
                descriptions.append(description)
                #### FIXME - this correspondence is only approximate
                per_lengthscale = 0.5*np.exp(k.lengthscale + k.period)/np.pi # This definition of lengthscale fits better with local smooth kernels
                if k.lengthscale > 2:
                    descriptions.append('The shape of this function within each period is very smooth and resembles a sinusoid')
                else:
                    descriptions.append('The shape of this function within each period has a typical lengthscale of %s' % english_length(per_lengthscale, unit))
            for (i, k) in enumerate(cos_kernels):
                if i <= len(ordinal_numbers):
                    if len(cos_kernels) > 1:
                        descriptions.append('The %s sinusoid has a period of %s' % (ordinal_numbers[i], english_length(np.exp(k.period), unit)))
                    else:
                        descriptions.append('The sinusoid has a period of %s' % english_length(np.exp(k.period), unit))
                else:
                    descriptions.append('The %dth sinusoid has a period of %s' % (i+1, english_length(np.exp(k.period), unit)))
    else:
        descriptions.append('This simple AI is not capable of describing the component who''s python representation is %s' % prod.__repr__())
        warnings.warn('I''m not intelligent enough to describe this kernel in natural language : %s', prod.__repr__())
    # Return a list of sentences
    return (summary, descriptions, extrap_descriptions)
    
def translate_interval(interval, unit, X):
    '''Describe a 1d interval, including relevant prepositions'''
    if interval == (-np.Inf, np.Inf):
        return 'to the entire input domain'
    elif interval[0] == -np.Inf:
        return 'until %s' % english_point(interval[1], unit, X)
    elif interval[1] == np.Inf:
        return 'from %s onwards' % english_point(interval[0], unit, X)
    else:
        return 'from %s until %s' % (english_point(interval[0], unit, X), english_point(interval[1], unit, X))

def translate_additive_component(k, X, monotonic, gradient, unit):
    '''
    Expects a kernel that is a single component after calling flexiblekernel.distribute_products
    Translates this into natural language or explains that why it's translation abilities aren't up to the job
    '''
    #### TODO
    #     - Discuss steepness of changepoints when there is only one / the form is simple enough
    k = k.canonical() # Just in case
    (intervals, k) = find_region_of_influence(k)
    # Calculate the description of the changepoint free part of the kernel
    (summary, descriptions, extrap_descriptions) = translate_product(k, X, monotonic, gradient, unit)
    # Describe the intervals this kernel acts upon
    intervals = sorted(intervals)
    if len(intervals) == 0:
        summary += '. The combination of changepoint operators is such that this simple AI cannot describe where this component acts; please see visual output or upgrade me'
        interval_description = 'The combination of changepoint operators is such that this simple AI cannot describe where this component acts; please see visual output or upgrade me'
    elif len(intervals) == 1:
        if not intervals == [(-np.Inf, np.Inf)]: 
            summary += '. This function applies %s' % translate_interval(intervals[0], unit, X)
        interval_description = 'This component applies %s' % translate_interval(intervals[0], unit, X)
    else:
        summary += '. This function applies %s and %s' % (', '.join(translate_interval(interval, unit, X) for interval in intervals[:-1]), translate_interval(intervals[-1], unit, X))
        interval_description = 'This component applies %s and %s' % (', '.join(translate_interval(interval, unit, X) for interval in intervals[:-1]), translate_interval(intervals[-1], unit, X))
    # Do the intervals contain infinity
    intervals_contain_positive_infinity = False
    for interval in intervals:
        if interval[1] == np.Inf:
            intervals_contain_positive_infinity = True
            break
    if not intervals_contain_positive_infinity:
        # Component stops - do not need to describe how it would extrapolate
        extrap_descriptions = ['This component is assumed to stop before the end of the data and will therefore be extrapolated as zero']
    # Combine and return the descriptions
    if not intervals == [(-np.Inf, np.Inf)]: 
        descriptions.append(interval_description)
    return (summary, descriptions, extrap_descriptions)

def translate_p_value(p):
    '''LaTeX bold for extreme values'''
    if p > 0.05:
        return '\\textcolor{gray}{%0.3f}' % p
    elif p > 0.01:
        return '%0.3f' % p
    else:
        return '\\textbf{%0.3f}' % p

def translate_cum_prob(p):
    '''LaTeX bold for extreme values'''
    if p >= 0.995:
        return '\\textbf{%0.3f}' % p
    elif p >= 0.975:
        return '%0.3f' % p
    elif p > 0.025:
        return '\\textcolor{gray}{%0.3f}' % p
    elif p > 0.005:
        return '%0.3f' % p
    else:
        return '\\textbf{%0.3f}' % p

def convert_cum_prob_to_p_value(p):
    '''p-value for two sided tests'''
    return min([p*2, (1-p)*2])

def translate_p_values(fit_data, component):
    '''Produces sentences describing extreme p-values'''
    i = component
    p_values = [convert_cum_prob_to_p_value(fit_data['acf_min_p'][i]),
                convert_cum_prob_to_p_value(fit_data['acf_min_loc_p'][i]),
                convert_cum_prob_to_p_value(fit_data['pxx_max_p'][i]),
                convert_cum_prob_to_p_value(fit_data['pxx_max_loc_p'][i]),
                fit_data['qq_d_max_p'][i],
                fit_data['qq_d_min_p'][i]]
    sort_indices = [el[0] for el in sorted(enumerate(p_values), key=lambda x:x[1])]
    # Produce descriptions of each p value - even if not significant
    descriptions = []
    hypotheses = []
    if fit_data['acf_min_p'][i] < 0.5:
        descriptions.append('The minimum value of the ACF is unexpectedly low.')
        hypotheses.append('')
    else:
        descriptions.append('The minimum value of the ACF is unexpectedly high.')
        hypotheses.append('')
    if fit_data['acf_min_loc_p'][i] < 0.5:
        descriptions.append('The location of the minimum value of the ACF is unexpectedly low.')
        hypotheses.append('')
    else:
        descriptions.append('The location of the minimum value of the ACF is unexpectedly high.')
        hypotheses.append('')
    if fit_data['pxx_max_p'][i] < 0.5:
        descriptions.append('The maximum value of the periodogram is unexpectedly low.')
        hypotheses.append('')
    else:
        descriptions.append('The maximum value of the periodogram is unexpectedly high.')
        hypotheses.append('The large maximum value of the periodogram can indicate periodicity that is not being captured by the model.')
    if fit_data['pxx_max_loc_p'][i] < 0.5:
        descriptions.append('The frequency of the maximum value of the periodogram is unexpectedly low.')
        hypotheses.append('')
    else:
        descriptions.append('The frequency of the maximum value of the periodogram is unexpectedly high.')
        hypotheses.append('')
    descriptions.append('The qq plot has an unexpectedly large positive deviation from equality ($x = y$).')
    hypotheses.append('The positive deviation in the qq-plot can indicate heavy positive tails if it occurs at the right of the plot or light negative tails if it occurs as the left.')
    descriptions.append('The qq plot has an unexpectedly large negative deviation from equality ($x = y$).')
    hypotheses.append('The negative deviation in the qq-plot can indicate light positive tails if it occurs at the right of the plot or heavy negative tails if it occurs as the left.')
    # Arrange nicely
    if np.all(np.array(p_values) > 0.05):
        text = 'No discrepancies between the prior and posterior of this component have been detected'
    else:
        text = '''
The following discrepancies between the prior and posterior distributions for this component have been detected.

\\begin{itemize}
'''

        summary_item = '''
    \item %s This discrepancy has an estimated $p$-value of %s.'''
        for index in sort_indices:
            if p_values[index] <= 0.05:
                text += summary_item % (descriptions[index],translate_p_value(p_values[index]))
    
        text += '''
\end{itemize}

'''

    for index in sort_indices:
        if (p_values[index] <= 0.05) and (not hypotheses[index] == ''):
            text += '%s\n' % hypotheses[index]

    return text

def produce_summary_document(dataset_name, n_components, fit_data, short_descriptions):
    '''
    Summarises the fit to dataset_name
    '''
    text = ''
    
    header = '''
\documentclass{article} %% For LaTeX2e
\usepackage{format/nips13submit_e}
\\nipsfinalcopy %% Uncomment for camera-ready version
\usepackage{times}
\usepackage{hyperref}
\usepackage{url}
\usepackage{color}
\definecolor{mydarkblue}{rgb}{0,0.08,0.45}
\hypersetup{
    pdfpagemode=UseNone,
    colorlinks=true,
    linkcolor=mydarkblue,
    citecolor=mydarkblue,
    filecolor=mydarkblue,
    urlcolor=mydarkblue,
    pdfview=FitH}

\usepackage{graphicx, amsmath, amsfonts, bm, lipsum, capt-of}

\usepackage{natbib, xcolor, wrapfig, booktabs, multirow, caption}

\usepackage{float}

\def\ie{i.e.\ }
\def\eg{e.g.\ }

\\title{An automatic report for the dataset : %(dataset_name)s}

\\author{
The Automatic Statistician
}

\\newcommand{\\fix}{\marginpar{FIX}}
\\newcommand{\\new}{\marginpar{NEW}}

\setlength{\marginparwidth}{0.9in}
\input{include/commenting.tex}

%%%% For submission, make all render blank.
%%\\renewcommand{\LATER}[1]{}
%%\\renewcommand{\\fLATER}[1]{}
%%\\renewcommand{\TBD}[1]{}
%%\\renewcommand{\\fTBD}[1]{}
%%\\renewcommand{\PROBLEM}[1]{}
%%\\renewcommand{\\fPROBLEM}[1]{}
%%\\renewcommand{\NA}[1]{#1}  %% Note, NA's pass through!

\\begin{document}

\\allowdisplaybreaks

\maketitle

\\begin{abstract}
This report was produced by the Automatic Bayesian Covariance Discovery (ABCD) algorithm.
%%See \url{http://arxiv.org/abs/1302.4922} and \url{http://www-kd.iai.uni-bonn.de/cml/proceedings/papers/2.pdf} for preliminary papers.
\end{abstract}

\section{Executive summary}

The raw data and full model posterior with extrapolations are shown in figure~\\ref{fig:rawandfit}.

\\begin{figure}[H]
\\newcommand{\wmgd}{0.5\columnwidth}
\\newcommand{\hmgd}{3.0cm}
\\newcommand{\mdrd}{%(dataset_name)s}
\\newcommand{\mbm}{\hspace{-0.3cm}}
\\begin{tabular}{cc}
\mbm \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_raw_data} & \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_all}
\end{tabular}
\caption{Raw data (left) and model posterior with extrapolation (right)}
\label{fig:rawandfit}
\end{figure}

The structure search algorithm has identified %(n_components)s additive components in the data.'''

    text += header % {'dataset_name' : dataset_name, 'n_components' : english_number(n_components)}

     # Comment on how much variance is explained by the fit
    
    if np.all(fit_data['cum_vars'][:-1] < 90):
        text += '\nThe full model explains %2.1f\%% of the variation in the data as shown by the coefficient of determination ($R^2$) values in table~\\ref{table:stats}.' % fit_data['cum_vars'][-1]
    else:
        for i in range(n_components):
            if fit_data['cum_vars'][i] >= 90:
                text += '\nThe '
                if i < n_components - 1:
                    text += ' first '
                if i == 0:
                    text += 'additive component explains'
                else:
                    text += '%d additive components explain' % (i+1)
                text += ' %2.1f\%% of the variation in the data as shown by the coefficient of determination ($R^2$) values in table~\\ref{table:stats}.' % fit_data['cum_vars'][i]
                if fit_data['cum_vars'][i] < 99:
                    for j in range(i+1, n_components):
                        if fit_data['cum_vars'][j] >= 99:
                            text += '\nThe '
                            if j < n_components - 1:
                                text += ' first '
                            text += '%d additive components explain %2.1f\%% of the variation in the data.' % (j+1, fit_data['cum_vars'][j])  
                            break
                break
                
    # Comment on the MAE
                
    for i in range(n_components):
        if np.all(fit_data['MAE_reductions'][i:] < 0.1):
            if True:#i < n_components - 1:
                if i > 0:
                    text += '\nAfter the first '
                    if i > 1:
                        text += '%d components' % (i)
                    else:
                        text += 'component'
                    text += ' the cross validated mean absolute error (MAE) does not decrease by more than 0.1\%'
                    text += '.\nThis suggests that subsequent terms are modelling very short term trends, uncorrelated noise or are artefacts of the model or search procedure.'
                else:
                    text += 'No single component reduces the cross validated mean absolute error (MAE) by more than 0.1\%'
                    text += '.\nThis suggests that the structure search algorithm has not found anything other than very short term trends or uncorrelated noise.'
                break
                
    text +='''
Short summaries of the additive components are as follows:
\\begin{itemize}
'''

    summary_item = '''
  \item \input{%(dataset_name)s/%(dataset_name)s_%(component)d_short_description.tex} 
'''
    for i in range(n_components):
        text += summary_item % {'dataset_name' : dataset_name, 'component' : i+1}
    
    summary_end = '''
\end{itemize}
'''
    
    text += summary_end

    # Summary statistic table
    
    text += '''
\\begin{table}[htb]
\\begin{center}
{\small
\\begin{tabular}{|r|rrrrr|}
\hline
\\bf{\#} & {$R^2$ (\%%)} & {$\Delta R^2$ (\%%)} & {Residual $R^2$ (\%%)} & {Cross validated MAE} & Reduction in MAE (\%%)\\\\
\hline
- & - & - & - & %1.2f & -\\\\
''' % fit_data['MAV_data']

    table_text = '''
%d & %2.1f & %2.1f & %2.1f & %1.2f & %2.1f\\\\
'''

    cum_var_deltas = [fit_data['cum_vars'][0]] + list(np.array(fit_data['cum_vars'][1:]) - np.array(fit_data['cum_vars'][:-1]))

    for i in range(n_components):
        text += table_text % (i+1, fit_data['cum_vars'][i], cum_var_deltas[i], fit_data['cum_resid_vars'][i], fit_data['MAEs'][i], fit_data['MAE_reductions'][i])
        
    text += '''
\hline
\end{tabular}
\caption{
Summary statistics for cumulative additive fits to the data.
The residual coefficient of determination ($R^2$) values are computed using the residuals from the previous fit as the target values; this measures how much of the residual variance is explained by each new component.
The mean absolute error (MAE) is calculated using 10 fold cross validation with a contiguous block design; this measures the ability of the model to interpolate and extrapolate over moderate distances.
The model is fit using the full data and the MAE values are calculated using this model; this double use of data means that the MAE values cannot be used reliably as an estimate of out-of-sample predictive performance.
}
\label{table:stats}
}
\end{center}
\end{table}
'''

    # Check for lack of model fit

    text += '''
Model checking statistics are summarised in table~\\ref{table:check} in section~\\ref{sec:check}.'''

    moderate_bad_fits = []
    bad_fits = []
    for i in range(n_components):
        p_values = [convert_cum_prob_to_p_value(fit_data['acf_min_p'][i]),
                    convert_cum_prob_to_p_value(fit_data['acf_min_loc_p'][i]),
                    convert_cum_prob_to_p_value(fit_data['pxx_max_p'][i]),
                    convert_cum_prob_to_p_value(fit_data['pxx_max_loc_p'][i]),
                    fit_data['qq_d_max_p'][i],
                    fit_data['qq_d_min_p'][i]]
        if np.any(np.array(p_values) <= 0.01):
            bad_fits.append(i)
        elif np.any(np.array(p_values) <= 0.05):
            moderate_bad_fits.append(i)

    if len(bad_fits) + len(moderate_bad_fits) == 0:
        text += '\nThese statistics have not revealed any inconsistencies between the model and observed data.\n'
    elif len(bad_fits) == 0:
        text += '\nThese statistics have revealed statistically significant discrepancies between the data and model in '
        if len(moderate_bad_fits) > 1:
            text += 'components %s and %d.\n' % (', '.join('%d' % (i+1) for i in moderate_bad_fits[:-1]), moderate_bad_fits[-1] + 1)
        else:
            text += 'component %d.\n' % (moderate_bad_fits[0] + 1)
    else:
        text += '\nThese statistics have revealed highly statistically significant discrepancies between the data and model in '
        if len(bad_fits) > 1:
            text += 'components %s and %d.\n' % (', '.join('%d' % (i+1) for i in bad_fits[:-1]), bad_fits[-1] + 1)
        else:
            text += 'component %d.\n' % (bad_fits[0] + 1)
        if len(moderate_bad_fits) > 0:
            text += 'Moderate discrepancies have also been detected in '
            if len(moderate_bad_fits) > 1:
                text += 'components %s and %d.\n' % (', '.join('%d' % (i+1) for i in moderate_bad_fits[:-1]), moderate_bad_fits[-1] + 1)
            else:
                text += 'component %d.\n' % (moderate_bad_fits[0] + 1)

    # Announce structure of document

    text += '''
The rest of the document is structured as follows.
In section~\\ref{sec:discussion} the forms of the additive components are described and their posterior distributions are displayed.
In section~\\ref{sec:extrap} the modelling assumptions of each component are discussed with reference to how this affects the extrapolations made by the model.
Section~\\ref{sec:check} discusses model checking statistics, with plots showing the form of any detected discrepancies between the model and observed data.
'''
#A glossary of terms is provided in section~\\ref{sec:glossary}.
#'''

    text += '''
\section{Detailed discussion of additive components}
\label{sec:discussion}
'''

    component_text = '''
\subsection{Component %(component)d : %(short_description)s}

\input{%(dataset_name)s/%(dataset_name)s_%(component)d_description.tex}

This component explains %(resid_var)0.1f\%% of the residual variance; this %(incdecvar)s the total variance explained from %(prev_var)0.1f\%% to %(var)0.1f\%%.
The addition of this component %(incdecmae)s the cross validated MAE by %(MAE_reduction)0.2f\%% from %(MAE_orig)0.2f to %(MAE_new)0.2f.
%(discussion)s

\\begin{figure}[H]
\\newcommand{\wmgd}{0.5\columnwidth}
\\newcommand{\hmgd}{3.0cm}
\\newcommand{\mdrd}{%(dataset_name)s}
\\newcommand{\mbm}{\hspace{-0.3cm}}
\\begin{tabular}{cc}
\mbm \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_%(component)d} & \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_%(component)d_cum}
\end{tabular}
\caption{Pointwise posterior of component %(component)d (left) and the posterior of the cumulative sum of components with data (right)}
\label{fig:comp%(component)d}
\end{figure}
'''
    first_component_text = '''
\subsection{Component %(component)d : %(short_description)s}

\input{%(dataset_name)s/%(dataset_name)s_%(component)d_description.tex}

This component explains %(resid_var)0.1f\%% of the total variance.
The addition of this component %(incdecmae)s the cross validated MAE by %(MAE_reduction)0.1f\%% from %(MAE_orig)0.1f to %(MAE_new)0.1f.
%(discussion)s

\\begin{figure}[H]
\\newcommand{\wmgd}{0.5\columnwidth}
\\newcommand{\hmgd}{3.0cm}
\\newcommand{\mdrd}{%(dataset_name)s}
\\newcommand{\mbm}{\hspace{-0.3cm}}
\\begin{tabular}{cc}
\mbm \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_%(component)d} & \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_%(component)d_cum}
\end{tabular}
\caption{Pointwise posterior of component %(component)d (left) and the posterior of the cumulative sum of components with data (right)}
\label{fig:comp%(component)d}
\end{figure}
'''

    residual_figure = '''
\\begin{figure}[H]
\\newcommand{\wmgd}{0.5\columnwidth}
\\newcommand{\hmgd}{3.0cm}
\\newcommand{\mdrd}{%(dataset_name)s}
\\newcommand{\mbm}{\hspace{-0.3cm}}
\\begin{tabular}{cc}
\mbm \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_%(component)d_anti_cum}
\end{tabular}
\caption{Pointwise posterior of residuals after adding component %(component)d}
\label{fig:comp%(component)d}
\end{figure}
'''

    for i in range(n_components):
        if fit_data['MAE_reductions'][i] < 0.1:
            if fit_data['cum_resid_vars'][i] < 0.1:
                discussion = 'This component neither explains residual variance nor improves MAE and therefore is likely to be an artefact of the model or search procedure.'
            else:
                discussion = 'This component explains residual variance but does not improve MAE which suggests that this component describes very short term patterns, uncorrelated noise or is an artefact of the model or search procedure.'
        else:
            discussion = ''
        if i == 0:
            text += first_component_text % {'short_description' : short_descriptions[i], 'dataset_name' : dataset_name, 'component' : i+1, 'resid_var' : fit_data['cum_resid_vars'][i],
                                      'var' : fit_data['cum_vars'][i], 'MAE_reduction' : np.abs(fit_data['MAE_reductions'][i]), 'MAE_orig' : fit_data['MAV_data'], 'MAE_new' : fit_data['MAEs'][i], 'discussion' : discussion,
                                      'incdecvar' : 'increases' if fit_data['cum_vars'][i] >= fit_data['cum_vars'][i-1] else 'reduces',
                                      'incdecmae' : 'reduces' if fit_data['MAE_reductions'][i] >= 0 else 'increases'}
        else:
            text += component_text % {'short_description' : short_descriptions[i], 'dataset_name' : dataset_name, 'component' : i+1, 'resid_var' : fit_data['cum_resid_vars'][i],
                                      'prev_var' : fit_data['cum_vars'][i-1], 'var' : fit_data['cum_vars'][i], 'MAE_reduction' : np.abs(fit_data['MAE_reductions'][i]),
                                      'MAE_orig' : fit_data['MAEs'][i-1], 'MAE_new' : fit_data['MAEs'][i], 'discussion' : discussion,
                                      'incdecvar' : 'increases' if fit_data['cum_vars'][i] >= fit_data['cum_vars'][i-1] else 'reduces',
                                      'incdecmae' : 'reduces' if fit_data['MAE_reductions'][i] >= 0 else 'increases'}
        if i+1 < n_components:
            text += residual_figure % {'short_description' : short_descriptions[i], 'dataset_name' : dataset_name, 'component' : i+1, 'resid_var' : fit_data['cum_resid_vars'][i],
                                       'prev_var' : fit_data['cum_vars'][i-1], 'var' : fit_data['cum_vars'][i], 'MAE_reduction' : np.abs(fit_data['MAE_reductions'][i]),
                                       'MAE_orig' : fit_data['MAEs'][i-1], 'MAE_new' : fit_data['MAEs'][i], 'discussion' : discussion,
                                       'incdecvar' : 'increases' if fit_data['cum_vars'][i] >= fit_data['cum_vars'][i-1] else 'reduces',
                                       'incdecmae' : 'reduces' if fit_data['MAE_reductions'][i] >= 0 else 'increases'}

    text += '''
\section{Extrapolation}
\label{sec:extrap}

Summaries of the posterior distribution of the full model are shown in figure~\\ref{fig:extrap}.
The plot on the left displays the mean of the posterior together with pointwise variance.
The plot on the right displays three random samples from the posterior.

\\begin{figure}[H]
\\newcommand{\wmgd}{0.5\columnwidth}
\\newcommand{\hmgd}{3.0cm}
\\newcommand{\mdrd}{%(dataset_name)s}
\\newcommand{\mbm}{\hspace{-0.3cm}}
\\begin{tabular}{cc}
\mbm \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_all} & \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_all_sample}
\end{tabular}
\caption{Full model posterior with extrapolation. Mean and pointwise variance (left) and three random samples (right)}
\label{fig:extrap}
\end{figure}

Below are descriptions of the modelling assumptions associated with each additive component and how they affect the predictive posterior.
Plots of the pointwise posterior and samples from the posterior are also presented, showing extrapolations from each component and the cuulative sum of components.
''' % {'dataset_name' : dataset_name}

    extrap_component_text = '''
\subsection{Component %(component)d : %(short_description)s}

\input{%(dataset_name)s/%(dataset_name)s_%(component)d_extrap_description.tex}

\\begin{figure}[H]
\\newcommand{\wmgd}{0.5\columnwidth}
\\newcommand{\hmgd}{3.0cm}
\\newcommand{\mdrd}{%(dataset_name)s}
\\newcommand{\mbm}{\hspace{-0.3cm}}
\\begin{tabular}{cc}
\mbm \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_%(component)d_extrap} & \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_%(component)d_sample} \\\\
\mbm \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_%(component)d_cum_extrap} & \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_%(component)d_cum_sample}
\end{tabular}
\caption{Posterior of component %(component)d (top) and cumulative sum of components (bottom) with extrapolation. Mean and pointwise variance (left) and three random samples from the posterior distribution (right).}
\label{fig:extrap%(component)d}
\end{figure}
'''

    for i in range(n_components):
        text += extrap_component_text % {'short_description' : short_descriptions[i], 'dataset_name' : dataset_name, 'component' : i+1, 'resid_var' : fit_data['cum_resid_vars'][i],
                                         'prev_var' : fit_data['cum_vars'][i-1], 'var' : fit_data['cum_vars'][i], 'MAE_reduction' : np.abs(fit_data['MAE_reductions'][i]),
                                         'MAE_orig' : fit_data['MAEs'][i-1], 'MAE_new' : fit_data['MAEs'][i], 'discussion' : 'Some discussion about this component',
                                         'incdecvar' : 'increases' if fit_data['cum_vars'][i] >= fit_data['cum_vars'][i-1] else 'reduces',
                                         'incdecmae' : 'reduces' if fit_data['MAE_reductions'][i] >= 0 else 'increases'}
                                         
    text += '''
\section{Model checking}
\label{sec:check}

Several posterior predictive checks have been performed to assess how well the model describes the observed data.
These tests take the form of comparing statistics evaluated on samples from the prior and posterior distributions for each additive component.
The statistics are derived from autocorrelation function (ACF) estimates, periodograms and quantile-quantile (qq) plots.

Table~\\ref{table:check} displays cumulative probability and $p$-value estimates for these quantities.
Cumulative probabilities near 0/1 indicate that the test statistic was lower/higher under the posterior compared to the prior unexpectedly often \ie they contain the same information as a $p$-value for a two-tailed test and they also express if the test statistic was higher or lower than expected.
$p$-values near 0 indicate that the test statistic was larger in magnitude under the posterior compared to the prior unexpectedly often.
'''
    text += '''
\\begin{table}[htb]
\\begin{center}
{\small
\\begin{tabular}{|r|rr|rr|rr|}
\hline
 & \multicolumn{2}{|c|}{ACF} & \multicolumn{2}{|c|}{Periodogram} & \multicolumn{2}{|c|}{QQ} \\\\
\\bf{\#} & {min} & {min loc} & {max} & {max loc} & {max} & {min}\\\\
\hline
'''

    table_text = '''
%d & %s & %s & %s & %s & %s & %s\\\\
'''

    for i in range(n_components):
        text += table_text % (i+1, translate_cum_prob(fit_data['acf_min_p'][i]), translate_cum_prob(fit_data['acf_min_loc_p'][i]), \
                                   translate_cum_prob(fit_data['pxx_max_p'][i]), translate_cum_prob(fit_data['pxx_max_loc_p'][i]), \
                                   translate_p_value(fit_data['qq_d_max_p'][i]), translate_p_value(fit_data['qq_d_min_p'][i]))
        
    text += '''
\hline
\end{tabular}
\caption{
Model checking statistics for each component.
Cumulative probabilities for minimum of autocorrelation function (ACF) and its location.
Cumulative probabilities for maximum of periodogram and its location.
$p$-values for maximum and minimum deviations of QQ-plot from straight line.
}
\label{table:check}
}
\end{center}
\end{table}
'''

    if len(bad_fits) + len(moderate_bad_fits) > 0:
        text += '''
The nature of any observed discrepancies is now described and plotted and hypotheses are given for the patterns in the data that may not be captured by the model.
'''
    else:
        text += '''
No statistically significant discrepancies between the data and model have been detected but model checking plots for each component are presented below.
'''

    model_check_component_text = '''
\subsubsection{Component %(component)d : %(short_description)s}

%(discussion)s

\\begin{figure}[H]
\\newcommand{\wmgd}{0.5\columnwidth}
\\newcommand{\hmgd}{3.0cm}
\\newcommand{\mdrd}{%(dataset_name)s}
\\newcommand{\mbm}{\hspace{-0.3cm}}
\\begin{tabular}{cc}
\mbm \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_acf_bands_%(component)d} & \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_pxx_bands_%(component)d} \\\\
\mbm \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_qq_bands_%(component)d}
\end{tabular}
\caption{
ACF (top left), periodogram (top right) and quantile-quantile (bottom left) uncertainty plots.
The blue line and shading are the pointwise mean and 90\\%% confidence interval of the plots under the prior distribution for component %(component)d.
The green line and green dashed lines are the corresponding quantities under the posterior.}
\label{fig:check%(component)d}
\end{figure}
'''

    if len(bad_fits) > 0:
        text += '''
\subsection{Highly statistically significant discrepancies}
'''

    for i in range(n_components):
        if (i in bad_fits):
            text += model_check_component_text % {'short_description' : short_descriptions[i], 'dataset_name' : dataset_name, 'component' : i+1, 'resid_var' : fit_data['cum_resid_vars'][i],
                                                  'discussion' : translate_p_values(fit_data, i),
                                                  'prev_var' : fit_data['cum_vars'][i-1], 'var' : fit_data['cum_vars'][i], 'MAE_reduction' : np.abs(fit_data['MAE_reductions'][i]),
                                                  'MAE_orig' : fit_data['MAEs'][i-1], 'MAE_new' : fit_data['MAEs'][i],
                                                  'incdecvar' : 'increases' if fit_data['cum_vars'][i] >= fit_data['cum_vars'][i-1] else 'reduces',
                                                  'incdecmae' : 'reduces' if fit_data['MAE_reductions'][i] >= 0 else 'increases'}

    if len(moderate_bad_fits) > 0:
        text += '''
\subsection{Moderately statistically significant discrepancies}
'''

    for i in range(n_components):
        if (i in moderate_bad_fits):
            text += model_check_component_text % {'short_description' : short_descriptions[i], 'dataset_name' : dataset_name, 'component' : i+1, 'resid_var' : fit_data['cum_resid_vars'][i],
                                                  'discussion' : translate_p_values(fit_data, i),
                                                  'prev_var' : fit_data['cum_vars'][i-1], 'var' : fit_data['cum_vars'][i], 'MAE_reduction' : np.abs(fit_data['MAE_reductions'][i]),
                                                  'MAE_orig' : fit_data['MAEs'][i-1], 'MAE_new' : fit_data['MAEs'][i],
                                                  'incdecvar' : 'increases' if fit_data['cum_vars'][i] >= fit_data['cum_vars'][i-1] else 'reduces',
                                                  'incdecmae' : 'reduces' if fit_data['MAE_reductions'][i] >= 0 else 'increases'}

    if len(moderate_bad_fits) + len(bad_fits) < n_components:
        text += '''
\subsection{Model checking plots for components without statistically significant discrepancies}
'''

    for i in range(n_components):
        if not ((i in moderate_bad_fits) or (i in bad_fits)):
            text += model_check_component_text % {'short_description' : short_descriptions[i], 'dataset_name' : dataset_name, 'component' : i+1, 'resid_var' : fit_data['cum_resid_vars'][i],
                                                  'discussion' : translate_p_values(fit_data, i),
                                                  'prev_var' : fit_data['cum_vars'][i-1], 'var' : fit_data['cum_vars'][i], 'MAE_reduction' : np.abs(fit_data['MAE_reductions'][i]),
                                                  'MAE_orig' : fit_data['MAEs'][i-1], 'MAE_new' : fit_data['MAEs'][i],
                                                  'incdecvar' : 'increases' if fit_data['cum_vars'][i] >= fit_data['cum_vars'][i-1] else 'reduces',
                                                  'incdecmae' : 'reduces' if fit_data['MAE_reductions'][i] >= 0 else 'increases'}

    text += '''
\section{MMD - experimental section}
\label{sec:mmd}
'''
    text += '''
\\begin{table}[htb]
\\begin{center}
{\small
\\begin{tabular}{|r|r|}
\hline
\\bf{\#} & {mmd}\\\\
\hline
'''

    table_text = '''
%d & %s\\\\
'''

    for i in range(n_components):
        text += table_text % (i+1, translate_p_value(fit_data['mmd_p'][i]))
        
    text += '''
\hline
\end{tabular}
\caption{
MMD $p$-values
}
\label{table:mmd}
}
\end{center}
\end{table}
'''

    model_check_component_text = '''
\subsubsection{Component %(component)d : %(short_description)s}

\\begin{figure}[H]
\\newcommand{\wmgd}{0.5\columnwidth}
\\newcommand{\hmgd}{3.0cm}
\\newcommand{\mdrd}{%(dataset_name)s}
\\newcommand{\mbm}{\hspace{-0.3cm}}
\includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_mmd_%(component)d}
\caption{
MMD plot}
\label{fig:mmd%(component)d}
\end{figure}
'''

    for i in range(n_components):
        text += model_check_component_text % {'short_description' : short_descriptions[i], 'dataset_name' : dataset_name, 'component' : i+1, 'resid_var' : fit_data['cum_resid_vars'][i],
                                              'discussion' : translate_p_values(fit_data, i),
                                              'prev_var' : fit_data['cum_vars'][i-1], 'var' : fit_data['cum_vars'][i], 'MAE_reduction' : np.abs(fit_data['MAE_reductions'][i]),
                                              'MAE_orig' : fit_data['MAEs'][i-1], 'MAE_new' : fit_data['MAEs'][i],
                                              'incdecvar' : 'increases' if fit_data['cum_vars'][i] >= fit_data['cum_vars'][i-1] else 'reduces',
                                              'incdecmae' : 'reduces' if fit_data['MAE_reductions'][i] >= 0 else 'increases'}

#     text += '''
# \\appendix
# '''

#     text += '''
# \section{Residual style quantities}

# This appendix contains plots of residual-like quantities.
# Their utility is still being investigated so there are currently no explanations of their calculation or interpretation.
# '''

#     text += '''
# \subsection{Leave one out}

# \\begin{figure}[H]
# \\newcommand{\wmgd}{0.5\columnwidth}
# \\newcommand{\hmgd}{3.0cm}
# \\newcommand{\mdrd}{%(dataset_name)s}
# \\newcommand{\mbm}{\hspace{-0.3cm}}
# \\begin{tabular}{cc}
# \mbm \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_loo_pp} & \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_loo_resid} \\\\
# \mbm \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_loo_qq}
# \end{tabular}
# \caption{LOO posterior predictive. Distribution (left), standardised residuals (right) and qq-plot (below)}
# \label{fig:loo}
# \end{figure}

# \subsection{Leave chunk out}

# \\begin{figure}[H]
# \\newcommand{\wmgd}{0.5\columnwidth}
# \\newcommand{\hmgd}{3.0cm}
# \\newcommand{\mdrd}{%(dataset_name)s}
# \\newcommand{\mbm}{\hspace{-0.3cm}}
# \\begin{tabular}{cc}
# \mbm \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_lco_pp} & \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_lco_resid} \\\\
# \mbm \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_lco_qq}
# \end{tabular}
# \caption{LCO posterior predictive. Distribution (left), standardised residuals (right) and qq-plot (below)}
# \label{fig:lco}
# \end{figure}

# \subsection{Next data point}

# \\begin{figure}[H]
# \\newcommand{\wmgd}{0.5\columnwidth}
# \\newcommand{\hmgd}{3.0cm}
# \\newcommand{\mdrd}{%(dataset_name)s}
# \\newcommand{\mbm}{\hspace{-0.3cm}}
# \\begin{tabular}{cc}
# \mbm \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_z_resid} & \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_z_qq}
# \end{tabular}
# \caption{Inverse Cholesky thing. Standardised values (left) and qq-plot (right)}
# \label{fig:z}
# \end{figure}
# ''' % {'dataset_name' : dataset_name}

#     text += '''
# \section{Glossary of terms}
# \label{sec:glossary}

# \\begin{itemize}
# \item \emph{lengthscale} - A description of what a lengthscale is
# \end{itemize}
# '''

    text += '''
\end{document}
'''

    # Document complete
    return text
    
