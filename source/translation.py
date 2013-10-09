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

import flexiblekernel as fk
import grammar

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
        return 'Unrecognised format'
        raise RuntimeError('I do not know about this unit of measurement', unit)
        
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
        return 'Unrecognised format'
        raise RuntimeError('I do not know about this unit of measurement', unit)

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
    if isinstance(k, fk.ChangePointTanhKernel) or isinstance(k, fk.ChangePointKernel):
        if isinstance(k.operands[0], fk.ZeroKernel) or isinstance(k.operands[0], fk.NoneKernel):
            local_intervals = [(k.location, np.Inf)]
            base_kernel = k.operands[1]
        else:
            local_intervals = [(-np.Inf, k.location)]
            base_kernel = k.operands[0]
    elif isinstance(k, fk.ChangeBurstTanhKernel):
        if isinstance(k.operands[0], fk.ZeroKernel) or isinstance(k.operands[0], fk.NoneKernel):
            local_intervals = [(k.location-np.exp(k.width)/2, k.location+np.exp(k.width)/2)]
            base_kernel = k.operands[1]
        else:
            local_intervals = [(-np.Inf, k.location-np.exp(k.width)/2), (k.location+np.exp(k.width)/2, np.Inf)]
            base_kernel = k.operands[0]
    elif isinstance(k, fk.ProductKernel) and not all(isinstance(op, fk.MaskKernel) or isinstance(op, fk.BaseKernel) for op in k.operands):
        # Product contains an operator - find it
        for (i, op) in enumerate(k.operands):
            if not (isinstance(op, fk.MaskKernel) or isinstance(op, fk.BaseKernel)):
                # Found an operator - place all other kernels within it - resulting in the operator at the top of the kernel
                other_kernels = k.operands[:i] + k.operands[(i+1):]
                if isinstance(op.operands[0], fk.ZeroKernel) or isinstance(op.operands[0], fk.NoneKernel):
                    for other_k in other_kernels:
                        op.operands[1] *= other_k
                    op.operands[1] = grammar.canonical(op.operands[1])
                else:
                    for other_k in other_kernels:
                        op.operands[0] *= other_k
                    op.operands[0] = grammar.canonical(op.operands[0])
                return find_region_of_influence(op, intervals)
    elif isinstance(k, fk.MaskKernel) or isinstance(k, fk.BaseKernel) or (isinstance(k, fk.ProductKernel) and all(isinstance(op, fk.MaskKernel) or isinstance(op, fk.BaseKernel) for op in k.operands)):
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
    if (lin_count > 0) and (exp_count == 0):
        description += 'The %s of the %s ' % (quantity, component)
        if lin_count == 1:
            if lin_location < np.min(X):
                summary += 'with %slinearly increasing %s' % (qualifier, quantity)
                description += 'increases %slinearly' % qualifier
            elif lin_location > np.max(X):
                summary += 'with %slinearly decreasing %s' % (qualifier, quantity)
                description += 'decreases %slinearly' % qualifier
            else:
                summary += 'with %s increasing %slinearly away from %s' % (quantity, qualifier, english_point(lin_location, unit, X))
                description += 'increases %slinearly away from %s' % (qualifier, english_point(lin_location, unit, X))
        elif lin_count <= len(poly_names):
            summary += 'with %s%sly varying %s' % (qualifier, poly_names[lin_count-1], quantity)
            description += 'varies %s%sly' % (qualifier, poly_names[lin_count-1])
        else:
            summary += 'with %s given %sby a polynomial of degree %d' % (qualifier, quantity, lin_count)
            description += 'is given %sby a polynomial of degree %d' % (qualifier, lin_count)
    elif (exp_count > 0) and (lin_count == 0):
        description += 'The %s of the %s ' % (quantity, component)
        if exp_rate > 0:
            summary = 'with exponentially %sincreasing %s' % (qualifier, quantity)
            description += 'increases %sexponentially' % qualifier
        else:
            summary = 'with exponentially %sdecreasing %s' % (qualifier, quantity)
            description += 'decreases %sexponentially' % (qualifier)
    else:
        #### TODO - this is the product of lin and exp - explanantions can be made nicer by looking for turning points
        if exp_rate > 0:
            description += 'The %s of the %s is given %sby the product of a polynomial of degree %d and an increasing exponential function' % (quantity, component, qualifier, lin_count)
            summary += 'with %s given %sby a product of a polynomial of degree %d and an increasing exponential function' % (quantity, qualifier, lin_count)
        else:
            description += 'The %s of the %s is given %sby the product of a polynomial of degree %d and a decreasing exponential function' % (quantity, component, qualifier, lin_count)
            summary += 'with %s given %sby a product of a polynomial of degree %d and a decreasing exponential function' % (quantity, qualifier, lin_count)
    return (summary, description)
                    
def translate_product(prod, X, monotonic, gradient, unit=''):
    '''
    Translates a product of base kernels
    '''
    #### TODO
    # - assumes 1d
    # - IMTLin not dealt with
    # - Step function not dealt with
    # Strip masks and produce list of base kernels in product
    prod = grammar.canonical(fk.strip_masks(grammar.canonical(prod)))
    if isinstance(prod, fk.BaseKernel):
        kernels = [prod]
    else:
        kernels = prod.operands
    # Initialise
    descriptions = []
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
        if isinstance(k, fk.SqExpKernel) or isinstance(k, fk.Matern5Kernel):
            #### FIXME - How accurate is it to assume that SqExp and Matern lengthscales multiply similarly
            los_count += 1
            lengthscale = -0.5 * np.log(np.exp(-2*lengthscale) + np.exp(-2*k.lengthscale))
        #elif isinstance(k, fk.LinKernel) or isinstance(k, fk.PureLinKernel):
        elif isinstance(k, fk.PureLinKernel):
            lin_count += 1
            lin_location = k.location
        elif isinstance(k, fk.CentredPeriodicKernel):
            per_count += 1
            per_kernels.append(k)
            min_period = np.min([np.exp(k.period), min_period])
        elif isinstance(k, fk.FourierKernel):
            per_count += 1
            per_kernels.append(k)
            min_period = np.min([np.exp(k.period), min_period])
        elif isinstance(k, fk.CosineKernel):
            cos_count += 1
            cos_kernels.append(k)
            min_period = np.min([np.exp(k.period), min_period])
        elif isinstance(k, fk.ExpKernel):
            exp_count += 1
            exp_rate += k.rate
        elif isinstance(k, fk.NoiseKernel):
            noi_count += 1
        elif not isinstance(k, fk.ConstKernel):
            # Cannot deal with whatever type of kernel this is
            unk_count +=1
    lengthscale = np.exp(lengthscale)
    domain_range = np.max(X) - np.min(X)
    # Now describe the properties of this product of kernels
    if (unk_count > 0):
        summary = 'This simple AI is not capable of describing the component who''s python representation is %s' % prod.__repr__()
        descriptions.append('This simple AI is not capable of describing the component who''s python representation is %s' % prod.__repr__())
        raise RuntimeError('I''m not intelligent enough to describe this kernel in natural language', prod)
    elif (noi_count > 0):
        summary = 'Uncorrelated noise'
        descriptions.append('This component models uncorrelated noise')  
        if lin_count + exp_count > 0:
            (var_summary, var_description) = translate_parametric_window(X, unit=unit, lin_count=lin_count, exp_count=exp_count, lin_location=lin_location, exp_rate=exp_rate, quantity='standard deviation', component='noise')
            descriptions.append(var_description)
            summary += ' %s' % var_summary
    elif (los_count == 0) and (lin_count == 0) and (per_count == 0) and (cos_count == 0) and (imt_count == 0):
        summary = 'A constant'
        descriptions.append('This component is constant')      
    elif (los_count > 0) and (per_count == 0) and (cos_count == 0) and (imt_count == 0):
        # This is a pure smooth and local component (possibly with parametric variance)
        if lengthscale > 0.5 * domain_range:
            if monotonic == 1:
                summary = 'A very smooth monotonically increasing function'
                descriptions.append('This function is very smooth and monotonically increasing')
            elif monotonic == -1:
                summary = 'A very smooth monotonically decreasing function'
                descriptions.append('This function is very smooth and monotonically decreasing')
            else:
                summary = 'A very smooth function'
                descriptions.append('This function is very smooth')
        elif lengthscale < domain_range * 0.005:
            summary = 'A rapidly varying smooth function'
            descriptions.append('This function is a rapidly varying but smooth function with a typical lengthscale of %s' % english_length(lengthscale, unit))
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
        if lin_count + exp_count > 0:
            # Parametric variance
            (var_summary, var_description) = translate_parametric_window(X, unit=unit, lin_count=lin_count, exp_count=exp_count, lin_location=lin_location, exp_rate=exp_rate, quantity='marginal standard deviation', component='function')
            descriptions.append(var_description)
            summary += ' %s' % var_summary
    elif (los_count == 0) and (lin_count > 0) and (per_count == 0) and (cos_count == 0) and (imt_count == 0):
        # This is a pure polynomial component
        if lin_count == 1:
            if gradient > 0:
                summary = 'A linearly increasing function'
                descriptions.append('This component is linearly increasing')
            else:
                summary = 'A linearly decreasing function'
                descriptions.append('This component is linearly decreasing')
        elif lin_count <= len(poly_names):
            # I know a special name for this type of polynomial
            summary = 'A %s polynomial' % poly_names[lin_count-1]
            descriptions.append('This component is a %s polynomial' % poly_names[lin_count-1])
        else:
            summary = 'A polynomial of degree %d' % lin_count
            descriptions.append('This component is a polynomial of degree %d' % lin_count)
    elif (per_count > 0) or (cos_count > 0) and (imt_count == 0):
        if ((per_count == 1) and (cos_count == 0)) or ((per_count == 0) and (cos_count == 1)):
            k = per_kernels[0] if per_count == 1 else cos_kernels[0]
            if (lin_count + exp_count == 0) and (los_count == 0):
                # Pure periodic
                summary = 'An exactly '
                main_description = 'This component is exactly '
                if per_count == 1:
                    summary += 'periodic function '
                    main_description += 'periodic '
                else:
                    summary += 'sinusoidal function '
                    main_description += 'sinusoidal '
                summary += 'with a period of %s' % english_length(np.exp(k.period), unit)
                main_description += 'with a period of %s' % english_length(np.exp(k.period), unit)
                descriptions.append(main_description)
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
                    else:
                        descriptions.append('Across periods the shape of this function varies smoothly with a typical lengthscale of %s' % english_length(lengthscale, unit))   
                    if per_count == 1:
                        per_lengthscale = 0.5*np.exp(k.lengthscale + k.period)/np.pi # This definition of lengthscale fits better with local smooth kernels
                        if k.lengthscale > 2:
                            descriptions.append('The shape of this function within each period is very smooth and resembles a sinusoid')
                        else:
                            descriptions.append('The shape of this function within each period has a typical lengthscale of %s' % english_length(per_lengthscale, unit))
            elif (lin_count + exp_count > 0) and (los_count == 0):
                # Pure periodic but with changing amplitude
                summary = 'An exactly '
                main_description = 'This component is exactly '
                if per_count == 1:
                    summary += 'periodic function '
                    main_description += 'periodic '
                else:
                    summary += 'sinusoidal function '
                    main_description += 'sinusoidal '
                summary += 'with a period of %s' % english_length(np.exp(k.period), unit)
                main_description += 'with a period of %s but with varying amplitude' % english_length(np.exp(k.period), unit)
                descriptions.append(main_description)
                (var_summary, var_description) = translate_parametric_window(X, unit=unit, lin_count=lin_count, exp_count=exp_count, lin_location=lin_location, exp_rate=exp_rate, quantity='amplitude', component='function')
                descriptions.append(var_description)
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
                    (var_summary, var_description) = translate_parametric_window(X, unit=unit, lin_count=lin_count, exp_count=exp_count, lin_location=lin_location, exp_rate=exp_rate, quantity='marginal standard deviation', component='function')
                    descriptions.append(var_description)
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
                    else:
                        descriptions.append('Across periods the shape of this function varies smoothly with a typical lengthscale of %s' % english_length(lengthscale, unit))   
                    (var_summary, var_description) = translate_parametric_window(X, unit=unit, lin_count=lin_count, exp_count=exp_count, lin_location=lin_location, exp_rate=exp_rate, quantity='amplitude', component='function', qualifier='approximately ')
                    descriptions.append(var_description)
                    summary += ' and %s' % var_summary
                    if per_count == 1:
                        per_lengthscale = 0.5*np.exp(k.lengthscale + k.period)/np.pi # This definition of lengthscale fits better with local smooth kernels
                        if k.lengthscale > 2:
                            descriptions.append('The shape of this function within each period is very smooth and resembles a sinusoid')
                        else:
                            descriptions.append('The shape of this function within each period has a typical lengthscale of %s' % english_length(per_lengthscale, unit))
        else: # Several periodic components
            if los_count > 0:
                summary = 'An approximate product of'
            else:
                summary = 'A product of'
            main_description = 'This component is a product of'
            #if los_count > 0:
            #    main_description += 'approximately '
            #main_description += 'like the product of'
            if per_count == 1:
                summary += ' a periodic function'
                main_description += ' a periodic function'
            elif per_count > 1:
                summary += ' several periodic functions'
                main_description += ' several periodic functions'
            if (per_count > 0) and (cos_count > 0):
                summary += ' and'
                main_description += ' and'
            if cos_count == 1:
                summary += ' a sinusoid'
                main_description += ' a sinusoid'
            elif cos_count > 1:
                summary += ' several sinusoids'
                main_description += ' several sinusoids'
            descriptions.append(main_description)
            if los_count > 0:
                descriptions.append('Across periods the shape of this function varies smoothly with a typical lengthscale of %s' % english_length(lengthscale, unit))
            if lin_count + exp_count > 0:
                qualifier = 'approximately ' if (los_count > 0) else ''
                (var_summary, var_description) = translate_parametric_window(X, unit=unit, lin_count=lin_count, exp_count=exp_count, lin_location=lin_location, exp_rate=exp_rate, quantity='amplitude', component='function', qualifier=qualifier)
                descriptions.append(var_description)
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
        raise RuntimeError('I''m not intelligent enough to describe this kernel in natural language', prod)
    if exp_count > 0:
        descriptions.append('There were also some undecribed exponentials')
        summary += '. There was also some quantity of exp'
    # Return a list of sentences
    return (summary, descriptions)
    
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
    #     - Frame in terms of total kernel variance
    #     - Evaluate kernel on data to determine if it is monotonic
    #     - Evaluate how much this kernel reduces residual variance on training data and on prediction (MAE)
    #     - The above can be done by plot_decomp MATLAB code saving data files with details about the components (and their order)
    #     - Discuss steepness of changepoints when there is only one / the form is simple enough
    k = grammar.canonical(k) # Just in case
    (intervals, k) = find_region_of_influence(k)
    # Calculate the description of the changepoint free part of the kernel
    (summary, descriptions) = translate_product(k, X, monotonic, gradient, unit)
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
    # Combine and return the descriptions
    if not intervals == [(-np.Inf, np.Inf)]: 
        descriptions.append(interval_description)
    return (summary, descriptions)

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
James Robert Lloyd\\\\
University of Cambridge\\\\
Department of Engineering\\\\
\\texttt{jrl44@cam.ac.uk}
\And
David Duvenaud\\\\
University of Cambridge \\\\
Department of Engineering \\\\
\\texttt{dkd23@cam.ac.uk}
\And
Roger Grosse\\\\
M.I.T.\\\\
Brain and Cognitive Sciences \\\\
\\texttt{rgrosse@mit.edu}
\And
Joshua B. Tenenbaum\\\\
M.I.T.\\\\
Brain and Cognitive Sciences \\\\
\\texttt{jbt@mit.edu}
\And
Zoubin Ghahramani\\\\
University of Cambridge \\\\
Department of Engineering \\\\
\\texttt{zoubin@eng.cam.ac.uk}
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
This report was produced automatically by the Gaussian process structure search algorithm.
See \url{http://arxiv.org/abs/1302.4922} for a preliminary paper and see \url{https://github.com/jamesrobertlloyd/gpss-research} for the latest source code.
\end{abstract}

\section{Executive summary}

The raw data and full model posterior with extrapolations are shown in figure~\\ref{fig:rawandfit}.

\\begin{figure}[H]
\\newcommand{\wmgd}{0.5\columnwidth}
\\newcommand{\hmgd}{3.0cm}
\\newcommand{\mdrd}{figures/%(dataset_name)s}
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
  \item \input{figures/%(dataset_name)s/%(dataset_name)s_%(component)d_short_description.tex} 
'''
    for i in range(n_components):
        text += summary_item % {'dataset_name' : dataset_name, 'component' : i+1}
    
    summary_end = '''
\end{itemize}
'''
    
    text += summary_end
    
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
The model is fit using the full data so the MAE values cannot be used reliably as an estimate of out-of-sample predictive performance.
}
\label{table:stats}
}
\end{center}
\end{table}

\section{Detailed discussion of additive components}
'''

    component_text = '''
\subsection{Component %(component)d : %(short_description)s}

\input{figures/%(dataset_name)s/%(dataset_name)s_%(component)d_description.tex}

This component explains %(resid_var)0.1f\%% of the residual variance; this %(incdecvar)s the total variance explained from %(prev_var)0.1f\%% to %(var)0.1f\%%.
The addition of this component %(incdecmae)s the cross validated MAE by %(MAE_reduction)0.2f\%% from %(MAE_orig)0.2f to %(MAE_new)0.2f.
%(discussion)s

\\begin{figure}[H]
\\newcommand{\wmgd}{0.5\columnwidth}
\\newcommand{\hmgd}{3.0cm}
\\newcommand{\mdrd}{figures/%(dataset_name)s}
\\newcommand{\mbm}{\hspace{-0.3cm}}
\\begin{tabular}{cc}
\mbm \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_%(component)d} & \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_%(component)d_cum}
\end{tabular}
\caption{Posterior of component %(component)d (left) and the posterior of the cumulative sum of components with data (right)}
\label{fig:comp%(component)d}
\end{figure}
'''
    first_component_text = '''
\subsection{Component %(component)d : %(short_description)s}

\input{figures/%(dataset_name)s/%(dataset_name)s_%(component)d_description.tex}

This component explains %(resid_var)0.1f\%% of the total variance.
The addition of this component %(incdecmae)s the cross validated MAE by %(MAE_reduction)0.1f\%% from %(MAE_orig)0.1f to %(MAE_new)0.1f.
%(discussion)s

\\begin{figure}[H]
\\newcommand{\wmgd}{0.5\columnwidth}
\\newcommand{\hmgd}{3.0cm}
\\newcommand{\mdrd}{figures/%(dataset_name)s}
\\newcommand{\mbm}{\hspace{-0.3cm}}
\\begin{tabular}{cc}
\mbm \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_%(component)d} & \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_%(component)d_cum}
\end{tabular}
\caption{Posterior of component %(component)d (left) and the posterior of the cumulative sum of components with data (right)}
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

    text += '''
\section{Extrapolation}

Summaries of the posterior distribution of the full model are shown in figure~\\ref{fig:extrap}.
The plot on the left displays the mean of the posterior together with pointwise variance.
The plot on the right displays three random samples from the posterior.

\\begin{figure}[H]
\\newcommand{\wmgd}{0.5\columnwidth}
\\newcommand{\hmgd}{3.0cm}
\\newcommand{\mdrd}{figures/%(dataset_name)s}
\\newcommand{\mbm}{\hspace{-0.3cm}}
\\begin{tabular}{cc}
\mbm \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_all} & \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_all_sample}
\end{tabular}
\caption{Full model posterior. Mean and pointwise variance (left) and three random samples (right)}
\label{fig:extrap}
\end{figure}
''' % {'dataset_name' : dataset_name}

    extrap_component_text = '''
\subsection{Component %(component)d : %(short_description)s}

Some discussion about extrapolation.

\\begin{figure}[H]
\\newcommand{\wmgd}{0.5\columnwidth}
\\newcommand{\hmgd}{3.0cm}
\\newcommand{\mdrd}{figures/%(dataset_name)s}
\\newcommand{\mbm}{\hspace{-0.3cm}}
\\begin{tabular}{cc}
\mbm \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_%(component)d_extrap} & \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_%(component)d_sample} \\\\
\mbm \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_%(component)d_cum_extrap} & \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_%(component)d_cum_sample}
\end{tabular}
\caption{Posterior of component %(component)d. Mean and pointwise variance (left) and three random samples from this distribution (right)}
\label{fig:extrap%(component)d}
\end{figure}
'''

    for i in range(n_components):
        text += extrap_component_text % {'short_description' : short_descriptions[i], 'dataset_name' : dataset_name, 'component' : i+1, 'resid_var' : fit_data['cum_resid_vars'][i],
                                         'prev_var' : fit_data['cum_vars'][i-1], 'var' : fit_data['cum_vars'][i], 'MAE_reduction' : np.abs(fit_data['MAE_reductions'][i]),
                                         'MAE_orig' : fit_data['MAEs'][i-1], 'MAE_new' : fit_data['MAEs'][i], 'discussion' : discussion,
                                         'incdecvar' : 'increases' if fit_data['cum_vars'][i] >= fit_data['cum_vars'][i-1] else 'reduces',
                                         'incdecmae' : 'reduces' if fit_data['MAE_reductions'][i] >= 0 else 'increases'}

    text += '''
\end{document}
'''

    # Document complete
    return text
    
