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
import flexiblekernel as fk
import grammar

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
                    
def translate_product(prod, X):
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
    los_count = 0 # Local smooth components
    lin_count = 0
    per_count = 0
    cos_count = 0
    imt_count = 0 # Integrated Matern components
    unk_count = 0 # 'Unknown' kernel function
    per_kernels = []
    cos_kernels = []
    # Count calculate a variety of summary quantities
    for k in kernels:
        if isinstance(k, fk.SqExpKernel) or isinstance(k, fk.Matern5Kernel):
            #### FIXME - How accurate is it to assume that SqExp and Matern lengthscales multiply similarly
            los_count += 1
            lengthscale = -0.5 * np.log(np.exp(-2*lengthscale) + np.exp(-2*k.lengthscale))
        elif isinstance(k, fk.LinKernel):
            lin_count += 1
            lin_location = k.location
        elif isinstance(k, fk.CentredPeriodicKernel):
            per_count += 1
            per_kernels.append(k)
        elif isinstance(k, fk.CosineKernel):
            cos_count += 1
            cos_kernels.append(k)
        elif not isinstance(k, fk.ConstKernel):
            # Cannot deal with whatever type of kernel this is
            unk_count +=1
    lengthscale = np.exp(lengthscale)
    domain_range = np.max(X) - np.min(X)
    poly_names = ['linear', 'quadratic', 'cubic', 'quartic', 'quintic']
    ordinal_numbers = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eigth', 'ninth', 'tenth']
    # Now describe the properties of this product of kernels
    if (unk_count > 0):
        descriptions.append('This simple AI is not capable of describing the component who''s python representation is %s' % prod.__repr__())
        raise RuntimeError('I''m not intelligent enough to describe this kernel in natural language', prod)
    elif (los_count == 0) and (lin_count == 0) and (per_count == 0) and (cos_count == 0) and (imt_count == 0):
        descriptions.append('This component is constant')
    elif (los_count > 0) and (per_count == 0) and (cos_count == 0) and (imt_count == 0):
        # This is a pure smooth and local component (possibly with polynomial variance)
        descriptions.append('This component is locally correlated and smooth')
        if lengthscale > domain_range:
            descriptions.append('This function is very smooth; the exact lengthscale is likely to be non-physical')
        elif lengthscale < domain_range * 0.005:
            descriptions.append('This function varies very rapidly and will look like noise unless one zooms in')
        descriptions.append('The typical lengthscale of this function is %f' % lengthscale)
        if lin_count > 0:
            description = 'The variance of this function '
            if lin_count == 1:
                if lin_location < np.min(X):
                    description += 'increases linearly'
                elif lin_location > np.max(X):
                    description += 'decreases linearly'
                else:
                    description += 'increases linearly away from %f' % lin_location
            elif lin_count <= len(poly_names):
                description += 'varies %sly' % poly_names[lin_count-1]
            else:
                description += 'follows a polynomial of degree %d' % lin_count
            descriptions.append(description)
        descriptions.append('This is a placeholder for potential discussion of monotonicity')
    elif (los_count == 0) and (lin_count > 0) and (per_count == 0) and (cos_count == 0) and (imt_count == 0):
        # This is a pure polynomial component
        if lin_count == 1:
            descriptions.append('This component is linear')
            #### FIXME - Use inference to figure out the exact gradient and location it passes through
            descriptions.append('This is a placeholder that will comment on the gradient and a point it passes through')
        elif lin_count <= len(poly_names):
            # I know a special name for this type of polynomial
            descriptions.append('This component is a %s polynomial' % poly_names[lin_count-1])
        else:
            descriptions.append('This component is a polynomial of degree %d' % lin_count)
    elif (per_count > 0) or (cos_count > 0) and (imt_count == 0):
        if ((per_count == 1) and (cos_count == 0)) or ((per_count == 0) and (cos_count == 1)):
            if per_count == 1:
                k = per_kernels[0]
            else:
                k = cos_kernels[0]
            main_description = 'This component is '
            #if los_count > 0:
            #    main_description += 'approximately '
            if per_count == 1:
                main_description += 'periodic '
            else:
                main_description += 'sinusoidal '
            main_description += 'with a period of %f' % np.exp(k.period)
            if lin_count > 0:
                main_description += ' with '
                if lin_count == 1:
                    if lin_location < np.min(X):
                        #if los_count > 0:
                        #    main_description += 'approximately '
                        main_description += 'linearly increasing amplitude'
                    elif lin_location > np.max(X):
                        #if los_count > 0:
                        #    main_description += 'approximately '
                        main_description += 'linearly decreasing amplitude'
                    else:
                        main_description += 'amplitude increasing '
                        #if los_count > 0:
                        #    main_description += 'approximately '
                        main_description += 'linearly away from %f' % lin_location
                elif lin_count <= len(poly_names):
                    #if los_count > 0:
                    #    main_description += 'approximately '
                    main_description += '%sly varying amplitude' % poly_names[lin_count-1]
                else:
                    main_description += 'a variance that '
                    #if los_count > 0:
                    #    main_description += 'approximately '
                    main_description += 'follows a polynomial of degree %d' % lin_count
            descriptions.append(main_description)
            if los_count > 0:
                descriptions.append('The exact form of the function changes but it is locally correlated and the changes are smooth')
                descriptions.append('The local correlation has a typical lengthscale of %f' % lengthscale)
            if per_count == 1:
                #### FIXME - this correspondence is only approximate - based on small angle approx
                per_lengthscale = 0.5*np.exp(k.lengthscale + k.period)/np.pi # This definition of lengthscale fits better with local smooth kernels
                descriptions.append('The typical lengthscale of the periodic function is %f' % per_lengthscale)
                if per_lengthscale > np.exp(k.period):
                    descriptions.append('The lengthscale is greater than the period so the function is almost sinusoidal')
        else: # Several periodic components
            main_description = 'This component behaves '
            #if los_count > 0:
            #    main_description += 'approximately '
            main_description += 'like the product of'
            if per_count == 1:
                main_description += ' a periodic function'
            elif per_count > 1:
                main_description += ' several periodic functions'
            if per_count > 0:
                main_description += ' and'
            if cos_count == 1:
                main_description += ' a sinusoid'
            elif cos_count > 1:
                main_description += ' several sinusoids'
            if lin_count > 0:
                main_description += ' with '
                if lin_count == 1:
                    if lin_location < np.min(X):
                        #if los_count > 0:
                        #    main_description += 'approximately '
                        main_description += 'linearly increasing amplitude'
                    elif lin_location > np.max(X):
                        #if los_count > 0:
                        #    main_description += 'approximately '
                        main_description += 'linearly decreasing amplitude'
                    else:
                        main_description += 'amplitude increasing '
                        #if los_count > 0:
                        #    main_description += 'approximately '
                        main_description += 'linearly away from %f' % lin_location
                elif lin_count <= len(poly_names):
                    #if los_count > 0:
                    #    main_description += 'approximately '
                    main_description += '%sly varying amplitude' % poly_names[lin_count-1]
                else:
                    main_description += 'a variance that '
                    #if los_count > 0:
                    #    main_description += 'approximately '
                    main_description += 'follows a polynomial of degree %d' % lin_count
            descriptions.append(main_description)
            if los_count > 0:
                descriptions.append('The exact form of the function changes but it is locally correlated and the changes are smooth')
                descriptions.append('The local correlation has a typical lengthscale of %f' % lengthscale)
            for (i, k) in enumerate(per_kernels):
                description = 'The '
                if i <= len(ordinal_numbers):
                    if len(per_kernels) > 1:
                        description += '%s ' % ordinal_numbers[i]
                else:
                    description += '%dth ' % (i+1)
                description += 'periodic function has a period of %f' % np.exp(k.period)
                descriptions.append(description)
                #### FIXME - this correspondence is only approximate
                per_lengthscale = 0.5*np.exp(k.lengthscale + k.period)/np.pi # This definition of lengthscale fits better with local smooth kernels
                descriptions.append('The typical lengthscale of this function is %f' % per_lengthscale)
                if per_lengthscale > np.exp(k.period):
                    descriptions.append('The lengthscale is greater than the period so this function is almost sinusoidal')
            for (i, k) in enumerate(cos_kernels):
                if i <= len(ordinal_numbers):
                    if len(cos_kernels) > 1:
                        descriptions.append('The %s sinusoid has a period of %f' % (ordinal_numbers[i], np.exp(k.period)))
                    else:
                        descriptions.append('The sinusoid has a period of %f' % np.exp(k.period))
                else:
                    descriptions.append('The %dth sinusoid has a period of %f' % (i+1, np.exp(k.period)))
    else:
        descriptions.append('This simple AI is not capable of describing the component who''s python representation is %s' % prod.__repr__())
        raise RuntimeError('I''m not intelligent enough to describe this kernel in natural language', prod)
    # Return a list of sentences
    return descriptions
    
def translate_interval(interval):
    '''Describe a 1d interval, including relevant prepositions'''
    if interval == (-np.Inf, np.Inf):
        return 'to the entire input domain'
    elif interval[0] == -np.Inf:
        return 'until %f' % interval[1]
    elif interval[1] == np.Inf:
        return 'from %f onwards' % interval[0]
    else:
        return 'from %f until %f' % (interval[0], interval[1])

def translate_additive_component(k, X):
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
    descriptions = translate_product(k, X)
    # Describe the intervals this kernel acts upon
    intervals = sorted(intervals)
    if len(intervals) == 0:
        interval_description = 'The combination of changepoint operators is such that this simple AI cannot describe where this component acts; please see visual output or upgrade me'
    elif len(intervals) == 1:
        interval_description = 'This component applies %s' % translate_interval(intervals[0])
    else:
        interval_description = 'This component applies %s and %s' % (', '.join(translate_interval(interval) for interval in intervals[:-1]), translate_interval(intervals[-1]))
    # Combine and return the descriptions
    descriptions.append(interval_description)
    return descriptions
    
