"""
Translates kernel expressions into natural language descriptions

@authors: James Robert Lloyd (jrl44@cam.ac.uk)
          
September 2013          
"""

import numpy as np
import flexiblekernel as fk
import grammar

def interval_intersection(int1, int2):
    '''Finds the intersection of two intervals'''
    if ((int1[0] <= int2[0]) and (int1[1] >= int2[0])) or ((int1[1] >= int2[1]) and (int1[0] <= int2[1])):
        return (np.max([int1[0], int2[0]]), np.min([int1[1], int2[1]])
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
    elif isinstance(k, fk.ChangeBurstTanhKernel) or isinstance(k, fk.ChangeBurstKernel):
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
                new_intervals.append((np.max([int1[0], int2[0]]), np.min([int1[1], int2[1]]))
    # Note that new_intervals may now be empty but we should recurse to return a kernel in a standard form
    return find_region_of_influence(base_kernel, new_intervals)
                    
def translate_product(k, X):
    '''
    Translates a product of base kernels
    '''
    return 'I don''t know how to describe base kernels yet so I can''t talk about this component'

def translate_additive_component(k, X):
    '''
    Expects a kernel that is a single component after calling flexiblekernel.distribute_products
    Translates this into natural language or explains that why it's translation abilities aren't up to the job
    '''
    #### TODO
    #     - Deal with arbitrarily nested kernels
    #     - Frame in terms of total kernel variance
    #     - Evaluate kernel on data to determine if it is monotonic
    #     - Evaluate how much this kernel reduces residual variance on training data and on prediction (MAE)
    #     - The above can be done by plot_decomp MATLAB code saving data files with details about the components (and their order)
    k = grammar.canonical(k) # Just in case
    (intervals, k) = find_region_of_influence(k)
    descriptions = 
    
