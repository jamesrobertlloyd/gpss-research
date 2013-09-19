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
                    
def translate_product(prod, X, monotonic, gradient):
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
    min_period = np.Inf
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
            min_period = np.min([np.exp(k.period), min_period])
        elif isinstance(k, fk.CosineKernel):
            cos_count += 1
            cos_kernels.append(k)
            min_period = np.min([np.exp(k.period), min_period])
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
    elif (los_count == 0) and (lin_count == 0) and (per_count == 0) and (cos_count == 0) and (imt_count == 0):
        summary = 'A constant'
        descriptions.append('This component is constant')
    elif (los_count > 0) and (per_count == 0) and (cos_count == 0) and (imt_count == 0):
        # This is a pure smooth and local component (possibly with polynomial variance)
        if lengthscale > domain_range:
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
            descriptions.append('This function is a rapidly varying but smooth function with a typical lengthscale of %f' % lengthscale)
        else:
            if monotonic == 1:
                summary = 'A smooth monotonically increasing function'
                descriptions.append('This component is a smooth and monotonically increasing function with a typical lengthscale of %f' % lengthscale)
            elif monotonic == -1:
                summary = 'A smooth monotonically decreasing function'
                descriptions.append('This component is a smooth and monotonically decreasing function with a typical lengthscale of %f' % lengthscale)
            else:
                summary = 'A smooth function'
                descriptions.append('This component is a smooth function with a typical lengthscale of %f' % lengthscale)
        if lin_count > 0:
            description = 'The variance of this function '
            if lin_count == 1:
                if lin_location < np.min(X):
                    summary += ' with linearly increasing variance'
                    description += 'increases linearly'
                elif lin_location > np.max(X):
                    summary += ' with linearly decreasing variance'
                    description += 'decreases linearly'
                else:
                    summary += ' with variance increasing linearly away from %f' % lin_location
                    description += 'increases linearly away from %f' % lin_location
            elif lin_count <= len(poly_names):
                summary += ' with %sly varying variance' % poly_names[lin_count-1]
                description += 'varies %sly' % poly_names[lin_count-1]
            else:
                summary += ' with variance following a polynomial of degree' % poly_names[lin_count-1]
                description += 'follows a polynomial of degree %d' % lin_count
            descriptions.append(description)
    elif (los_count == 0) and (lin_count > 0) and (per_count == 0) and (cos_count == 0) and (imt_count == 0):
        # This is a pure polynomial component
        if lin_count == 1:
            summary = 'A linear function with a gradient of %f' % gradient
            descriptions.append('This component is linear with a gradient of %f' % gradient)
        elif lin_count <= len(poly_names):
            # I know a special name for this type of polynomial
            summary = 'A %s polynomial' % poly_names[lin_count-1]
            descriptions.append('This component is a %s polynomial' % poly_names[lin_count-1])
        else:
            summary = 'A polynomial of degree %d' % lin_count
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
            if los_count > 0:
                summary = 'An approximately '
            else:
                summary = 'A '
            if per_count == 1:
                summary += 'periodic function'
                main_description += 'periodic '
            else:
                summary += 'sinusoidal function'
                main_description += 'sinusoidal '
            summary += ' with a period of %f' % np.exp(k.period)
            main_description += 'with a period of %f' % np.exp(k.period)
            if lin_count > 0:
                summary += ' with '
                main_description += ' with '
                if lin_count == 1:
                    if lin_location < np.min(X):
                        #if los_count > 0:
                        #    main_description += 'approximately '
                        summary += 'linearly increasing amplitude'
                        main_description += 'linearly increasing amplitude'
                    elif lin_location > np.max(X):
                        #if los_count > 0:
                        #    main_description += 'approximately '
                        summary += 'linearly decreasing amplitude'
                        main_description += 'linearly decreasing amplitude'
                    else:
                        main_description += 'amplitude increasing '
                        #if los_count > 0:
                        #    main_description += 'approximately '
                        summary += 'linearly away from %f' % lin_location
                        main_description += 'linearly away from %f' % lin_location
                elif lin_count <= len(poly_names):
                    #if los_count > 0:
                    #    main_description += 'approximately '
                    summary += '%sly varying amplitude' % poly_names[lin_count-1]
                    main_description += '%sly varying amplitude' % poly_names[lin_count-1]
                else:
                    summary += 'a variance that '
                    main_description += 'a variance that '
                    #if los_count > 0:
                    #    main_description += 'approximately '
                    summary += 'follows a polynomial of degree %d' % lin_count
                    main_description += 'follows a polynomial of degree %d' % lin_count
            descriptions.append(main_description)
            if los_count > 0:
                if lengthscale > domain_range:
                    descriptions.append('The exact form of the function changes smoothly but very slowly')
                else:
                    descriptions.append('The exact form of the function changes smoothly with a typical lengthscale of %f' % lengthscale)
                if lengthscale < min_period * 0.5:
                    descriptions.append('Since this lengthscale is smaller than half the period this component may more closely resemble a smooth function without periodicity')
            if per_count == 1:
                #### FIXME - this correspondence is only approximate - based on small angle approx
                per_lengthscale = 0.5*np.exp(k.lengthscale + k.period)/np.pi # This definition of lengthscale fits better with local smooth kernels
                descriptions.append('The typical lengthscale of the periodic function is %f' % per_lengthscale)
                if per_lengthscale > np.exp(k.period):
                    descriptions.append('The lengthscale of this periodic function is greater than its period so the function is almost sinusoidal')
        else: # Several periodic components
            if los_count > 0:
                summary = 'An approxiate product of'
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
            if lin_count > 0:
                summary += ' with '
                main_description += ' with '
                if lin_count == 1:
                    if lin_location < np.min(X):
                        #if los_count > 0:
                        #    main_description += 'approximately '
                        summary += 'linearly increasing amplitude'
                        main_description += 'linearly increasing amplitude'
                    elif lin_location > np.max(X):
                        #if los_count > 0:
                        #    main_description += 'approximately '
                        summary += 'linearly decreasing amplitude'
                        main_description += 'linearly decreasing amplitude'
                    else:
                        summary += 'amplitude increasing '
                        main_description += 'amplitude increasing '
                        #if los_count > 0:
                        #    main_description += 'approximately '
                        summary += 'linearly away from %f' % lin_location
                        main_description += 'linearly away from %f' % lin_location
                elif lin_count <= len(poly_names):
                    #if los_count > 0:
                    #    main_description += 'approximately '
                    summary += '%sly varying amplitude' % poly_names[lin_count-1]
                    main_description += '%sly varying amplitude' % poly_names[lin_count-1]
                else:
                    summary += 'a variance that '
                    main_description += 'a variance that '
                    #if los_count > 0:
                    #    main_description += 'approximately '
                    summary += 'follows a polynomial of degree %d' % lin_count
                    main_description += 'follows a polynomial of degree %d' % lin_count
            descriptions.append(main_description)
            if los_count > 0:
                if lengthscale > domain_range:
                    descriptions.append('The exact form of the function changes smoothly but very slowly')
                else:
                    descriptions.append('The exact form of the function changes smoothly with a typical lengthscale of %f' % lengthscale)
                if lengthscale < min_period * 0.5:
                    descriptions.append('Since this lengthscale is smaller than half the minimum period over the various components this function may more closely resemble a smooth function without periodicity')
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
    return (summary, descriptions)
    
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

def translate_additive_component(k, X, monotonic, gradient):
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
    (summary, descriptions) = translate_product(k, X, monotonic, gradient)
    # Describe the intervals this kernel acts upon
    intervals = sorted(intervals)
    if len(intervals) == 0:
        summary += '. The combination of changepoint operators is such that this simple AI cannot describe where this component acts; please see visual output or upgrade me'
        interval_description = 'The combination of changepoint operators is such that this simple AI cannot describe where this component acts; please see visual output or upgrade me'
    elif len(intervals) == 1:
        if not intervals == [(-np.Inf, np.Inf)]: 
            summary += '. The function applies %s' % translate_interval(intervals[0])
        interval_description = 'This component applies %s' % translate_interval(intervals[0])
    else:
        summary += '. The function applies %s and %s' % (', '.join(translate_interval(interval) for interval in intervals[:-1]), translate_interval(intervals[-1]))
        interval_description = 'This component applies %s and %s' % (', '.join(translate_interval(interval) for interval in intervals[:-1]), translate_interval(intervals[-1]))
    # Combine and return the descriptions
    if not intervals == [(-np.Inf, np.Inf)]: 
        descriptions.append(interval_description)
    return (summary, descriptions)

def produce_summary_document(dataset_name, n_components, fit_data):
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

\usepackage{graphicx, amsmath, amsfonts, bm, lipsum, capt-of}

\usepackage{natbib, xcolor, wrapfig, booktabs, multirow, caption}

\usepackage{float}

\def\ie{i.e.\ }
\def\eg{e.g.\ }

\\title{An automatic report for the dataset : %(dataset_name)s}

\\author{
James Robert Lloyd\\\\
University of Cambridge
\And
David Duvenaud\\\\
University of Cambridge
\And
Roger Grosse\\\\
Massachussets Institute of Technology
\And
Joshua B. Tenenbaum\\\\
Massachussets Institute of Technology
\And
Zoubin Ghahramani\\\\
University of Cambridge
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

The raw data and full model posterior are shown in figure~\\ref{fig:rawandfit}.

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

The structure search algorithm has identified %(n_components)s additive components in the data
\\begin{itemize}
'''

    text += header % {'dataset_name' : dataset_name, 'n_components' : english_number(n_components)}

    summary_item = '''
  \item \input{figures/%(dataset_name)s/%(dataset_name)s_%(component)d_short_description.tex} 
'''
    for i in range(n_components):
        text += summary_item % {'dataset_name' : dataset_name, 'component' : i+1}
    
    summary_end = '''
\end{itemize}
'''
    
    text += summary_end
    
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

    text += '\nSome comment about cross validated predictive mean absolute error (MAE)?'
    
    text += '''
\\begin{table}[htb]
\\begin{center}
{\small
\\begin{tabular}{|r|r|rrrr|}
\hline
& \multicolumn{1}{c|}{\\bf{Additive components}} & \multicolumn{4}{c|}{\\bf{Cumulative fit}}\\\\
\\bf{\#} & {$R^2$ (\%%)}& {$R^2$ (\%%)} & {Residual $R^2$ (\%%)} & {Cross validated MAE} & Reduction in MAE (\%%)\\\\
\hline
0 & - & - & - & %1.2f & -\\\\
''' % fit_data['MAV_data']

    table_text = '''
%d & %2.1f & %2.1f & %2.1f & %1.2f & %2.1f\\\\
'''

    for i in range(n_components):
        text += table_text % (i+1, fit_data['vars'][i], fit_data['cum_vars'][i], fit_data['cum_resid_vars'][i], fit_data['MAEs'][i], fit_data['MAE_reductions'][i])
        
    text += '''
\hline
\end{tabular}
\caption{
Summary statistics for individual additive component functions and cumulative fits.
The cross validated MAE measures the ability of the model to interpolate and extrapolate (TODO - be less vague).
}
\label{table:stats}
}
\end{center}
\end{table}

\section{Detailed discussion of additive components}
'''

    component_text = '''
\subsection{Component %(component)d}

\input{figures/%(dataset_name)s/%(dataset_name)s_%(component)d_description.tex} 

\\begin{figure}[H]
\\newcommand{\wmgd}{0.5\columnwidth}
\\newcommand{\hmgd}{3.0cm}
\\newcommand{\mdrd}{figures/%(dataset_name)s}
\\newcommand{\mbm}{\hspace{-0.3cm}}
\\begin{tabular}{cc}
\mbm \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_%(component)d} & \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_%(component)d_cum}
\end{tabular}
\caption{Posterior of component %(component)d (left) and posterior of sum of components with data (right)}
\label{fig:comp%(component)d}
\end{figure}
'''

    for i in range(n_components):
        text += component_text % {'dataset_name' : dataset_name, 'component' : i+1}
    
    text += '''
\subsection{Residuals}

Some discussion of the size of the residuals and their independence.

\\begin{figure}[H]
\\newcommand{\wmgd}{0.5\columnwidth}
\\newcommand{\hmgd}{3.0cm}
\\newcommand{\mdrd}{figures/%(dataset_name)s}
\\newcommand{\mbm}{\hspace{-0.3cm}}
\\begin{center}
\\begin{tabular}{c}
\mbm \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(dataset_name)s_resid}
\end{tabular}
\end{center}
\caption{Residuals}
\label{fig:resid}
\end{figure}

\end{document}
''' % {'dataset_name' : dataset_name}

    # Document complete
    return text
    
