import itertools

import flexiblekernel as fk


ONE_D_RULES = [('A', ('+', 'A', 'B'), {'A': 'any', 'B': 'base'}),        # replace with K plus a base kernel
               ('A', ('*', 'A', 'B'), {'A': 'any', 'B': 'base'}),        # replace with K times a base kernel
               ('A', 'B', {'A': 'base', 'B': 'base'}),                   # replace one base kernel with another
               ]

#### FIXME - Code duplication - do we need the OneDGrammar as a special case - else remove
class OneDGrammar:
    def __init__(self):
        self.rules = ONE_D_RULES
    
    def type_matches(self, kernel, tp):
        if tp == 'any':
            return True
        elif tp == 'base':
            return isinstance(kernel, fk.BaseKernel)
        else:
            raise RuntimeError('Unknown type: %s' % tp)
    
    def list_options(self, tp):
        if tp == 'any':
            raise RuntimeError("Can't expand the 'any' type")
        elif tp == 'base':
            return list(fk.base_kernel_families())
        else:
            raise RuntimeError('Unknown type: %s' % tp)
            
#### TODO - make these experiment parameters
####        Also - these are search operators rather than grammar rules - do we need to make this distinction clear?
        
MULTI_D_RULES = [('A', ('+', 'A', 'B'), {'A': 'multi', 'B': 'mask'}),
                 ('A', ('*', 'A', 'B'), {'A': 'multi', 'B': 'mask-not-const'}),
                 ('A', ('*-const', 'A', 'B'), {'A': 'multi', 'B': 'mask-not-const'}), # Might be possible to do this more elegantly
                 ('A', 'B', {'A': 'multi', 'B': 'mask'}),
                 ('A', ('CP', 'A'), {'A': 'multi'}),
                 ('A', ('CB', 'A'), {'A': 'multi'}),
                 ('A', ('B', 'A'), {'A': 'multi'}),
                 ('A', ('BL', 'A'), {'A': 'multi'}),
                 ('A', ('None',), {'A': 'multi'}),
                 ]
    
class MultiDGrammar:
    def __init__(self, ndim, debug=False, base_kernels='SE'):
        self.rules = MULTI_D_RULES
        self.ndim = ndim
        self.debug = debug
        if not debug:
            self.base_kernels = base_kernels
        else:
            self.base_kernels = 'SE'
        
    def type_matches(self, kernel, tp):
        #### FIXME - code duplication
        if tp == 'multi':
            if isinstance(kernel, fk.BaseKernel):
                return False
            elif isinstance(kernel, fk.MaskKernel):
                return True
            elif isinstance(kernel, fk.ChangePointKernel):
                return all([self.type_matches(op, 'multi') for op in kernel.operands])
            elif isinstance(kernel, fk.ChangePointTanhKernel):
                return all([self.type_matches(op, 'multi') for op in kernel.operands])
            elif isinstance(kernel, fk.ChangeBurstTanhKernel):
                return all([self.type_matches(op, 'multi') for op in kernel.operands])
            elif isinstance(kernel, fk.BurstKernel):
                return all([self.type_matches(op, 'multi') for op in kernel.operands])
            elif isinstance(kernel, fk.BurstTanhKernel):
                return all([self.type_matches(op, 'multi') for op in kernel.operands])
            elif isinstance(kernel, fk.BlackoutKernel):
                return all([self.type_matches(op, 'multi') for op in kernel.operands])
            elif isinstance(kernel, fk.BlackoutTanhKernel):
                return all([self.type_matches(op, 'multi') for op in kernel.operands])
            elif isinstance(kernel, fk.SumKernel):
                return all([self.type_matches(op, 'multi') for op in kernel.operands])
            elif isinstance(kernel, fk.ProductKernel):
                return all([self.type_matches(op, 'multi') for op in kernel.operands])
            else:
                raise RuntimeError('Invalid kernel: %s' % kernel.pretty_print())
        elif tp == '1d':
            if isinstance(kernel, fk.BaseKernel):
                return True
            elif isinstance(kernel, fk.MaskKernel):
                return all([self.type_matches(op, '1d') for op in kernel.operands])
            elif isinstance(kernel, fk.ChangePointKernel):
                return all([self.type_matches(op, '1d') for op in kernel.operands])
            elif isinstance(kernel, fk.BurstKernel):
                return all([self.type_matches(op, '1d') for op in kernel.operands])
            elif isinstance(kernel, fk.BlackoutKernel):
                return all([self.type_matches(op, '1d') for op in kernel.operands])
            elif isinstance(kernel, fk.ChangePointTanhKernel):
                return all([self.type_matches(op, '1d') for op in kernel.operands])
            elif isinstance(kernel, fk.ChangeBurstTanhKernel):
                return all([self.type_matches(op, '1d') for op in kernel.operands])
            elif isinstance(kernel, fk.BurstTanhKernel):
                return all([self.type_matches(op, '1d') for op in kernel.operands])
            elif isinstance(kernel, fk.BlackoutTanhKernel):
                return all([self.type_matches(op, '1d') for op in kernel.operands])
            elif isinstance(kernel, fk.SumKernel):
                return all([self.type_matches(op, '1d') for op in kernel.operands])
            elif isinstance(kernel, fk.ProductKernel):
                return all([self.type_matches(op, '1d') for op in kernel.operands])
            else:
                raise RuntimeError('Invalid kernel: %s' % kernel.pretty_print())
        elif tp == 'base':
            return isinstance(kernel, fk.BaseKernel)
        elif tp == 'mask':
            #### FIXME - this is a burst kernel hack
            return (isinstance(kernel, fk.MaskKernel) or (isinstance(kernel, fk.BurstTanhKernel) and all([self.type_matches(op, tp) for op in kernel.operands])))
        elif tp == 'mask-not-const':
            #### FIXME - this is a burst kernel hack
            return ((isinstance(kernel, fk.MaskKernel) and (not isinstance(kernel.base_kernel, fk.ConstKernel))) or (isinstance(kernel, fk.BurstTanhKernel) and all([self.type_matches(op, tp) for op in kernel.operands])))
        else:
            raise RuntimeError('Unknown type: %s' % tp)
        
    def list_options(self, tp):
        if tp in ['1d', 'multi']:
            raise RuntimeError("Can't expand the '%s' type" % tp)
        elif tp == 'base':
            #### FIXME - base and mask should use the same function to ensure they are the same
            return [fam.default() for fam in fk.base_kernel_families(self.base_kernels)]
        elif tp == 'mask':
            return list(fk.base_kernels(self.ndim, self.base_kernels))
        elif tp == 'mask-not-const':
            #### FIXME - this is a burst kernel hack
            return [k for k in list(fk.base_kernels(self.ndim, self.base_kernels)) if ((isinstance(k, fk.MaskKernel) and (not isinstance(k.base_kernel, fk.ConstKernel))) or (isinstance(k, fk.BurstTanhKernel)))]
        else:
            raise RuntimeError('Unknown type: %s' % tp)
    
def replace_all(polish_expr, mapping):
    if type(polish_expr) == tuple:
        return tuple([replace_all(e, mapping) for e in polish_expr])
    elif type(polish_expr) == str:
        if polish_expr in mapping:
            return mapping[polish_expr].copy()
        else:
            return polish_expr
    else:
        assert isinstance(polish_expr, fk.Kernel)
        return polish_expr.copy()
    
def polish_to_kernel(polish_expr):
    if type(polish_expr) == tuple:
        if polish_expr[0] == '+':
            operands = [polish_to_kernel(e) for e in polish_expr[1:]]
            return fk.SumKernel(operands)
        elif polish_expr[0] == '*':
            operands = [polish_to_kernel(e) for e in polish_expr[1:]]
            return fk.ProductKernel(operands)
        elif polish_expr[0] == '*-const':
            operands = [polish_to_kernel(e) for e in polish_expr[1:]]
            # A * (B + Const)
            #### FIXME - assumes 1d - inelegant as well
            return fk.ProductKernel([operands[0], fk.SumKernel([operands[1], fk.MaskKernel(1, 0, fk.ConstKernelFamily().default())])])
        elif polish_expr[0] == 'CP':
            base_kernel = polish_to_kernel(polish_expr[1])
            #### FIXME - there should not be constants here!
            return fk.ChangePointTanhKernel(0., 0., [base_kernel, base_kernel.copy()])
        elif polish_expr[0] == 'CB':
            base_kernel = polish_to_kernel(polish_expr[1])
            #### FIXME - there should not be constants here!
            return fk.ChangeBurstTanhKernel(0., 0., 0., [base_kernel, base_kernel.copy()])
        elif polish_expr[0] == 'B':
            base_kernel = polish_to_kernel(polish_expr[1])
            #### FIXME - there should not be constants here!
            #return fk.BurstTanhKernel(0., 0., 0., [base_kernel])
            #### FIXME - assumes 1d - inelegant as well
            return fk.ChangeBurstTanhKernel(0., 0., 0., [fk.MaskKernel(1, 0, fk.ConstKernelFamily().default()), base_kernel])
        elif polish_expr[0] == 'BL':
            base_kernel = polish_to_kernel(polish_expr[1])
            #### FIXME - there should not be constants here!
            #return fk.BlackoutTanhKernel(0., 0., 0., 0., [base_kernel])
            #### FIXME - assumes 1d - inelegant as well
            return fk.ChangeBurstTanhKernel(0., 0., 0., [base_kernel, fk.MaskKernel(1, 0, fk.ConstKernelFamily().default())])
        elif polish_expr[0] == 'None':
            return fk.NoneKernel()
        else:
            raise RuntimeError('Unknown operator: %s' % polish_expr[0])
    else:
        assert isinstance(polish_expr, fk.Kernel) or (polish_expr is None)
        return polish_expr


def expand_single_tree(kernel, grammar):
    assert isinstance(kernel, fk.Kernel)
    result = []
    for lhs, rhs, types in grammar.rules:
        if grammar.type_matches(kernel, types[lhs]):
            free_vars = types.keys()
            assert lhs in free_vars
            free_vars.remove(lhs)
            choices = itertools.product(*[grammar.list_options(types[v]) for v in free_vars])
            for c in choices:
                mapping = dict(zip(free_vars, c))
                mapping[lhs] = kernel
                full_polish = replace_all(rhs, mapping)
                result.append(polish_to_kernel(full_polish))
    return result

def expand(kernel, grammar):
    result = expand_single_tree(kernel, grammar)
    if isinstance(kernel, fk.BaseKernel):
        pass
    elif isinstance(kernel, fk.MaskKernel):
        result += [fk.MaskKernel(kernel.ndim, kernel.active_dimension, e)
                   for e in expand(kernel.base_kernel, grammar)]
    elif isinstance(kernel, fk.ChangePointKernel):
        for i, op in enumerate(kernel.operands):
            for e in expand(op, grammar):
                new_ops = kernel.operands[:i] + [e] + kernel.operands[i+1:]
                new_ops = [op.copy() for op in new_ops]
                result.append(fk.ChangePointKernel(kernel.location, kernel.steepness, new_ops))
    elif isinstance(kernel, fk.BurstKernel):
        for i, op in enumerate(kernel.operands):
            for e in expand(op, grammar):
                new_ops = kernel.operands[:i] + [e] + kernel.operands[i+1:]
                new_ops = [op.copy() for op in new_ops]
                result.append(fk.BurstKernel(kernel.location, kernel.steepness, kernel.width, new_ops))
    elif isinstance(kernel, fk.BlackoutKernel):
        for i, op in enumerate(kernel.operands):
            for e in expand(op, grammar):
                new_ops = kernel.operands[:i] + [e] + kernel.operands[i+1:]
                new_ops = [op.copy() for op in new_ops]
                result.append(fk.BlackoutKernel(kernel.location, kernel.steepness, kernel.width, kernel.sf, new_ops))
    elif isinstance(kernel, fk.ChangePointTanhKernel):
        for i, op in enumerate(kernel.operands):
            for e in expand(op, grammar):
                new_ops = kernel.operands[:i] + [e] + kernel.operands[i+1:]
                new_ops = [op.copy() for op in new_ops]
                result.append(fk.ChangePointTanhKernel(kernel.location, kernel.steepness, new_ops))
    elif isinstance(kernel, fk.ChangeBurstTanhKernel):
        for i, op in enumerate(kernel.operands):
            for e in expand(op, grammar):
                new_ops = kernel.operands[:i] + [e] + kernel.operands[i+1:]
                new_ops = [op.copy() for op in new_ops]
                result.append(fk.ChangeBurstTanhKernel(kernel.location, kernel.steepness, kernel.width, new_ops))
    elif isinstance(kernel, fk.BurstTanhKernel):
        for i, op in enumerate(kernel.operands):
            for e in expand(op, grammar):
                new_ops = kernel.operands[:i] + [e] + kernel.operands[i+1:]
                new_ops = [op.copy() for op in new_ops]
                result.append(fk.BurstTanhKernel(kernel.location, kernel.steepness, kernel.width, new_ops))
    elif isinstance(kernel, fk.BlackoutTanhKernel):
        for i, op in enumerate(kernel.operands):
            for e in expand(op, grammar):
                new_ops = kernel.operands[:i] + [e] + kernel.operands[i+1:]
                new_ops = [op.copy() for op in new_ops]
                result.append(fk.BlackoutTanhKernel(kernel.location, kernel.steepness, kernel.width, kernel.sf, new_ops))
    elif isinstance(kernel, fk.SumKernel):
        for i, op in enumerate(kernel.operands):
            for e in expand(op, grammar):
                new_ops = kernel.operands[:i] + [e] + kernel.operands[i+1:]
                new_ops = [op.copy() for op in new_ops]
                result.append(fk.SumKernel(new_ops))
    elif isinstance(kernel, fk.ProductKernel):
        for i, op in enumerate(kernel.operands):
            for e in expand(op, grammar):
                new_ops = kernel.operands[:i] + [e] + kernel.operands[i+1:]
                new_ops = [op.copy() for op in new_ops]
                result.append(fk.ProductKernel(new_ops))
    else:
        raise RuntimeError('Unknown kernel class:', kernel.__class__)
    return result

def canonical(kernel):
    '''Sorts a kernel tree into a canonical form.'''
    if isinstance(kernel, fk.BaseKernel):
        return kernel.copy()
    elif isinstance(kernel, fk.MaskKernel):
        if isinstance(canonical(kernel.base_kernel), fk.NoneKernel):
            return fk.NoneKernel()
        else:
            return fk.MaskKernel(kernel.ndim, kernel.active_dimension, canonical(kernel.base_kernel))
    elif isinstance(kernel, fk.ChangePointKernel):
        canop = [canonical(o) for o in kernel.operands]
        if isinstance(canop[0], fk.NoneKernel) or isinstance(canop[1], fk.NoneKernel):
            #### TODO - might want to allow the zero kernel to appear i.e. change point to zero?
            return fk.NoneKernel()
        else:
            return fk.ChangePointKernel(kernel.location, kernel.steepness, canop)
    elif isinstance(kernel, fk.BurstKernel):
        canop = [canonical(o) for o in kernel.operands]
        if isinstance(canop[0], fk.NoneKernel):
            return fk.NoneKernel()
        else:
            return fk.BurstKernel(kernel.location, kernel.steepness, kernel.width, canop)
    elif isinstance(kernel, fk.BlackoutKernel):
        canop = [canonical(o) for o in kernel.operands]
        if isinstance(canop[0], fk.NoneKernel):
            return fk.NoneKernel()
        else:
            return fk.BlackoutKernel(kernel.location, kernel.steepness, kernel.width, kernel.sf, canop)
    elif isinstance(kernel, fk.ChangePointTanhKernel):
        canop = [canonical(o) for o in kernel.operands]
        if isinstance(canop[0], fk.NoneKernel) or isinstance(canop[1], fk.NoneKernel):
            #### TODO - might want to allow the zero kernel to appear
            return fk.NoneKernel()
        else:
            return fk.ChangePointTanhKernel(kernel.location, kernel.steepness, canop)
    elif isinstance(kernel, fk.ChangeBurstTanhKernel):
        canop = [canonical(o) for o in kernel.operands]
        if isinstance(canop[0], fk.NoneKernel) or isinstance(canop[1], fk.NoneKernel):
            #### TODO - might want to allow the zero kernel to appear
            return fk.NoneKernel()
        else:
            return fk.ChangeBurstTanhKernel(kernel.location, kernel.steepness, kernel.width, canop)
    elif isinstance(kernel, fk.BurstTanhKernel):
        canop = [canonical(o) for o in kernel.operands]
        if isinstance(canop[0], fk.NoneKernel):
            return fk.NoneKernel()
        else:
            return fk.BurstTanhKernel(kernel.location, kernel.steepness, kernel.width, canop)
    elif isinstance(kernel, fk.BlackoutTanhKernel):
        canop = [canonical(o) for o in kernel.operands]
        if isinstance(canop[0], fk.NoneKernel):
            return fk.NoneKernel()
        else:
            return fk.BlackoutTanhKernel(kernel.location, kernel.steepness, kernel.width, kernel.sf, canop)
    elif isinstance(kernel, fk.SumKernel):
        new_ops = []
        for op in kernel.operands:
            op_canon = canonical(op)
            if isinstance(op_canon, fk.SumKernel):
                new_ops += op_canon.operands
            elif not isinstance(op_canon, fk.NoneKernel):
                new_ops.append(op_canon)
        if len(new_ops) == 0:
            return fk.NoneKernel()
        elif len(new_ops) == 1:
            return new_ops[0]
        else:
            return fk.SumKernel(sorted(new_ops))
    elif isinstance(kernel, fk.ProductKernel):
        new_ops = []
        for op in kernel.operands:
            op_canon = canonical(op)
            if isinstance(op_canon, fk.ProductKernel):
                new_ops += op_canon.operands
            elif not isinstance(op_canon, fk.NoneKernel):
                new_ops.append(op_canon)
        if len(new_ops) == 0:
            return fk.NoneKernel()
        elif len(new_ops) == 1:
            return new_ops[0]
        else:
            return fk.ProductKernel(sorted(new_ops))
    else:
        raise RuntimeError('Unknown kernel class:', kernel.__class__)

def remove_duplicates(kernels):
    # This is possible since kernels are now hashable
    return list(set(kernels))
    
def expand_kernels(D, seed_kernels, verbose=False, debug=False, base_kernels='SE'):    
    '''Makes a list of all expansions of a set of kernels in D dimensions.'''
    g = MultiDGrammar(D, debug=debug, base_kernels=base_kernels)
    if verbose:
        print 'Seed kernels :'
        for k in seed_kernels:
            print k.pretty_print()
    kernels = []
    for k in seed_kernels:
        kernels = kernels + expand(k, g)
    kernels = remove_duplicates(kernels)
    kernels = [k for k in kernels if not isinstance(k, fk.NoneKernel)]
    if verbose:
        print 'Expanded kernels :'
        for k in kernels:
            print k.pretty_print()
    return (kernels)



            
