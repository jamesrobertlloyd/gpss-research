import unittest

import numpy as np

import experiment
import flexible_function as ff
import grammar
#import translation

class ff_testcase(unittest.TestCase):

    def test_noise_kernel(self):
        k = ff.NoiseKernel()
        print '\n', k.pretty_print(), '\n'
        print '\n', k.syntax, '\n'
        print '\n', k, '\n'
        print '\n', k.get_gpml_expression(dimensions=3), '\n'
        k.initialise_params(data_shape = {'y_sd' : 0})
        print '\n', k, '\n'
        k = k.copy()
        print '\n', k, '\n'
        assert k == k.copy()
        k.load_param_vector(k.param_vector)
        print '\n', k, '\n'

    def test_sq_exp(self):
        k = ff.SqExpKernel()
        print '\n', k.pretty_print(), '\n'
        print '\n', k.syntax, '\n'
        print '\n', k, '\n'
        k.dimension = 1
        print '\n', k.get_gpml_expression(dimensions=3), '\n'
        k.initialise_params(data_shape = {'y_sd' : 0, 'x_sd' : [0,2], 'x_min' : [-10,-100], 'x_max' : [10,100]})
        print '\n', k, '\n'
        assert k == k.copy()
        k = k.copy()
        print '\n', k, '\n'
        assert k == k.copy()
        k.load_param_vector(k.param_vector)
        print '\n', k, '\n'

    def test_sum(self):
        k = ff.SqExpKernel()
        k1 = k.copy()
        k2 = k.copy()
        k = k1 + k2
        print '\n', k.pretty_print(), '\n'
        print '\n', k.syntax, '\n'
        print '\n', k, '\n'
        k.operands[0].dimension = 0
        k.operands[1].dimension = 1
        print '\n', k.pretty_print(), '\n'
        print '\n', k.syntax, '\n'
        print '\n', k, '\n'
        print '\n', k.get_gpml_expression(dimensions=3), '\n'
        k.initialise_params(data_shape = {'y_sd' : 0, 'x_sd' : [0,2], 'x_min' : [-10,-100], 'x_max' : [10,100]})
        print '\n', k, '\n'
        assert k == k.copy()
        k = k.copy()
        print '\n', k, '\n'
        assert k == k.copy()
        k.load_param_vector(k.param_vector)
        print '\n', k, '\n'
        k = k + k.copy()
        print '\n', k.pretty_print(), '\n'
        print '\n', k.syntax, '\n'
        print '\n', k, '\n'
        print '\n', k.pretty_print(), '\n'
        print '\n', k.syntax, '\n'
        print '\n', k, '\n'
        print '\n', k.get_gpml_expression(dimensions=3), '\n'
        k.initialise_params(data_shape = {'y_sd' : 0, 'x_sd' : [0,2], 'x_min' : [-10,-100], 'x_max' : [10,100]})
        print '\n', k, '\n'
        assert k == k.copy()
        k = k.copy()
        print '\n', k, '\n'
        assert k == k.copy()
        k.load_param_vector(k.param_vector)
        print '\n', k, '\n'
        k.sf

    def test_prod(self):
        k = ff.SqExpKernel()
        k1 = k.copy()
        k2 = k.copy()
        k = k1 * k2
        print '\n', k.pretty_print(), '\n'
        print '\n', k.syntax, '\n'
        print '\n', k, '\n'
        k.operands[0].dimension = 0
        k.operands[1].dimension = 1
        print '\n', k.pretty_print(), '\n'
        print '\n', k.syntax, '\n'
        print '\n', k, '\n'
        print '\n', k.get_gpml_expression(dimensions=3), '\n'
        k.initialise_params(data_shape = {'y_sd' : 0, 'x_sd' : [0,2], 'x_min' : [-10,-100], 'x_max' : [10,100]})
        print '\n', k, '\n'
        assert k == k.copy()
        k = k.copy()
        print '\n', k, '\n'
        assert k == k.copy()
        k.load_param_vector(k.param_vector)
        print '\n', k, '\n'
        k = k + k.copy()
        print '\n', k.pretty_print(), '\n'
        print '\n', k.syntax, '\n'
        print '\n', k, '\n'
        print '\n', k.pretty_print(), '\n'
        print '\n', k.syntax, '\n'
        print '\n', k, '\n'
        print '\n', k.get_gpml_expression(dimensions=3), '\n'
        k.initialise_params(data_shape = {'y_sd' : 0, 'x_sd' : [0,2], 'x_min' : [-10,-100], 'x_max' : [10,100]})
        print '\n', k, '\n'
        assert k == k.copy()
        k = k.copy()
        print '\n', k, '\n'
        assert k == k.copy()
        k.load_param_vector(k.param_vector)
        print '\n', k, '\n'
        k.sf

    def test_model(self):
        print 'model'
        m = ff.MeanZero()
        k = ff.SqExpKernel()
        l = ff.LikGauss()
        regression_model = ff.GPModel(mean=m, kernel=k, likelihood=l, nll=0, ndata=100)
        print '\n', regression_model.pretty_print(), '\n'
        print '\n', regression_model.__repr__(), '\n'
        print regression_model.bic
        print regression_model.aic
        print regression_model.pl2
        print ff.GPModel.score(regression_model, criterion='nll')

    def test_base(self):
        kernels = ff.base_kernels_without_dimension('SE,Const,Noise')
        for k in kernels:
            print '\n', k.pretty_print(), '\n'
        kernels = ff.base_kernels(3, 'SE,Const,Noise')
        for k in kernels:
            print '\n', k.pretty_print(), '\n'

    def test_repr(self):
        m = ff.MeanZero()
        k = ff.SqExpKernel()
        l = ff.LikGauss()
        regression_model = ff.GPModel(mean=m, kernel=k, likelihood=l)
        print regression_model
        print  ff.repr_to_model(regression_model.__repr__())
        assert regression_model == ff.repr_to_model(regression_model.__repr__())

    def test_collapse_add_idempotent(self):
        k = ff.SqExpKernel()
        k1 = k.copy()
        k2 = k.copy()
        k = ff.NoiseKernel(sf=-1)
        k3 = k.copy()
        k4 = k.copy()
        k = ff.ConstKernel(sf=1)
        k5 = k.copy()
        k6 = k.copy()
        k = k1 + k2 + k3 + k4 + k5 + k6
        print '\n', k.pretty_print(), '\n'
        k = k.collapse_additive_idempotency()
        print '\n', k.pretty_print(), '\n'

    def test_collapse_mult_idempotent(self):
        k = ff.SqExpKernel(dimension=0, lengthscale=0, sf=1)
        k1 = k.copy()
        k2 = k.copy()
        k = k1 * k2
        print '\n', k.pretty_print(), '\n'
        k = k.collapse_multiplicative_idempotency()
        assert (isinstance(k, ff.SqExpKernel)) and (k.dimension == 0)
        print '\n', k.pretty_print(), '\n'
        k = ff.SqExpKernel(dimension=0, lengthscale=0, sf=1)
        k1 = k.copy()
        k2 = k.copy()
        k = ff.SqExpKernel(dimension=1, lengthscale=2, sf=2)
        k3 = k.copy()
        k4 = k.copy()
        k5 = ff.NoiseKernel(sf=-1)
        k6 = ff.ConstKernel(sf=1)
        k = k1 * k2 * k3 * k4 * k5 * k5.copy() + k6 + k6.copy()
        print '\n', k.pretty_print(), '\n'
        k = k.collapse_multiplicative_idempotency()
        print '\n', k.pretty_print(), '\n'
        k = k1 * k2 * k3 * k4 * k5 * k5.copy() * k6 * k6.copy()
        print '\n', k.pretty_print(), '\n'
        k = k.collapse_multiplicative_idempotency()
        print '\n', k.pretty_print(), '\n'

    def test_collapse_zero(self):
        k1 = ff.SqExpKernel(dimension=0, lengthscale=0, sf=1)
        k2 = ff.NoiseKernel(sf=-1)
        k = k1 * k2
        print '\n', k.pretty_print(), '\n'
        k = k.collapse_multiplicative_zero()
        assert isinstance(k, ff.NoiseKernel)
        print '\n', k.pretty_print(), '\n'
        k = (k1 + k1.copy() + k1.copy() * k2.copy()) * k2
        print (k1 + k1.copy()).sf
        print (k1.copy() * k2.copy()).sf
        print (k1 + k1.copy() + k1.copy() * k2.copy()).sf
        print k.sf
        print '\n', k.pretty_print(), '\n'
        k = k.collapse_multiplicative_zero()
        assert isinstance(k, ff.NoiseKernel)
        print '\n', k.pretty_print(), '\n'

    def test_collapse_identity(self):
        print 'collapse identity'
        k1 = ff.SqExpKernel(dimension=0, lengthscale=0, sf=1)
        k2 = ff.ConstKernel(sf=-1)
        k = k1 * k2
        print '\n', k.pretty_print(), '\n'
        k = k.collapse_multiplicative_identity()
        assert isinstance(k, ff.SqExpKernel)
        print '\n', k.pretty_print(), '\n'
        k1 = ff.SqExpKernel(dimension=0, lengthscale=0, sf=1)
        k = (k1 + k1.copy() + k1.copy() * k2.copy()) * k2
        print (k1 + k1.copy()).sf
        print (k1.copy() * k2.copy()).sf
        print (k1 + k1.copy() + k1.copy() * k2.copy()).sf
        print k.sf
        print '\n', k.pretty_print(), '\n'
        k = k.collapse_multiplicative_identity()
        print '\n', k.pretty_print(), '\n'

    def test_simplified_k(self):
        print 'simplified_k'
        k = ff.SqExpKernel(dimension=0, lengthscale=0, sf=1)
        k1 = k.copy()
        k2 = k.copy()
        k = k1 * k2
        k = ff.SqExpKernel(dimension=0, lengthscale=0, sf=1)
        k1 = k.copy()
        k2 = k.copy()
        k = ff.SqExpKernel(dimension=1, lengthscale=2, sf=2)
        k3 = k.copy()
        k4 = k.copy()
        k5 = ff.NoiseKernel(sf=-1)
        k6 = ff.ConstKernel(sf=1)
        k = k1 * k2 * k3 * k4 * k5 * k5.copy() + k6 + k6.copy() + k1.copy() * k1.copy() * k3.copy()
        print '\n', k.pretty_print(), '\n'
        k = k.simplified()
        print '\n', k.pretty_print(), '\n'

    def test_distribute_products_k(self):
        print 'distribute'
        k = ff.SqExpKernel(dimension=0, lengthscale=0, sf=1)
        k1 = k.copy()
        k2 = k.copy()
        k = ff.SqExpKernel(dimension=1, lengthscale=2, sf=2)
        k3 = k.copy()
        k4 = k.copy()
        k5 = ff.NoiseKernel(sf=-1)
        k6 = ff.ConstKernel(sf=1)
        k = (k1 + k2 + k3) * (k4 + k5)
        print '\n', k.pretty_print(), '\n'
        components = k.distribute_products().simplified()
        print components
        print components.collapse_additive_idempotency()
        for k in components.operands:
            print '\n', k.pretty_print(), '\n'
        k = ff.SqExpKernel(dimension=0, lengthscale=0, sf=1)
        k1 = k.copy()
        k2 = k.copy()
        k = ff.SqExpKernel(dimension=1, lengthscale=2, sf=2)
        k3 = k.copy()
        k4 = k.copy()
        k5 = ff.NoiseKernel(sf=-1)
        k6 = ff.ConstKernel(sf=1)
        k = (k1 * (k2 + k3)) + (k4 * k5)
        print '\n', k.pretty_print(), '\n'
        components = k.distribute_products().simplified()
        print components
        print components.collapse_additive_idempotency()
        for k in components.operands:
            print '\n', k.pretty_print(), '\n'

    def test_jitter(self):
        print 'jitter'
        k = ff.SqExpKernel(dimension=0, lengthscale=0, sf=1)
        k1 = k.copy()
        k2 = k.copy()
        print [k,k1,k2]
        assert (k == k1) and (k == k2) and (k1 == k2)
        ff.add_jitter_k([k1, k2])
        assert (not k == k1) and (not k == k2) and (not k1 == k2)
        print [k,k1,k2]

    def test_jitter_model(self):
        print 'jitter model'
        k = ff.SqExpKernel(dimension=0, lengthscale=0, sf=1)
        k1 = k.copy()
        k2 = k.copy()
        print [k,k1,k2]
        assert (k == k1) and (k == k2) and (k1 == k2)
        m1 = ff.GPModel(kernel=k1)
        m2 = ff.GPModel(kernel=k2)
        ff.add_jitter([m1, m2])
        assert (not k == k1) and (not k == k2) and (not k1 == k2)
        print [k,k1,k2]

    def test_restarts(self):
        print 'restart'
        data_shape = {'y_sd' : 0, 'x_sd' : [0,2], 'x_min' : [-10,-100], 'x_max' : [10,100]}
        k = ff.SqExpKernel(dimension=0)
        k1 = k.copy()
        k2 = k.copy()
        print [k,k1,k2]
        assert (k == k1) and (k == k2) and (k1 == k2)
        kernel_list = ff.add_random_restarts_k([k1, k2], data_shape=data_shape, sd=1)
        k1 = kernel_list[0]
        k2 = kernel_list[1]
        assert (not k == k1) and (not k == k2) and (not k1 == k2)
        print [k,k1,k2]

    def test_restarts_model(self):
        print 'restart model'
        data_shape = {'y_sd' : 0, 'x_sd' : [0,2], 'x_min' : [-10,-100], 'x_max' : [10,100]}
        k = ff.SqExpKernel(dimension=0)
        k1 = k.copy()
        k2 = k.copy()
        print [k,k1,k2]
        assert (k == k1) and (k == k2) and (k1 == k2)
        m1 = ff.GPModel(kernel=k1)
        m2 = ff.GPModel(kernel=k2)
        model_list = ff.add_random_restarts([m1, m2], n_rand=1, data_shape=data_shape, sd=1)
        k1 = model_list[0].kernel
        k2 = model_list[1].kernel
        assert (not k == k1) and (not k == k2) and (not k1 == k2)
        print [k,k1,k2]

    def test_additive_form_k(self):
        print 'additive form'
        k = ff.SqExpKernel(dimension=0, lengthscale=0, sf=1)
        k1 = k.copy()
        k2 = k.copy()
        k = ff.SqExpKernel(dimension=1, lengthscale=2, sf=2)
        k3 = k.copy()
        k4 = k.copy()
        k5 = ff.NoiseKernel(sf=-1)
        k6 = ff.ConstKernel(sf=1)
        k = (k1 * (k2 + k3)) + (k4 * k5)
        print '\n', k.pretty_print(), '\n'
        components = k.additive_form().simplified()
        print components
        for k in components.operands:
            print '\n', k.pretty_print(), '\n'

    def test_canonical_k(self):
        print 'canonical_k form'
        k = ff.SqExpKernel(dimension=0, lengthscale=0, sf=1)
        k1 = k.copy()
        k2 = k.copy()
        k = ff.SqExpKernel(dimension=1, lengthscale=2, sf=2)
        k3 = k.copy()
        k4 = k.copy()
        k5 = ff.NoiseKernel(sf=-1)
        k6 = ff.ConstKernel(sf=1)
        k = k1 * k2 * k3 * k4 * k5 * k5.copy() + k6 + k6.copy() + k1.copy() * k1.copy() * k3.copy()
        print '\n', k.pretty_print(), '\n'
        print '\n', k.canonical().pretty_print(), '\n'

    def test_canonical_k_2(self):
        print 'canonical_k form 2'
        k = ff.SqExpKernel(dimension=0, lengthscale=0, sf=1)
        k1 = k.copy()
        k2 = ff.NoneKernel()
        k = ff.ChangePointKernel(operands=[k1,k2])
        print '\n', k, '\n'
        k = k.canonical()
        print '\n', k, '\n'
        assert k == k1
        k = ff.ChangePointKernel(operands=[ff.ChangePointKernel(operands=[k1,k2]),k2])
        print '\n', k, '\n'
        k = k.canonical()
        print '\n', k, '\n'
        assert k == k1

    def test_hash_and_cmp(self):
        k = ff.SqExpKernel(dimension=0, lengthscale=0, sf=1)
        k1 = k.copy()
        k2 = k.copy()
        k3 = ff.SqExpKernel(dimension=1, lengthscale=0, sf=1)
        assert sorted(list(set([k1,k2,k3]))) == sorted([k1,k3])
        k = ff.SqExpKernel(dimension=0, lengthscale=0, sf=1)
        k4 = k.copy()
        k5 = ff.NoneKernel()
        k6 = ff.ChangePointKernel(operands=[k4,k5])
        k7 = ff.ChangePointKernel(operands=[ff.ChangePointKernel(operands=[k4,k5]),k1])
        assert sorted([k1,k2,k3,k4,k5,k6,k7]) == sorted([k7,k1,k6,k2,k5,k3,k4])
        assert sorted(k.canonical() for k in [k1,k2,k3,k4,k5,k6,k7]) == sorted(k.canonical() for k in [k7,k1,k6,k2,k5,k3,k4])
        assert sorted(k.additive_form() for k in [k1,k2,k3,k4,k5,k6,k7]) == sorted(k.additive_form() for k in [k7,k1,k6,k2,k5,k3,k4])

class grammar_testcase(unittest.TestCase):

    def test_expand(self):
        print 'expand'
        print '1d'
        k = ff.SqExpKernel(dimension=0, lengthscale=0, sf=0)
        expanded = grammar.expand_kernels(1, [k], base_kernels='SE', rules=None)
        for k in expanded:
            print '\n', k.pretty_print(), '\n'
        print '2d'
        k = ff.SqExpKernel(dimension=0, lengthscale=0, sf=0)
        expanded = grammar.expand_kernels(2, [k], base_kernels='SE', rules=None)
        for k in expanded:
            print '\n', k.pretty_print(), '\n'
        print '3d'
        k = ff.SqExpKernel(dimension=0, lengthscale=0, sf=0)
        expanded = grammar.expand_kernels(3, [k], base_kernels='SE', rules=None)
        for k in expanded:
            print '\n', k.pretty_print(), '\n'
        print '3d with two SEs'
        k = ff.SqExpKernel(dimension=0, lengthscale=0, sf=0)
        expanded = grammar.expand_kernels(3, [k + k.copy()], base_kernels='SE', rules=None)
        for k in expanded:
            print '\n', k.pretty_print(), '\n'

    def test_expand_model(self):
        print 'expand model'
        print '2d'
        k = ff.SqExpKernel(dimension=0, lengthscale=0, sf=0)
        m = ff.GPModel(mean=ff.MeanZero(), kernel=k, likelihood=ff.LikGauss())
        expanded = grammar.expand_models(2, [m], base_kernels='SE', rules=None)
        for k in expanded:
            print '\n', k.pretty_print(), '\n'

class experiment_testcase(unittest.TestCase):

    def test_nan_score(self):
        k = ff.SqExpKernel(dimension=0, lengthscale=0, sf=0)
        m1 = ff.GPModel(kernel=k, nll=np.nan, ndata=100)
        m2 = ff.GPModel(kernel=k.copy(), nll=0, ndata=100)
        (not_nan, eq_nan) = experiment.remove_nan_scored_models([m1,m2], score='bic')
        assert (len(not_nan) == 1) and (len(eq_nan) == 1)

class misc_testcase(unittest.TestCase):

    def test_3_operands_to_binary(self):
        assert len(ff.ChangePointKernel(operands=[ff.ConstKernel(), ff.ChangePointKernel(operands=[ff.ConstKernel(), ff.ConstKernel()])]).canonical().operands) == 2

    def test_simplify(self):
        m = ff.GPModel(mean=ff.MeanZero(), kernel=ff.SumKernel(operands=[ff.ProductKernel(operands=[ff.ConstKernel(sf=0.170186999131), ff.SqExpKernel(dimension=0, lengthscale=1.02215322228, sf=5.9042619611)]), ff.ProductKernel(operands=[ff.NoiseKernel(sf=2.43188502201), ff.ConstKernel(sf=-0.368638271154)]), ff.ProductKernel(operands=[ff.NoiseKernel(sf=1.47110516981), ff.PeriodicKernel(dimension=0, lengthscale=-1.19651800365, period=0.550394248167, sf=0.131044872864)]), ff.ProductKernel(operands=[ff.SqExpKernel(dimension=0, lengthscale=3.33346140605, sf=3.7579461353), ff.PeriodicKernel(dimension=0, lengthscale=0.669624964607, period=0.00216264543496, sf=2.41995024965)])]), likelihood=ff.LikGauss(sf=-np.inf), nll=599.59757993, ndata=144)
        assert not m.simplified() == m
        m = ff.GPModel(mean=ff.MeanZero(), kernel=ff.SumKernel(operands=[ff.ProductKernel(operands=[ff.ConstKernel(sf=0.170186999131), ff.SqExpKernel(dimension=0, lengthscale=1.02215322228, sf=5.9042619611)]), ff.ProductKernel(operands=[ff.NoiseKernel(sf=2.43188502201), ff.ConstKernel(sf=-0.368638271154)])]), likelihood=ff.LikGauss(sf=-np.inf), nll=599.59757993, ndata=144)
        assert not m.simplified() == m

    def test_param_loading(self):
        k = ff.ChangePointKernel(dimension=0, location=0, steepness=0, operands=[ff.ConstKernel(sf=0), ff.ConstKernel(sf=0)])
        param_vector = [1,1,1,1]
        k.load_param_vector(param_vector)
        assert np.all(k.param_vector == param_vector)
        param_vector = [0,0,0,0]
        assert not np.any(k.param_vector == param_vector)

    # def test_wrong_dimension(self):
    #     try:
    #         k = fk.MaskKernelFamily(1,1,fk.SqExpKernelFamily())
    #     except:
    #         pass
    #     else:
    #         raise RuntimeError('I gave a mask kernel inconsistent number of dimensions and active dimension')

    # #def test_none_dimensions(self):
    # #    k = fk.MaskKernelFamily(None,None,fk.SqExpKernelFamily())

    # def test_addition(self):
    #     k = fk.SqExpKernelFamily().default() + fk.SqExpKernelFamily().default()
    #     assert isinstance(k, fk.SumKernel)

    # def test_creation(self):
    #     # Check that both of these ways of creating a kernel work
    #     k = fk.SqExpKernelFamily().default()
    #     k = fk.SqExpKernelFamily.default()

    # def test_addition_2(self):
    #     k = fk.SqExpKernelFamily().default() + fk.SqExpKernelFamily().default()
    #     k = k + k.copy()
    #     assert isinstance(k, fk.SumKernel) and (not isinstance(k.operands[0], fk.SumKernel))

    # def test_multiplication(self):
    #     k = fk.SqExpKernelFamily().default() * fk.SqExpKernelFamily().default()
    #     assert isinstance(k, fk.ProductKernel)

    # def test_addition_2(self):
    #     k = fk.SqExpKernelFamily().default() * fk.SqExpKernelFamily().default()
    #     k = k * k.copy()
    #     assert isinstance(k, fk.ProductKernel) and (not isinstance(k.operands[0], fk.ProductKernel))

    # def test_defaults(self):
    #     k = fk.SqExpKernelFamily()
    #     dummy = fk.ChangePointTanhKernelFamily(operands=[k,k]).default()
    #     dummy = fk.ChangeBurstTanhKernelFamily(operands=[k,k]).default()
    #     dummy = (k.default() + k.default()).family().default()
    #     dummy = (k.default() * k.default()).family().default()

    # def test_default_family_default(self):
    #     k = fk.SqExpKernelFamily()
    #     assert (k.default() * k.default()).family().default() == (k.default() * k.default())

# class experiment_testcase(unittest.TestCase):

#     def test_nan_score(self):
#         k1 = fk.ScoredKernel(fk.SqExpKernelFamily.default())
#         k2 = fk.ScoredKernel(fk.SqExpKernelFamily.default(), bic_nle=0)
#         (not_nan, eq_nan) = experiment.remove_nan_scored_kernels([k1,k2], score='bic')
#         assert (len(not_nan) == 1) and (len(eq_nan) == 1)

# class grammar_testcase(unittest.TestCase):

#     def test_type_match(self):
#         g = grammar.MultiDGrammar(ndim=2)
#         k = fk.MaskKernel(2,0,fk.SqExpKernelFamily.default())
#         assert g.type_matches(k, 'multi')
#         k = fk.MaskKernel(2,0,fk.FourierKernelFamily.default())
#         assert g.type_matches(k, 'multi')
#         k = k + k.copy()
#         assert g.type_matches(k, 'multi')
#         k = k * k.copy()
#         assert g.type_matches(k, 'multi')
#         k = fk.MaskKernel(2,0,fk.SqExpKernelFamily.default()).family()
#         k = fk.ChangePointTanhKernelFamily(operands=[k,k]).default()
#         assert g.type_matches(k, 'multi')
#         k = fk.MaskKernel(2,0,fk.SqExpKernelFamily.default()).family()
#         k = fk.ChangeBurstTanhKernelFamily(operands=[k,k]).default()
#         assert g.type_matches(k, 'multi')

# class translation_testcase(unittest.TestCase):

#     def test_SE(self):
#         k = fk.SqExpKernelFamily().default()
#         sentences = translation.translate_additive_component(k, np.array([-1,0,1]), monotonic=0, gradient=0, unit='year')

#     def test_SE_metres(self):
#         k = fk.SqExpKernelFamily().default()
#         sentences = translation.translate_additive_component(k, np.array([-1,0,1]), monotonic=0, gradient=0, unit='metre')

#     def test_SE_number(self):
#         k = fk.SqExpKernelFamily().default()
#         sentences = translation.translate_additive_component(k, np.array([-1,0,1]), monotonic=0, gradient=0, unit='number')

#     def test_SE_no_unit(self):
#         k = fk.SqExpKernelFamily().default()
#         sentences = translation.translate_additive_component(k, np.array([-1,0,1]), monotonic=0, gradient=0, unit='')
        
#     def test_BroadSE(self):
#         k = fk.SqExpKernelFamily().default()
#         sentences = translation.translate_additive_component(k, np.array([0,0.5]), monotonic=0, gradient=0, unit='year')
        
#     def test_poly(self):
#         k = fk.LinKernelFamily().default() * fk.LinKernelFamily().default() * fk.LinKernelFamily().default()
#         sentences = translation.translate_additive_component(k, np.array([0,0.5]), monotonic=0, gradient=0, unit='year')
        
#     def test_SEpolydecrease(self):
#         k = fk.SqExpKernelFamily().default() * fk.LinKernelFamily().default()
#         sentences = translation.translate_additive_component(k, np.array([0.2,0.5]), monotonic=0, gradient=0, unit='year')
        
#     def test_complicated(self):
#         k = fk.SqExpKernelFamily().default() * fk.CentredPeriodicKernelFamily().default() * fk.CosineKernelFamily().default() * fk.CosineKernelFamily().default() * fk.LinKernelFamily().default() * fk.LinKernelFamily().default()
#         op = [fk.ZeroKernel(), k]
#         k = fk.ChangePointTanhKernel(location = 1.5, steepness=2, operands=op)
#         op = [k, fk.ZeroKernel()]
#         k = fk.ChangePointTanhKernel(location = 1.8, steepness=2, operands=op)
#         sentences = translation.translate_additive_component(k, np.array([1,2]), monotonic=0, gradient=0, unit='year')
        
#     def test_IMT3(self):
#         k = fk.IMT3LinKernelFamily().default()
#         sentences = translation.translate_additive_component(k, np.array([1,2]), monotonic=0, gradient=0, unit='year')
        
#     def test_Const(self):
#         k = fk.ConstKernelFamily().default()
#         sentences = translation.translate_additive_component(k, np.array([1,2]), monotonic=0, gradient=0, unit='year')
        
#     def test_ConstSE(self):
#         k = fk.ConstKernelFamily().default() * fk.SqExpKernelFamily().default()
#         sentences = translation.translate_additive_component(k, np.array([1,2]), monotonic=0, gradient=0, unit='year')
        
#     def test_Window(self):
#         k = fk.SqExpKernelFamily().default()
#         op = [k, fk.ZeroKernel()]
#         k = fk.ChangeBurstTanhKernel(location = 1.5, steepness=2, width=np.log(0.2), operands=op)
#         sentences = translation.translate_additive_component(k, np.array([1,2]), monotonic=0, gradient=0, unit='year')
        
#     def test_cos(self):
#         k = fk.CosineKernelFamily().default()
#         sentences = translation.translate_additive_component(k, np.array([1,2]), monotonic=0, gradient=0, unit='year')
        
#     def test_Window2(self):
#         k = fk.SqExpKernelFamily().default()
#         op = [fk.ZeroKernel(), k]
#         k = fk.ChangeBurstTanhKernel(location = 1.5, steepness=2, width=np.log(0.2), operands=op)
#         sentences = translation.translate_additive_component(k, np.array([1,2]), monotonic=0, gradient=0, unit='year')
        
#     def test_IMT3Complicated(self):
#         k = fk.SqExpKernelFamily().default() * fk.CentredPeriodicKernelFamily().default() * fk.CosineKernelFamily().default() * fk.CosineKernelFamily().default() * fk.LinKernelFamily().default() * fk.IMT3LinKernelFamily().default()
#         op = [fk.ZeroKernel(), k]
#         k = fk.ChangePointTanhKernel(location = 1.5, steepness=2, operands=op)
#         op = [k, fk.ZeroKernel()]
#         k = fk.ChangePointTanhKernel(location = 1.8, steepness=2, operands=op)
#         sentences = translation.translate_additive_component(k, np.array([1,2]), monotonic=0, gradient=0, unit='year')

#     def test_error_1(self):
#         k = fk.MaskKernelFamily(1,0,fk.SqExpKernelFamily())
#         try:
#             sentences = translation.translate_additive_component(k, np.array([-1,0,1]), monotonic=0, gradient=0, unit='year')
#         except:
#             pass
#         else:
#             raise RuntimeError('I should not be able to describe a mask kernel on its own')


if __name__ == "__main__":
    unittest.main()
