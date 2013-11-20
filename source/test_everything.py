import unittest

import numpy as np

#import experiment
import model
#import grammar
#import translation

class model_testcase(unittest.TestCase):

    def test_noise_kernel(self):
        k = model.NoiseKernel()
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
        k = model.SqExpKernel()
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
        k = model.SqExpKernel()
        k1 = k.copy()
        k2 = k.copy()
        k = k1 + k2
        print '\n', k.pretty_print(), '\n'
        print '\n', k.syntax, '\n'
        print '\n', k, '\n'
        k1.dimension = 0
        k2.dimension = 1
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
        k1.dimension = 0
        k2.dimension = 1
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

    def test_model(self):
        m = model.MeanZero()
        k = model.SqExpKernel()
        l = model.LikGauss()
        regression_model = model.RegressionModel(mean=m, kernel=k, likelihood=l)
        print '\n', regression_model.pretty_print(), '\n'
        print '\n', regression_model.__repr__(), '\n'

    def test_base(self):
        kernels = model.base_kernels_without_dimension('SE,Const,Noise')
        for k in kernels:
            print '\n', k.pretty_print(), '\n'
        kernels = model.base_kernels(3, 'SE,Const,Noise')
        for k in kernels:
            print '\n', k.pretty_print(), '\n'

    def test_repr(self):
        m = model.MeanZero()
        k = model.SqExpKernel()
        l = model.LikGauss()
        regression_model = model.RegressionModel(mean=m, kernel=k, likelihood=l)
        print regression_model
        print  model.repr_to_model(regression_model.__repr__())
        assert regression_model == model.repr_to_model(regression_model.__repr__())

    def test_collapse_add_idempotent(self):
        k = model.SqExpKernel()
        k1 = k.copy()
        k2 = k.copy()
        k = model.NoiseKernel(sf=-1)
        k3 = k.copy()
        k4 = k.copy()
        k = model.ConstKernel(sf=1)
        k5 = k.copy()
        k6 = k.copy()
        k = k1 + k2 + k3 + k4 + k5 + k6
        print '\n', k.pretty_print(), '\n'
        k = model.collapse_additive_idempotency(k)
        print '\n', k.pretty_print(), '\n'

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
