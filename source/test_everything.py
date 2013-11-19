import unittest

import numpy as np

import flexiblekernel as fk
import translation


class translation_testcase(unittest.TestCase):

    ##########################################
    # TRANSLATION                            #
    ##########################################

    def test_SE(self):
        k = fk.SqExpKernelFamily().default()
        sentences = translation.translate_additive_component(k, np.array([-1,0,1]), monotonic=0, gradient=0, unit='year')

    def test_SE_metres(self):
        k = fk.SqExpKernelFamily().default()
        sentences = translation.translate_additive_component(k, np.array([-1,0,1]), monotonic=0, gradient=0, unit='metre')

    def test_SE_number(self):
        k = fk.SqExpKernelFamily().default()
        sentences = translation.translate_additive_component(k, np.array([-1,0,1]), monotonic=0, gradient=0, unit='number')

    def test_SE_no_unit(self):
        k = fk.SqExpKernelFamily().default()
        sentences = translation.translate_additive_component(k, np.array([-1,0,1]), monotonic=0, gradient=0, unit='')
        
    def test_BroadSE(self):
        k = fk.SqExpKernelFamily().default()
        sentences = translation.translate_additive_component(k, np.array([0,0.5]), monotonic=0, gradient=0, unit='year')
        
    def test_poly(self):
        k = fk.LinKernelFamily().default() * fk.LinKernelFamily().default() * fk.LinKernelFamily().default()
        sentences = translation.translate_additive_component(k, np.array([0,0.5]), monotonic=0, gradient=0, unit='year')
        
    def test_SEpolydecrease(self):
        k = fk.SqExpKernelFamily().default() * fk.LinKernelFamily().default()
        sentences = translation.translate_additive_component(k, np.array([0.2,0.5]), monotonic=0, gradient=0, unit='year')
        
    def test_complicated(self):
        k = fk.SqExpKernelFamily().default() * fk.CentredPeriodicKernelFamily().default() * fk.CosineKernelFamily().default() * fk.CosineKernelFamily().default() * fk.LinKernelFamily().default() * fk.LinKernelFamily().default()
        op = [fk.ZeroKernel(), k]
        k = fk.ChangePointTanhKernel(location = 1.5, steepness=2, operands=op)
        op = [k, fk.ZeroKernel()]
        k = fk.ChangePointTanhKernel(location = 1.8, steepness=2, operands=op)
        sentences = translation.translate_additive_component(k, np.array([1,2]), monotonic=0, gradient=0, unit='year')
        
    def test_IMT3(self):
        k = fk.IMT3LinKernelFamily().default()
        sentences = translation.translate_additive_component(k, np.array([1,2]), monotonic=0, gradient=0, unit='year')
        
    def test_Const(self):
        k = fk.ConstKernelFamily().default()
        sentences = translation.translate_additive_component(k, np.array([1,2]), monotonic=0, gradient=0, unit='year')
        
    def test_ConstSE(self):
        k = fk.ConstKernelFamily().default() * fk.SqExpKernelFamily().default()
        sentences = translation.translate_additive_component(k, np.array([1,2]), monotonic=0, gradient=0, unit='year')
        
    def test_Window(self):
        k = fk.SqExpKernelFamily().default()
        op = [k, fk.ZeroKernel()]
        k = fk.ChangeBurstTanhKernel(location = 1.5, steepness=2, width=np.log(0.2), operands=op)
        sentences = translation.translate_additive_component(k, np.array([1,2]), monotonic=0, gradient=0, unit='year')
        
    def test_cos(self):
        k = fk.CosineKernelFamily().default()
        sentences = translation.translate_additive_component(k, np.array([1,2]), monotonic=0, gradient=0, unit='year')
        
    def test_Window2(self):
        k = fk.SqExpKernelFamily().default()
        op = [fk.ZeroKernel(), k]
        k = fk.ChangeBurstTanhKernel(location = 1.5, steepness=2, width=np.log(0.2), operands=op)
        sentences = translation.translate_additive_component(k, np.array([1,2]), monotonic=0, gradient=0, unit='year')
        
    def test_IMT3Complicated(self):
        k = fk.SqExpKernelFamily().default() * fk.CentredPeriodicKernelFamily().default() * fk.CosineKernelFamily().default() * fk.CosineKernelFamily().default() * fk.LinKernelFamily().default() * fk.IMT3LinKernelFamily().default()
        op = [fk.ZeroKernel(), k]
        k = fk.ChangePointTanhKernel(location = 1.5, steepness=2, operands=op)
        op = [k, fk.ZeroKernel()]
        k = fk.ChangePointTanhKernel(location = 1.8, steepness=2, operands=op)
        sentences = translation.translate_additive_component(k, np.array([1,2]), monotonic=0, gradient=0, unit='year')

if __name__ == "__main__":
    unittest.main()
