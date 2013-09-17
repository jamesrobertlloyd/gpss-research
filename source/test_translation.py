import unittest

import numpy as np

import flexiblekernel as fk
import translation

class translation_testcase(unittest.TestCase):
    def test_SE(self):
        k = fk.SqExpKernelFamily().default()
        sentences = translation.translate_additive_component(k, np.array([-1,0,1]))
        #print '.\n'.join(sentences) + '.'
        
    def test_BroadSE(self):
        k = fk.SqExpKernelFamily().default()
        sentences = translation.translate_additive_component(k, np.array([0,0.5]))
        #print '.\n'.join(sentences) + '.'
        
    def test_poly(self):
        k = fk.LinKernelFamily().default() * fk.LinKernelFamily().default() * fk.LinKernelFamily().default()
        sentences = translation.translate_additive_component(k, np.array([0,0.5]))
        #print '.\n'.join(sentences) + '.'
        
    def test_SEpolydecrease(self):
        k = fk.SqExpKernelFamily().default() * fk.LinKernelFamily().default()
        sentences = translation.translate_additive_component(k, np.array([0.2,0.5]))
        #print '.\n'.join(sentences) + '.'
        
    def test_complicated(self):
        k = fk.SqExpKernelFamily().default() * fk.CentredPeriodicKernelFamily().default() * fk.CosineKernelFamily().default() * fk.CosineKernelFamily().default() * fk.LinKernelFamily().default() * fk.LinKernelFamily().default()
        op = [fk.ZeroKernel(), k]
        k = fk.ChangePointTanhKernel(location = 1.5, steepness=2, operands=op)
        op = [k, fk.ZeroKernel()]
        k = fk.ChangePointTanhKernel(location = 1.8, steepness=2, operands=op)
        sentences = translation.translate_additive_component(k, np.array([1,2]))
        #print k.pretty_print()
        #print '.\n'.join(sentences) + '.'
        
    def test_IMT3(self):
        k = fk.IMT3LinKernelFamily().default()
        sentences = translation.translate_additive_component(k, np.array([1,2]))
        #print '.\n'.join(sentences) + '.'
        
    def test_IMT3Complicated(self):
        k = fk.SqExpKernelFamily().default() * fk.CentredPeriodicKernelFamily().default() * fk.CosineKernelFamily().default() * fk.CosineKernelFamily().default() * fk.LinKernelFamily().default() * fk.IMT3LinKernelFamily().default()
        op = [fk.ZeroKernel(), k]
        k = fk.ChangePointTanhKernel(location = 1.5, steepness=2, operands=op)
        op = [k, fk.ZeroKernel()]
        k = fk.ChangePointTanhKernel(location = 1.8, steepness=2, operands=op)
        sentences = translation.translate_additive_component(k, np.array([1,2]))
        #print '.\n'.join(sentences) + '.'

if __name__ == "__main__":
    unittest.main()
