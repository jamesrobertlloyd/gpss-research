import unittest

import flexible_function as ff


class ff_testcase(unittest.TestCase):

    def test_param_vector_load(self):
        k1 = ff.SqExpKernel(dimension=0, lengthscale=1.0, sf=2.0)
        k2 = ff.SqExpKernel(dimension=1, lengthscale=2.0, sf=3.0)
        k3 = ff.SqExpKernel(dimension=2, lengthscale=3.0, sf=4.0)
        k4 = ff.NoiseKernel(sf=5.0)

        k = k1 + k2 * k3 + k4
        l = k.copy()
        l.load_gpy_param_vector(k.gpy_object.param_array)

        assert k == l

if __name__ == "__main__":
    unittest.main()
