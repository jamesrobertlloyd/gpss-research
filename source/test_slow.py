import unittest

import numpy as np

import experiment

class experiment_testcase(unittest.TestCase):

    def test_noise_kernel(self):
        experiment.run_debug_kfold()

if __name__ == "__main__":
    unittest.main()
