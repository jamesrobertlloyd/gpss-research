import unittest

import numpy as np

import experiment
import postprocessing

class experiment_testcase(unittest.TestCase):

    def test_experiment_and_writeup_cp(self):
        experiment.run_experiment_file('../experiments/debug/debug_changepoint.py')
        postprocessing.make_all_1d_figures(['../results/debug-changepoint/'], '../analyses/debug-changepoint/figures/', rescale=False, data_folder='../data/debug/', skip_kernel_evaluation=False)

    def test_experiment_and_writeup_cw(self):
        experiment.run_experiment_file('../experiments/debug/debug_changewindow.py')
        postprocessing.make_all_1d_figures(['../results/debug-changewindow/'], '../analyses/debug-changewindow/figures/', rescale=False, data_folder='../data/debug/', skip_kernel_evaluation=False)

    def test_experiment_and_writeup(self):
        experiment.run_debug_kfold()
        postprocessing.make_all_1d_figures(['../results/debug/'], '../analyses/debug/figures/', rescale=False, data_folder='../data/debug/', skip_kernel_evaluation=False)

if __name__ == "__main__":
    unittest.main()
