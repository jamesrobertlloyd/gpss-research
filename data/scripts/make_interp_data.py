from __future__ import division

raw_data_folder = '../tsdlr/'
extrap_data_folder = '../tsdlr_5050/'

import scipy.io
import os
import numpy as np

if not os.path.isdir(extrap_data_folder):
	os.mkdir(extrap_data_folder)

for data_file_name in os.listdir(raw_data_folder):
    if data_file_name.endswith('.mat'):
        data = scipy.io.loadmat(os.path.join(raw_data_folder, data_file_name))
        X = data['X']
        y = data['y']
        indices = np.random.permutation(X.shape[0])
        cut_off = int(np.floor(X.shape[0]/2))
        train_idx, test_idx = indices[:cut_off], indices[cut_off:]
        X_test = X[test_idx]
        y_test = y[test_idx]
        X = X[train_idx]
        y = y[train_idx]
        data = {'X' : X, 'y' : y, 'Xtest' : X_test, 'ytest' : y_test}
        scipy.io.savemat(os.path.join(extrap_data_folder, data_file_name), data)