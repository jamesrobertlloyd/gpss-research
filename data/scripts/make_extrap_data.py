raw_data_folder = '../tsdlr/'
extrap_data_folder = '../tsdlr_9010/'

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
        cut_off = np.min(X) + 0.9*(np.max(X) - np.min(X))
        X_test = X[X > cut_off]
        y_test = y[X > cut_off]
        y = y[X <= cut_off]
        X = X[X <= cut_off]
        data = {'X' : X, 'y' : y, 'Xtest' : X_test, 'ytest' : y_test}
        scipy.io.savemat(os.path.join(extrap_data_folder, data_file_name), data)