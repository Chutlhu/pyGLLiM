import os
import numpy as np
import scipy.io
import pickle

from gllim import gllim_train_wrapper
from gllim import gllim_inverse_map

def save_obj(obj, name ):
    with open('./data/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('./data/' + name, 'rb') as f:
        return pickle.load(f)

path_to_matlab_dataset = '../the_GLLiM_toolbox_matlab/data/matlab_workspace.mat'
dataset = scipy.io.loadmat(path_to_matlab_dataset)

parameters = dataset['poses']
observations = dataset['images']

n_test = 8 # dimension of test set
n_components = 25

n_feature, n_obs = observations.shape
n_train = n_obs - n_test

# Training set
#random_indeces = np.random.permutation(n_obs)
random_indeces = [k for k in range(n_obs)]
training_params = parameters[:,random_indeces[0:n_train]]
training_obs = observations[:,random_indeces[0:n_train]]

# Testing set
testing_params = parameters[:,random_indeces[n_train:n_obs]]
testing_obs = observations[:,random_indeces[n_train:n_obs]]

# Training a GLLiM model using...
#... no latent variables

if not os.path.exists('./data/estimated_params.pkl'):
    estimated_params = {}
    estimated_params['Lw0'] = \
        gllim_train_wrapper(training_params, training_obs, n_components,
                     dim_hidden_components=0, n_iter=20, verbose=1)
    estimated_params['Lw1'] = \
        gllim_train_wrapper(training_params, training_obs, n_components,
        dim_hidden_components=1, n_iter=20, verbose=1)
    save_obj(estimated_params, 'estimated_params')
else:
    estimated_params = load_obj('estimated_params.pkl')

# Post Processing: Head pose estimation
estimated_poses = {}
regression_error = {}
for Lw in ['Lw1', 'Lw0']:
    print(Lw)
    estimated_poses[Lw] = \
        gllim_inverse_map(testing_obs, estimated_params[Lw][0])
    regression_error[Lw] = np.sqrt(np.sum((estimated_poses[Lw][0]-testing_params)**2, axis = 0))
    print(regression_error[Lw])

# Results
