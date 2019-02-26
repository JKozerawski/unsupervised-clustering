from util import show_affinity_matrix, get_density, get_points_according_to_density, get_training_data, matrix_similarity
from load_data import modify_data, Data

import numpy as np
import pickle


n_samples = 250
iters = 200
n_clusters = 20

dset = Data("CIFAR")
dset.load_dataset()

#dset.train_net()

dset.choose_data_small(n_samples = n_samples)

#dset.inputs_small, dset.ground_truth, dset.classes, dset.colors = modify_data(dset.inputs_small, 0, 1)

dset.run_dataset(n_clusters=50, iters=iters)
'''
old_affinity_matrix = np.mean(dset.all_affinity_matrices,axis=0)

pickle.dump( old_affinity_matrix, open( "./affinity.p", "wb" ) )
data = dset.visualize_data(old_affinity_matrix)
'''






'''
#new_affinity_matrices = matrix_similarity(dset.all_affinity_matrices, k = 2)

data = dset.visualize_data(old_affinity_matrix)
for affinity_matrix in new_affinity_matrices:
	data = visualize_data(affinity_matrix, dset.ground_truth, dset.classes, dset.colors)
'''





'''
new_data, new_labels = get_training_data(data, inputs)

helper_model = train_net(new_data, new_labels)
affinity_matrix = test_net(helper_model, inputs, ground_truth, classes)
data = visualize_data(affinity_matrix, ground_truth, classes)
'''





'''
f , kernel = get_density(data)
inputs, ground_truth = get_points_according_to_density(inputs, ground_truth, data, kernel)
affinity_matrix = run_dataset(inputs, n_clusters=5, iters=iters, dataset = dataset)
data = visualize_data(affinity_matrix, ground_truth, classes)

#affinity_matrix = run_dataset(inputs, n_samples = n_samples, n_clusters=5, iters=iters, no_of_categories = no_of_categories, dataset = dataset)
#data = visualize_data(affinity_matrix, ground_truth, classes)

#show_affinity_matrix(affinity_matrix)
'''
