import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
import numpy as np
from time import time

from sklearn.neighbors.kde import KernelDensity
from collections import Counter
from scipy.spatial.distance import hamming
from sklearn.cluster import KMeans
	

#-----------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------#
'''
def perform_pca(affinity_matrix, n_components = 5):
	pca = PCA(n_components=n_components)
	return pca.fit_transform(affinity_matrix)

def visualize_data(affinity_matrix, ground_truth, classes, colors):

	def on_pick(event):
		print event.ind
		print data[event.ind], "clicked"

	ground_truth = ground_truth.astype(int)
	categories = np.unique(ground_truth)
	#print categories
	n_categories = len(categories)
	data = perform_pca(affinity_matrix)
	data = normalize_data(data)

	pt_size = 10
	fig, axarr = plt.subplots(2, 2)

	for i in categories:
		i = int(i)
		indices = np.where(ground_truth == i)[0]
		axarr[0,0].scatter([data[j,0] for j in indices],[data[j,1] for j in indices], picker = 5, s=pt_size, label=classes[i], c=colors[i])
		axarr[0,1].scatter([data[j,1] for j in indices],[data[j,2] for j in indices], picker = 5, s=pt_size, label=classes[i], c=colors[i])
		axarr[1,0].scatter([data[j,2] for j in indices],[data[j,3] for j in indices], picker = 5, s=pt_size, label=classes[i], c=colors[i])
		axarr[1,1].scatter([data[j,3] for j in indices],[data[j,4] for j in indices], picker = 5, s=pt_size, label=classes[i], c=colors[i])
	
	axarr[0, 0].set_title('PCA dims: 1 & 2')
	axarr[0, 1].set_title('PCA dims: 2 & 3')
	axarr[1, 0].set_title('PCA dims: 3 & 4')
	axarr[1, 1].set_title('PCA dims: 4 & 5')
	# show legend:
	axarr[0,0].legend(bbox_to_anchor=(-0.3, 1), loc='upper left', ncol=1, fontsize = 10)
	fig.canvas.mpl_connect('pick_event', on_pick)
	plt.show()
	return data
	#plt.clf()
	#get_density(data)

def normalize_data(data):
	data_center = np.mean(data)
	data = data - data_center
	x_min = np.min(data[:,0])
	y_min = np.min(data[:,1])
	z_min = np.min(data[:,2])
	u_min = np.min(data[:,3])
	w_min = np.min(data[:,4])
	x_max = np.max(data[:,0])
	y_max = np.max(data[:,1])
	z_max = np.max(data[:,2])
	u_max = np.max(data[:,3])
	w_max = np.max(data[:,4])
	for i in xrange(5):
		data[:,i] = data[:,i]/max([abs(np.min(data[:,i])), abs(np.max(data[:,i]))])
	#max_val = max([abs(x_min),abs(x_max),abs(y_min),abs(y_max),abs(z_min),abs(z_max),abs(u_min),abs(u_max),abs(w_min),abs(w_max)])
	#data = data/max_val
	return data
'''
def get_density(data):
	xmin, xmax, ymin, ymax = -1.0, 1.0, -1.0, 1.0
	
	xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
	positions = np.vstack([xx.ravel(), yy.ravel()])
	values = np.vstack([data[:,0], data[:,1]])
	kernel = st.gaussian_kde(values)
	f = np.reshape(kernel(positions).T, xx.shape)

	# scale it according to the distance from the mean
	f = f*np.abs(xx)
	f = f*np.abs(yy)
	
	fig = plt.figure()
	ax = fig.gca()
	ax.set_xlim(xmin, xmax)
	ax.set_ylim(ymin, ymax)
	# Contourf plot
	cfset = ax.contourf(xx, yy, f, cmap='Blues')
	## Or kernel density estimate plot instead of the contourf plot
	#ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ym	ax])
	# Contour plot
	cset = ax.contour(xx, yy, f, colors='k')
	# Label plot
	ax.clabel(cset, inline=1, fontsize=10)
	ax.set_xlabel('Y1')
	ax.set_ylabel('Y0')

	plt.show()
	return f , kernel

def get_points_according_to_density(inputs, ground_truth, data, kernel):
	new_data = []
	new_ground_truth = []
	threshold = 0.2*max(kernel(data.T))
	print threshold
	for i, point in enumerate(data):
		if( kernel(point)*np.linalg.norm(point) >= threshold ):
			new_data.append(inputs[i])
			new_ground_truth.append(ground_truth[i])
	return np.asarray(new_data), np.asarray(new_ground_truth)

def get_training_data(data, inputs):
	main_dim = 0
	xmax = np.max(data[:,main_dim])
	xmin = np.min(data[:,main_dim])
	x_range = xmax-xmin
	thresh_1 = xmin+0.3*x_range
	thresh_2 = xmax-0.3*x_range
	indices_1 = np.where(data[:,main_dim]<=thresh_1)[0]
	indices_2 = np.where(data[:,main_dim]>=thresh_2)[0]

	categories = np.asarray([inputs[i] for i in indices_1] + [inputs[i] for i in indices_2])
	ground_truth = np.asarray([0 for i in indices_1] + [1 for i in indices_2])
	
	# shuffle the data:
	permute = np.random.permutation(len(categories))
	categories = np.asarray([categories[i,:] for i in permute])
	ground_truth = np.asarray([ground_truth[i] for i in permute])
	return categories, ground_truth

def cluster_to_class(predicted_clusters, ground_truth, classes, n_clusters = 10):
	assert len(predicted_clusters) == len(ground_truth)
	all_elems = []
	purity = []
	K = 15
	for i in xrange(n_clusters):
		indices = np.where(predicted_clusters == i)[0]
		clustered_instances = [ground_truth[idx] for idx in indices]
		clustered_instances = [classes[int(idx)] for idx in clustered_instances]
		print clustered_instances
		elems = Counter(clustered_instances)
		all_elems.append(elems)
		purity.append(max(elems.values())/float(len(indices)))
		print i, elems
	print np.mean(np.asarray(purity)), np.median(np.asarray(purity))

def show_affinity_matrix(affinity_matrix):
	plt.style.use('grayscale')
	plt.matshow(affinity_matrix)
	plt.show()



def matrix_similarity(affinity_matrices, k = 2):
	start = time()
	affinity_vectors = [affinity_matrix.flatten() for affinity_matrix in affinity_matrices]
	affinity_vectors = np.asarray(affinity_vectors)
	affinity_vectors = perform_pca(affinity_vectors, 3)

	plt.scatter(affinity_vectors[:,0],affinity_vectors[:,1])
	plt.show()

	kmeans = KMeans(n_clusters=k).fit(affinity_vectors)
	new_affinity_matrices = []
	for i in xrange(k):
		indices = np.where(kmeans.labels_==i)[0]
		new_affinity_matrices.append( np.mean(np.asarray([affinity_matrices[idx] for idx in indices]),axis = 0) )
	print "Grouping time:", time()-start
	return new_affinity_matrices

	






