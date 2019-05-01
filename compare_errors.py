import torch
import torch.optim as optim
import torch.nn as nn
from time import time
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import mixture
from scipy.stats import entropy
import itertools
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans
from math import ceil
from sklearn.preprocessing import normalize
import cv2
import warnings

from load_data import Data
from mobilenet_v2 import MobileNetV2
from models import LeNetMNIST, LeNetCIFAR, LeNetVOC, LeNetCIFAR_train, LeNetMNIST_train, LeNet5, MLP


warnings.filterwarnings("ignore", category=DeprecationWarning)
datset_name = "MNIST"

#sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

def extract_sift(dset, train_data, test_data):

	dim = 64
	#sift = cv2.xfeatures2d.SIFT_create()
	sift = cv2.xfeatures2d.SURF_create()
	#sift = cv2.ORB_create()


	start = time()
	features = []
	train_features = []
	for i in xrange(len(train_data)):
		img = dset.inv_normalize(torch.from_numpy(train_data[i].copy())).numpy()
		img = cv2.cvtColor(np.transpose(img, (1, 2, 0)), cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img,(320,320))
		img = np.uint8(255*img)
		kps, des = sift.detectAndCompute(img, None)
		if des is None: feature = np.zeros((1, dim))
		else: feature = np.asarray(des)
		train_features.append(feature)
		features.extend(feature)
	features = np.asarray(features)
	print "Features extracted", time()-start
	start = time()
	# Bag of Visual Words:
	K = 100
	kmeans = MiniBatchKMeans(n_clusters=K).fit(features)  # KMeans(n_clusters=K).fit(features)
	print "KMeans finished", time()-start

	# vectorize bag of words:
	train_vectors = []
	for i in xrange(len(train_data)):
		des = train_features[i]
		vector = np.zeros((1, K))
		labels = kmeans.predict(des)
		for j in xrange(len(des)):
			vector[0, labels[j]] += 1.0
		train_vectors.extend(normalize(vector))

	print "Test vectors calculation"
	test_vectors = []
	for i in xrange(len(test_data)):
		img = dset.inv_normalize(torch.from_numpy(test_data[i].copy())).numpy()
		img = cv2.cvtColor(np.transpose(img, (1, 2, 0)), cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (320, 320))
		img = np.uint8(255 * img)
		kps, des = sift.detectAndCompute(img, None)
		if des is None: des = np.zeros((1, dim))
		vector = np.zeros((1, K))
		labels = kmeans.predict(des)
		for j in xrange(len(des)):
			vector[0, labels[j]] += 1.0
		test_vectors.extend(normalize(vector))

	return np.asarray(train_vectors), np.asarray(test_vectors)

def flatten_data(data):
	return np.reshape(data, (len(data), -1))

def get_cluster_centers(data, labels):
	cluster_centers = []
	categories = np.unique(labels)
	for i in categories:
		indices = np.where(labels == i)[0]
		cluster_points = np.asarray([data[idx] for idx in indices])
		cluster_centers.append(np.mean(cluster_points, axis=0))
	return np.asarray(cluster_centers)

def predict(data, centers):
	predictions = []
	for i in xrange(len(data)):
		point = data[i, :]
		distances = [np.linalg.norm((point - centers[j, ...])) for j in xrange(len(centers))]
		predictions.append(np.argmin(distances))
	return np.asarray(predictions)

def predict_svm(train_data, train_labels, test_data):
	clf = SVC(kernel='rbf')
	clf.fit(train_data, train_labels)
	return clf.predict(test_data)

def predict_kNN(train_data, train_labels, test_data, k = 5):
	clf = KNeighborsClassifier(n_neighbors=k)
	clf.fit(train_data, train_labels)
	return clf.predict(test_data)


def get_best_k(train_data, train_labels, test_data, test_labels):
	for k in xrange(1,20,2):
		predictions = predict_kNN(train_data, train_labels, test_data, k)
		accuracy = accuracy_score(test_labels, predictions)
		print "Accuracy for :", k, "is:", accuracy

def shuffle_data(data, labels):
	assert len(data)==len(labels)
	indices = np.arange(len(data))
	shuffle(indices)
	new_data = []
	new_labels = []
	for idx in indices:
		new_data.append(data[idx])
		new_labels.append(labels[idx])
	return np.asarray(new_data), np.asarray(new_labels)

def compare_models(dset):
	# get data:
	train_data, train_labels = dset.get_data(dset.train_loader, 500)
	test_data, test_labels = dset.get_data(dset.test_loader, 200)


	print "SIFT features:"
	sift_train_data, sift_test_data = extract_sift(dset, train_data, test_data)
	print np.shape(sift_train_data), np.shape(sift_test_data)
	print
	print "Raw data:"
	baselines(sift_train_data, train_labels, sift_test_data, test_labels)
	print
	print "Baseline MLP:"
	baseline_MLP(sift_train_data, train_labels, sift_test_data, test_labels)
	'''
	print
	print "Raw data:"
	baselines(flatten_data(train_data), train_labels, flatten_data(test_data), test_labels)
	print
	print "Baseline MLP:"
	baseline_MLP(flatten_data(train_data), train_labels, flatten_data(test_data), test_labels)
	print
	print "Baseline CNN:"
	baseline_CNN(train_data, train_labels, test_data, test_labels, dataset_name=dset.dataset_name)

	print
	print "Ours:"
	#run_random_networks(dset, train_data, train_labels, test_data, test_labels)
	test_affinity_approach(dset, train_data, train_labels, test_data, test_labels)
	'''

def baselines(train_data, train_labels, test_data, test_labels):

	cluster_centers = get_cluster_centers(train_data, train_labels)
	predictions = predict(test_data, cluster_centers)
	accuracy = accuracy_score(test_labels, predictions)
	print "Accuracy for closest to center baseline method:", accuracy

	#predictions = predict_svm(train_data, train_labels, test_data)
	#accuracy = accuracy_score(test_labels, predictions)
	#print "Accuracy for svm baseline method:", accuracy

	predictions = predict_kNN(train_data, train_labels, test_data, k=7)
	accuracy = accuracy_score(test_labels, predictions)
	print "Accuracy for kNN baseline method:", accuracy

def baseline_MLP(train_data, train_labels, test_data, test_labels, epochs=100):
	n = len(train_data[0])
	model = MLP(n)
	optimizer = optim.Adam(model.parameters(), lr=2e-3)
	if is_cuda:
		model.cuda()
	criterion = nn.CrossEntropyLoss()
	model.train()
	train_data, train_labels = shuffle_data(train_data, train_labels)
	batch_size = 128
	no_of_batches = int(ceil(float(len(train_data)) / batch_size))
	for epoch in xrange(epochs):
		for i in xrange(no_of_batches):
			start_idx = i*batch_size
			end_idx = min([(i+1)*batch_size,len(train_data)-1])
			data = torch.from_numpy(train_data[start_idx:end_idx])
			target = torch.from_numpy(train_labels[start_idx:end_idx])
			if is_cuda:
				data, target = data.float().cuda(), target.cuda()
			optimizer.zero_grad()
			output = model(data)
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()
	predictions = []
	for i in xrange(len(test_data)):
		data = torch.from_numpy(np.expand_dims(test_data[i], axis=0))
		if is_cuda:
			data = data.float().cuda()
		output = model(data)
		_, predicted = torch.max(output.data, 1)
		predictions.append(int(predicted.data.cpu().numpy()[0]))
	predictions = np.asarray(predictions)
	accuracy = accuracy_score(test_labels, predictions)
	print "Accuracy for MLP baseline method:", accuracy

def baseline_CNN(train_data, train_labels, test_data, test_labels, dataset_name="MNIST", epochs=100):

	if dataset_name == 'MNIST':
		curr_lr = 2e-3  # 0.1
		model = LeNet5()  # LeNetMNIST_train()
		optimizer = optim.Adam(model.parameters(), lr=curr_lr)
	elif dataset_name == 'CIFAR':
		curr_lr = 2e-3  # 0.1
		model = LeNetCIFAR_train()  # MobileNetV2()
		optimizer = optim.Adam(model.parameters(), lr=curr_lr)
		epochs = 60
	if is_cuda:
		model.cuda()
	criterion = nn.CrossEntropyLoss()
	model.train()

	train_data, train_labels = shuffle_data(train_data, train_labels)

	batch_size = 128
	no_of_batches = int(ceil(float(len(train_data)) / batch_size))
	for epoch in xrange(epochs):
		#if epoch == 15 or epoch == 30:
			#curr_lr = curr_lr / 10
			#for g in optimizer.param_groups:
			#	g['lr'] = curr_lr
		for i in xrange(no_of_batches):
			start_idx = i*batch_size
			end_idx = min([(i+1)*batch_size,len(train_data)-1])
			data = torch.from_numpy(train_data[start_idx:end_idx])
			target = torch.from_numpy(train_labels[start_idx:end_idx])
			if is_cuda:
				data, target = data.cuda(), target.cuda()
			optimizer.zero_grad()
			output = model(data)
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()
	predictions = []
	for i in xrange(len(test_data)):
		data = torch.from_numpy(np.expand_dims(test_data[i], axis=0))
		if is_cuda:
			data = data.cuda()
		output = model(data)
		_, predicted = torch.max(output.data, 1)
		predictions.append(int(predicted.data.cpu().numpy()[0]))
	predictions = np.asarray(predictions)
	accuracy = accuracy_score(test_labels, predictions)
	print "Accuracy for CNN baseline method:", accuracy

def test_affinity_approach(dset, train_data, train_labels, test_data, test_labels):
	new_train_data, new_test_data = run_random_networks(dset, train_data, train_labels, test_data, test_labels)

	print "Base case:"
	baselines(new_train_data, train_labels, new_test_data, test_labels)
	print "Baseline MLP:"
	baseline_MLP(new_train_data, train_labels, new_test_data, test_labels, epochs=20)

	print "PCA:"
	pca_train_data, pca_test_data = perform_pca(new_train_data, new_test_data)
	baselines(pca_train_data, train_labels, pca_test_data, test_labels)
	print "Baseline MLP:"
	baseline_MLP(pca_train_data, train_labels, pca_test_data, test_labels, epochs=20)

	print "LDA:"
	lda_train_data, lda_test_data = perform_lda(new_train_data, train_labels, new_test_data)
	baselines(lda_train_data, train_labels, lda_test_data, test_labels)
	print "Baseline MLP:"
	baseline_MLP(lda_train_data, train_labels, lda_test_data, test_labels, epochs=20)


def perform_pca(train_data, test_data):
	clf = PCA(n_components=50)
	clf.fit(train_data)
	return clf.transform(train_data), clf.transform(test_data)

def perform_lda(train_data, train_labels, test_data):
	clf = LinearDiscriminantAnalysis()
	clf.fit(train_data, train_labels)
	return clf.transform(train_data), clf.transform(test_data)

def run_random_networks(dset, train_data, train_labels, test_data, test_labels):
	iters = 100
	joined_data = np.concatenate((train_data, test_data), axis=0)
	n = len(joined_data)
	m = len(train_data)
	affinity_matrix = np.zeros((n, n))
	prev_accuracy = 0.0
	i = 0
	count = 0
	while i<iters:
		count += 1
		model = dset.create_network()
		#affinity_matrix = pass_random_network(model, joined_data)
		temp_affinity_matrix = (affinity_matrix + pass_random_network(model, joined_data))#/(min([i+1,2]))
		curr_accuracy = test_affinity_matrix(temp_affinity_matrix[:m,...], train_labels, temp_affinity_matrix[m:,...], test_labels)
		if curr_accuracy > prev_accuracy:
			print "Improved:", count, "Current accuracy is", curr_accuracy
			prev_accuracy = curr_accuracy
			affinity_matrix = temp_affinity_matrix[...]
			i += 1
		else:
			"No improvement"
		del temp_affinity_matrix
		if(count>=200): return affinity_matrix[:m,...], affinity_matrix[m:,...]
	return affinity_matrix[:m,...], affinity_matrix[m:,...]

def test_affinity_matrix(train_data, train_labels, test_data, test_labels):
	cluster_centers = get_cluster_centers(train_data, train_labels)
	predictions = predict(test_data, cluster_centers)
	accuracy = accuracy_score(test_labels, predictions)
	return accuracy



'''
dset_class = Data(datset_name)
dset_class.load_dataset()
is_cuda = dset_class.is_cuda

compare_models(dset_class)
'''