import torch
import torch.optim as optim
import torch.nn as nn
from time import time
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt

from load_data import Data
from models import LeNetMNIST, LeNetCIFAR, LeNetVOC, LeNetCIFAR_train, LeNetMNIST_train

datset_name = "MNIST"


def train_net(dataset_name, trainloader, testloader, n_epochs=20):
	print "Training"
	if dataset_name == 'MNIST':
		model = LeNetMNIST_train()
		optimizer = optim.SGD(model.fc2.parameters(), lr=0.01)
	elif dataset_name == 'CIFAR':
		model = LeNetCIFAR_train()
		optimizer = optim.SGD(model.fc3.parameters(), lr=0.1)

	if is_cuda:
		model.cuda()	
	criterion = nn.CrossEntropyLoss()
	model.train()
	for epoch in xrange(n_epochs):
		train_loss = 0.0
		for data, target in trainloader:
			if is_cuda:
				data, target = data.cuda(), target.cuda()
			optimizer.zero_grad()
			output = model(data)
			loss = criterion(output, target)
			loss.backward()
			# perform a single optimization step (parameter update)
			optimizer.step()
			# update training loss
			train_loss += loss.item()*data.size(0)
		train_loss = train_loss/len(trainloader.dataset)
		# print training/validation statistics 
		print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
		test_net(model, testloader)
	return model


def test_net(model, dataloader):
	if is_cuda:
		model.cuda()
	correct = 0
	for data, target in dataloader:
		if is_cuda:
			data, target = data.cuda(), target.cuda()
		output = model(data)
		_, predicted = torch.max(output.data, 1)
		correct += (predicted == target).sum()
	print "Test accuracy:", float(correct)/len(dataloader.dataset)


def get_affinity_scores(dset):
	iters = 20
	n_clusters = 50
	dset.create_networks(iters)
	train_data, train_labels = dset.get_data(dset.train_loader, 500)
	test_data, test_labels = dset.get_data(dset.test_loader, 500)
	train_affinity_matrices = dset.run_dataset(train_data, n_clusters=n_clusters, iters=iters)
	test_affinity_matrices = dset.run_dataset(test_data, n_clusters=n_clusters, iters=iters)
	for k in xrange(len(train_affinity_matrices)):
		if k == 0:
			affinity_matrix = train_affinity_matrices[0, ...]
		else:
			affinity_matrix = np.mean(train_affinity_matrices[:k, ...], axis=0)
		data = dset.perform_pca(affinity_matrix, n_components=25)
		# data = dset.normalize_data(data)
		# data = affinity_matrix[...]
		categories = np.unique(train_labels)
		centers = []
		# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'maroon', 'coral', 'olive']
		for i in categories:
			indices = np.where(train_labels == i)[0]
			chosen_data = np.asarray([data[j, :] for j in indices])
			centers.append(np.mean(chosen_data, axis=0))
			# plt.scatter([data[j,0] for j in indices],[data[j,1] for j in indices], s=10, label=str(i), c=colors[i])
		scores = []
		predictions = []
		correct = 0
		centers = np.asarray(centers)
		test_data = dset.pca.transform(np.mean(test_affinity_matrices, axis=0))
		for i in xrange(len(test_data)):
			point = test_data[i, :]
			label = test_labels[i]
			distances = [np.linalg.norm((point - centers[j, ...])) for j in xrange(len(centers))]
			scores.append(distances[label])
			predictions.append(np.argmin(distances))
			if np.argmin(distances) == label:
				correct += 1
		# plt.scatter([centers[j,0] for j in xrange(10)],[centers[j,1] for j in xrange(10)], s=20, label='centers', c='r')
		# plt.legend(bbox_to_anchor=(-0.3, 1), loc='upper left', ncol=1, fontsize = 10)
		
		print "No of networks:", k+1, "Accuracy", float(correct)/len(test_data)
		# plt.show()


dset_class = Data(datset_name)
dset_class.load_dataset()
is_cuda = dset_class.is_cuda

get_affinity_scores(dset_class)

# model = train_net(datset_name, dset.train_loader, dset.test_loader)
# test_net(model, dset.test_loader)
