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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from load_data import Data
from models import LeNetMNIST, LeNetCIFAR, LeNetVOC, LeNetCIFAR_train, LeNetMNIST_train

datset_name = "MNIST"

#sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})


def train_net(dataset_name, trainloader, testloader, n_epochs=10):
	print "Training"
	curr_lr = 0.1
	if dataset_name == 'MNIST':
		model = LeNetMNIST_train()
		optimizer = optim.SGD(model.fc2.parameters(), lr=curr_lr)
	elif dataset_name == 'CIFAR':
		model = LeNetCIFAR_train()
		optimizer = optim.SGD(model.fc3.parameters(), lr=curr_lr)

	if is_cuda:
		model.cuda()	
	criterion = nn.CrossEntropyLoss()
	model.train()
	for epoch in xrange(n_epochs):
		if epoch == 8:
			curr_lr = curr_lr/10
			for g in optimizer.param_groups:
				g['lr'] = curr_lr
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

def compare_accuracy(model, affinity_predictions, test_data, test_labels):
	model_predictions = []
	print np.shape(test_data)
	for i in xrange(len(test_data)):
		data = torch.from_numpy(np.expand_dims(test_data[i], axis=0))
		if is_cuda:
			data = data.cuda()
		output = model(data)
		_, predicted = torch.max(output.data, 1)
		model_predictions.append(int(predicted.data.cpu().numpy()[0]))
	assert len(model_predictions) == len(affinity_predictions)
	model_mistakes = np.where(model_predictions != test_labels)[0]
	print "Model mistakes and predictions:", len(model_mistakes), len(model_predictions)
	affinity_mistakes = np.where(affinity_predictions != test_labels)[0]
	found = 0
	for i in model_mistakes:
		if i in affinity_mistakes:
			found += 1.
	print "Found accuracy:", found/len(model_mistakes), "or:", found/len(affinity_mistakes)
	return model_predictions

def point_entropy(point, centers):
	dists = np.asarray([np.linalg.norm(point-center) for center in centers])
	probs = dists/np.max(dists)	# normalize to 0-1
	probs = np.abs(1.0-dists)	# reverse
	return entropy(probs)


def plot_mistakes(model_predictions, test_labels, test_data_pca):
	pt_size = 10
	fig, axarr = plt.subplots(2, 5)
	#g = mixture.GaussianMixture(n_components=10, random_state=40)
	# get the entropy for every point
	#g.fit(test_data_pca)
	#probs = g.predict_proba(test_data_pca)
	#entropy_data = [entropy(prob) for prob in probs]

	cluster_centers = []
	for i in xrange(10):
		indices = np.where(test_labels == i)[0]
		cluster_points = np.asarray([test_data_pca[idx] for idx in indices])
		cluster_centers.append(np.mean(cluster_points, axis=0))

	for i in xrange(10):
		indices = np.where(test_labels == i)[0]
		predicted = np.asarray([model_predictions[idx] for idx in indices])

		#cluster_entropies =  np.asarray([entropy_data[idx] for idx in indices])
		cluster_points = np.asarray([test_data_pca[idx] for idx in indices])
		#distances = np.asarray([np.linalg.norm(point-cluster_centers[i]) for point in cluster_points])
		distances = np.asarray([point_entropy(point, cluster_centers) for point in cluster_points])

		correctly_predicted = np.where(predicted == i)[0]
		wrongly_predicted = np.where(predicted != i)[0]

		correct_points = np.asarray([distances[j] for j in correctly_predicted])
		wrong_points = np.asarray([distances[j] for j in wrongly_predicted])

		a = i % 5
		b = 0
		if(i>=5): b = 1
		axarr[b, a].scatter(correct_points, np.zeros(len(correct_points)), s=pt_size, c ='b')
		sns.kdeplot(correct_points, shade=True,  color='b', ax=axarr[b, a])
		axarr[b, a].scatter(wrong_points, np.zeros(len(wrong_points)), s=pt_size, c='r')
		sns.kdeplot(wrong_points, shade=True, color='r', ax=axarr[b, a])
	plt.show()



def get_affinity_scores(dset):
	iters = 300
	n_clusters = 50
	dset.create_networks(iters)
	train_data, train_labels = dset.get_data(dset.train_loader, 1000)
	test_data, test_labels = dset.get_data(dset.test_loader, 500)
	joined_data = np.concatenate((train_data, test_data), axis=0)  # test_data[...]  # np.concatenate((train_data, test_data), axis=0)
	affinity_matrix = dset.run_dataset(joined_data, n_clusters=n_clusters, iters=iters)

	start = time()
	print "Dimensionality reduction..."
	clf = LinearDiscriminantAnalysis()
	clf.fit(affinity_matrix[:len(train_data), ...], train_labels)
	data = clf.transform(affinity_matrix[:len(train_data), ...])
	#data = dset.perform_pca(affinity_matrix, n_components=25)

	print "Calculating accuracy..."
	#train_labels = test_labels
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
	test_data_pca = clf.transform(affinity_matrix[len(train_data):, ...])
	for i in xrange(len(test_data_pca)):
		point = test_data_pca[i, :]
		label = test_labels[i]
		distances = [np.linalg.norm((point - centers[j, ...])) for j in xrange(len(centers))]
		scores.append(distances[label])
		predictions.append(np.argmin(distances))
		if np.argmin(distances) == label:
			correct += 1
	# plt.scatter([centers[j,0] for j in xrange(10)],[centers[j,1] for j in xrange(10)], s=20, label='centers', c='r')
	# plt.legend(bbox_to_anchor=(-0.3, 1), loc='upper left', ncol=1, fontsize = 10)

	print "Accuracy", float(correct)/len(test_data)
	print "Time elapsed", time()-start
	# plt.show()
	return test_data, test_labels, predictions, test_data_pca


dset_class = Data(datset_name)
dset_class.load_dataset()
is_cuda = dset_class.is_cuda

test_data, test_labels, predictions, test_data_pca = get_affinity_scores(dset_class)
'''
model = train_net(datset_name, dset_class.train_loader, dset_class.test_loader)
model_predictions = compare_accuracy(model, predictions, test_data, test_labels)
plot_mistakes(model_predictions, test_labels, test_data_pca)
'''