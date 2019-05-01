import os
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch
import numpy as np
import itertools

from random import shuffle
from time import time
from sklearn.cluster import KMeans

from models import LeNetMNIST, LeNetCIFAR, LeNetVOC, LeNetCIFAR_train, LeNetMNIST_train, LeNet5, LeNetFilter


class Data:
	def __init__(self, dataset_name="MNIST"):
		self.dataset_name = dataset_name
		self.is_cuda = torch.cuda.is_available()
		self.batch_size = 128
		self.max_size = 32
		if self.dataset_name == "MNIST":
			self.root = './data'
			self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
			self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'maroon', 'coral', 'olive']
		elif self.dataset_name == "CIFAR":
			self.root = './cifar'
			self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
			self.colors = ['r', 'r', 'b', 'b', 'b', 'b', 'b', 'b', 'r', 'r', ]
			# ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'maroon', 'coral', 'olive']
		elif self.dataset_name == "VOC2012":
			self.root = '/media/jedrzej/Seagate/DATA/VOC2012/PyTorch/'
			self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
			"diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
			"sofa", "train", "tvmonitor"]
			self.colors = ["red", "blue", "gold", "black", "g", "magenta", "darkcyan", "olive", "aqua", "crimson",
							"peru", "mediumpurple", "c", "orange", "lightcoral", "lavender", "wheat", "y",
						   "darksalmon", "silver"]
			self.max_size = 64
		if not os.path.exists(self.root):
			os.mkdir(self.root)
		self.n_categories = len(self.classes)
		self.networks = []

	def create_networks(self, n_networks):
		for i in xrange(n_networks):
			if self.dataset_name == "MNIST": model=LeNetFilter()  # model=LeNetMNIST()
			elif self.dataset_name == "CIFAR": model=LeNetFilter() # MobileNetV2()  # LeNetCIFAR()
			elif self.dataset_name == "VOC2012": model=LeNetVOC()
			self.networks.append(model)

	def create_network(self):
		kernels = [3, 5, 7, 9]
		kernel = np.random.choice(kernels, 1)[0]
		gains = [0.9, 1.0, 1.1]  # [0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 2.0, 5.0, 10.0]
		gain = np.random.choice(gains, 1)[0]
		if self.dataset_name == "MNIST":
			feat = (28-kernel+1)/2
			dim = 1
		elif self.dataset_name == "CIFAR":
			feat = (32-kernel+1)/2
			dim = 3
		#elif self.dataset_name == "VOC2012": model=LeNetVOC()
		return LeNetFilter(kernel, feat, dim, gain) # model=LeNetMNIST() # MobileNetV2()  # LeNetCIFAR()

	def load_dataset(self):
		if self.dataset_name == "MNIST":
			trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
			train_set = dset.MNIST(root=self.root, train=True, transform=trans, download=True)
			test_set = dset.MNIST(root=self.root, train=False, transform=trans, download=True)
		elif self.dataset_name == "CIFAR":
			trans = transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
					#transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
					])
			self.inv_normalize = transforms.Normalize(mean=[-0.5, -0.5, -0.5],
													  std=[1.0, 1.0, 1.0])
			#self.inv_normalize = transforms.Normalize(mean=[-0.4914/0.247, -0.4822/0.243, -0.4465/0.255], std=[1/0.247, 1/0.243, 1/0.261])
			train_set = dset.CIFAR10(root=self.root, train=True, download=True, transform=trans)
			test_set = dset.CIFAR10(root=self.root, train=False, download=True, transform=trans)

		elif self.dataset_name == "VOC2012":
			self.inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.4562/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
			trans = transforms.Compose([
					transforms.Resize(256),
					transforms.CenterCrop(224),
					transforms.ToTensor(),
					transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
					])
			train_set = dset.ImageFolder(root=self.root+"train", transform=trans)
			test_set = dset.ImageFolder(root=self.root+"val", transform=trans)

		self.train_loader = torch.utils.data.DataLoader(
		         dataset=train_set,
		         batch_size=self.batch_size,
		         shuffle=True)
		self.test_loader = torch.utils.data.DataLoader(
		        dataset=test_set,
		        batch_size=self.batch_size,
		        shuffle=True)

		print '==>>> total training batch number: {}'.format(len(self.train_loader))
		print '==>>> total testing batch number: {}'.format(len(self.test_loader))

	def get_data(self, dataloader, pick_k=None):
		all_inputs = [[] for i in xrange(self.n_categories)]
		ground_truth = [[] for i in xrange(self.n_categories)]
		for batch_idx, (x, target) in enumerate(dataloader):
			for i in xrange(self.n_categories):
				indices = np.where(target == i)[0]
				shuffle(indices)
				matching_digits = [x.data.cpu().numpy()[idx] for idx in indices]
				all_inputs[i].extend(matching_digits)
				ground_truth[i].extend([i for j in xrange(len(indices))])

		inputs = []
		gt = []
		for i in xrange(len(all_inputs)):
			if pick_k != None:
				inputs.extend(all_inputs[i][:pick_k])
				gt.extend(ground_truth[i][:pick_k])
			else:
				inputs.extend(all_inputs[i])
				gt.extend(ground_truth[i])
		return np.asarray(inputs), np.asarray(gt)

	def run_dataset(self, inputs, n_clusters=10, iters=151):

		tot_len = len(inputs)
	
		inputs = torch.from_numpy(inputs)
		start = time()

		possible_clusters = [cluster for cluster in xrange(10, n_clusters, 5)]

		# set max batch size:
		max_size = self.max_size
		first_pass = True
		i = 0
		while i<iters:
			affinity_matrix = np.zeros((tot_len,tot_len))
			# choose model:
			model = self.networks[i]
			if self.is_cuda:
				model.cuda()

			no_of_passes = -((-tot_len)//max_size)
			for k in xrange(no_of_passes):
				start_idx = k*max_size
				end_idx = min([(k+1)*max_size, tot_len])
				#if self.is_cuda:
					#temp_inputs = inputs[start_idx:end_idx].cuda()
				################### EXTRACTING FEATURES ################

				if k == 0:
					features = model(inputs[start_idx:end_idx].cuda()).data.cpu().numpy()
				else:
					features = np.concatenate((features, model(inputs[start_idx:end_idx].cuda()).data.cpu().numpy()), axis=0)

			try:
				################### CLUSTERING #########################
				n_clusters = np.random.choice(possible_clusters)
				kmeans = KMeans(n_clusters=n_clusters).fit(features)

				################## AFFINITY MATRIX ####################
				for j in xrange(n_clusters):
					indices = np.where(kmeans.labels_ == j)[0]
					for pair in itertools.product(indices, repeat=2):
							affinity_matrix[pair[0],pair[1]] += 1

				if first_pass:
					all_affinity_matrices = affinity_matrix[...]
					first_pass = False
				else:
					all_affinity_matrices += affinity_matrix
				i += 1
				del kmeans, indices
			except:
				print "Some error"
			del features, affinity_matrix
			if i % 10 == 0:
				print "Iteration:", i, "Time elapsed:", time()-start
				start = time()
		return all_affinity_matrices/i