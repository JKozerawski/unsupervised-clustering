import os
import torchvision.models as mdl
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt
import cv2
from random import shuffle
from scipy.ndimage import rotate
from time import time
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from models import LeNetMNIST, LeNetCIFAR, LeNetVOC, LeNetCIFAR_train, LeNetMNIST_train

class Data():
	def __init__(self, dataset_name = "MNIST"):
		self.dataset_name = dataset_name
		self.is_cuda = torch.cuda.is_available()
		self.batch_size = 32
		self.max_size = 256
		if(self.dataset_name=="MNIST"):
			self.root = './data'
			self.classes = ['0','1','2','3','4','5','6','7','8','9']
			self.colors = ['b','g','r','c','m','y','k','maroon','coral','olive']
		elif(self.dataset_name == "CIFAR"):
			self.root = './cifar'
			self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
			self.colors = ['r','r','b','b','b','b','b','b','r','r',]#['b','g','r','c','m','y','k','maroon','coral','olive']
		elif(self.dataset_name == "VOC2012"):
			self.root = '/media/jedrzej/Seagate/DATA/VOC2012/PyTorch/'
			self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair","cow",
					"diningtable", "dog", "horse", "motorbike", "person", "pottedplant","sheep", "sofa", "train", "tvmonitor"]
			self.colors = ["red", "blue", "gold", "black", "g", "magenta", "darkcyan", "olive", "aqua","crimson",
					"peru", "mediumpurple", "c", "orange", "lightcoral", "lavender","wheat", "y", "darksalmon", "silver"]
			self.max_size = 64
		if not os.path.exists(self.root):
	    		os.mkdir(self.root)
		self.n_categories = len(self.classes)

	def create_networks(self, n_networks):
		self.networks = []
		for i in xrange(n_networks):
			if(self.dataset_name == "MNIST"): model =  LeNetMNIST()
			elif(self.dataset_name == "CIFAR"): model = LeNetCIFAR()
			elif(self.dataset_name == "VOC2012"): model = LeNetVOC()
			self.networks.append(model)

		print "Networks created"

	def cluster_affinity_matrix(self, n_clusters=10):
		#n_clusters = = self.n_categories
		mean_affinity_matrix = np.mean(self.all_affinity_matrices,axis=0)
		clustering = SpectralClustering(n_clusters = n_clusters,affinity="precomputed", assign_labels="discretize").fit(mean_affinity_matrix)
		self.predicted_labels = clustering.labels_
		self.show_predicted_groups()

	def show_predicted_groups(self):
		n = len(np.unique(self.predicted_labels))
		to_show = 20
		columns = []
		for i in xrange(n):
			indices = np.random.choice(np.where(self.predicted_labels==i)[0], to_show, replace=False)
			row = []
			for idx in indices:
				row.append(self.get_image(idx))
			image = row[0]
			for i in xrange(1, to_show):
				image = np.concatenate((image, row[i]), axis=1)
			columns.append(image)
		image = columns[0]
		for i in xrange(1,len(columns)):
			image = np.concatenate((image, columns[i]), axis=0)
		#cv2.imwrite('./images.jpg', image)
		cv2.imshow('Images', image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def load_dataset(self):
		if(self.dataset_name == "MNIST"):
			trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
			train_set = dset.MNIST(root=self.root, train=True, transform=trans, download=True)
			test_set = dset.MNIST(root=self.root, train=False, transform=trans, download=True)
		elif(self.dataset_name == "CIFAR"):
			trans = transforms.Compose([
	    			transforms.ToTensor(),
	   			transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
				])
			self.inv_normalize = transforms.Normalize(mean=[-0.4914/0.247, -0.4822/0.243, -0.4465/0.255],std=[1/0.247, 1/0.243, 1/0.261])
			train_set = dset.CIFAR10(root=self.root, train=True, download=True, transform=trans)
			test_set = dset.CIFAR10(root=self.root, train=False, download=True, transform=trans)

		elif(self.dataset_name == "VOC2012"):
			self.inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.4562/0.224, -0.406/0.225],std=[1/0.229, 1/0.224, 1/0.225])
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

	def get_data(self, dataloader, pick_k = None):
		all_inputs = [[] for i in xrange(self.n_categories)]
		ground_truth = [[] for i in xrange(self.n_categories)]
		for batch_idx, (x, target) in enumerate(dataloader):
			for i in xrange(self.n_categories):
				indices = np.where(target==i)[0]
				shuffle(indices)
				matching_digits = [x.data.cpu().numpy()[idx] for idx in indices]
				all_inputs[i].extend(matching_digits)
				ground_truth[i].extend([i for j in xrange(len(indices))])

		inputs = []
		gt = []
		for i in xrange(len(all_inputs)):
			if(pick_k!=None):
				inputs.extend(all_inputs[i][:pick_k])
				gt.extend(ground_truth[i][:pick_k])
			else:
				inputs.extend(all_inputs[i])
				gt.extend(ground_truth[i])
		inputs = np.asarray(inputs)
		ground_truth = np.asarray(gt)
		return inputs, ground_truth

	def run_dataset(self, inputs, n_clusters = 10, iters = 151):
		
		save_every = 5

		tot_len = len(inputs)

		all_affinity_matrices = []
		temp_affinity_matrices = []
	
		inputs = torch.from_numpy(inputs)
		start = time()

		possible_clusters = [i for i in xrange(10,n_clusters,5)]

		# set max batch size:
		max_size = self.max_size
		
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
				end_idx = min([(k+1)*max_size,tot_len])
				if self.is_cuda:
					temp_inputs = inputs[start_idx:end_idx].cuda()
				################### EXTRACTING FEATURES ################
				if(k==0):
					features = model(temp_inputs).data.cpu().numpy()
				else:
					features = np.concatenate((features, model(temp_inputs).data.cpu().numpy()), axis = 0)

			try:
				################### CLUSTERING #########################
				n_clusters = np.random.choice(possible_clusters)
				kmeans = KMeans(n_clusters=n_clusters).fit(features)

				################## AFFINITY MATRIX ####################
				for j in xrange(n_clusters):
					indices = np.where(kmeans.labels_==j)[0]
					for pair in itertools.product(indices, repeat=2):
							affinity_matrix[pair[0],pair[1]]+=1
				temp_affinity_matrices.append(affinity_matrix)
				i+=1
				del kmeans, indices
			except:
				print "Some error"
			del features, affinity_matrix
			if (i%save_every==0):
				print "Iteration:",i, "Time elapsed:", time()-start
				start = time()
				all_affinity_matrices.append(np.mean(np.asarray(temp_affinity_matrices),axis=0))
				del temp_affinity_matrices
				temp_affinity_matrices = []
		return np.asarray(all_affinity_matrices)

	def train_net(self):
		print "Training"
		if(self.dataset_name == "MNIST"):
			model = LeNetMNIST_train()
			optimizer = optim.SGD(model.fc2.parameters(), lr=0.1)
		elif(self.dataset_name == "CIFAR"):
			model = LeNetCIFAR_train()
			optimizer = optim.SGD(model.fc3.parameters(), lr=0.01)

		# specify loss function
		criterion = nn.CrossEntropyLoss()

		n_epochs = 30 # you may increase this number to train a final model

		valid_loss_min = np.Inf # track change in validation loss

		if self.is_cuda:
			model.cuda()	
		for epoch in range(1, n_epochs+1):

			# keep track of training and validation loss
			train_loss = 0.0

			###################
			# train the model #
			###################

			model.train()
			for data, target in self.train_loader:
				# move tensors to GPU if CUDA is available
				if self.is_cuda:
					data, target = data.cuda(), target.cuda()
				# clear the gradients of all optimized variables
				optimizer.zero_grad()
				# forward pass: compute predicted outputs by passing inputs to the model
				output = model(data)
				# calculate the batch loss
				loss = criterion(output, target)
				# backward pass: compute gradient of the loss with respect to model parameters
				loss.backward()
				# perform a single optimization step (parameter update)
				optimizer.step()
				# update training loss
				train_loss += loss.item()*data.size(0)

			train_loss = train_loss/len(self.train_loader.dataset)
			# print training/validation statistics 
			print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
		#return model

	def visualize_data(self, affinity_matrix):


		print "Affinity matrix shape", np.shape(affinity_matrix)
		def on_pick(event):
			category = None
			if(event.artist in plots1):
				category = plots1.index(event.artist)
			elif(event.artist in plots2):
				category = plots2.index(event.artist)
			elif(event.artist in plots3):
				category = plots3.index(event.artist)
			#elif(event.artist in plots4):
			#	category = plots4.index(event.artist)
			self.show_image(event.ind[0], category)

		categories = np.unique(self.ground_truth)
		n_categories = len(categories)
		data = self.perform_pca(affinity_matrix)
		data = self.normalize_data(data)


		self.visualize_dimension(data)

		pt_size = 10
		fig, axarr = plt.subplots(2, 2)

		plots1 = [[] for i in xrange(len(categories))]
		plots2 = [[] for i in xrange(len(categories))]
		plots3 = [[] for i in xrange(len(categories))]
		#plots4 = [[] for i in xrange(len(categories))]
		for i in categories:
			indices = np.where(self.ground_truth == i)[0]
			plots1[i] = axarr[0,0].scatter([data[j,0] for j in indices],[data[j,1] for j in indices], picker = 5, s=pt_size, label=self.classes[i], c=self.colors[i])
			plots2[i] = axarr[0,1].scatter([data[j,1] for j in indices],[data[j,2] for j in indices], picker = 5, s=pt_size, label=self.classes[i], c=self.colors[i])
			plots3[i] = axarr[1,0].scatter([data[j,2] for j in indices],[data[j,3] for j in indices], picker = 5, s=pt_size, label=self.classes[i], c=self.colors[i])
			#plots4[i] = axarr[1,1].scatter([data[j,3] for j in indices],[data[j,4] for j in indices], picker = 5, s=pt_size, label=self.classes[i], c=self.colors[i])

		axarr[0, 0].set_title('PCA dims: 1 & 2')
		axarr[0, 1].set_title('PCA dims: 2 & 3')
		axarr[1, 0].set_title('PCA dims: 3 & 4')
		#axarr[1, 1].set_title('PCA dims: 4 & 5')
		# show legend:
		axarr[0,0].legend(bbox_to_anchor=(-0.3, 1), loc='upper left', ncol=1, fontsize = 10)
		fig.canvas.mpl_connect('pick_event', on_pick)
		plt.show()
		return data

	def normalize_data(self, data, k = 5):
		data = data[:,:k]
		data_center = np.mean(data)
		data = data - data_center
		for i in xrange(k):
			data[:,i] = data[:,i]/max([abs(np.min(data[:,i])), abs(np.max(data[:,i]))])
		return data

	def perform_pca(self, affinity_matrix, n_components = 5):
		self.pca = PCA(n_components=n_components)
		return self.pca.fit_transform(affinity_matrix)

	def get_image(self, idx):
		if(self.dataset_name == "CIFAR"):
			img =self.inv_normalize(torch.from_numpy(self.inputs_small[idx].copy())).numpy()
			img = cv2.cvtColor(np.transpose(img, (1, 2, 0)),cv2.COLOR_BGR2RGB)
		elif(self.dataset_name == "MNIST"):
			img = self.inputs_small[idx,0].copy()
		return cv2.resize(img, (0,0), fx=3.0, fy=3.0)

	def show_image(self, idx, category):
		#print idx, category
		#print np.shape(self.inputs_small)
		img_per_category = len(self.inputs_small)/self.n_categories
		idx = img_per_category*category+idx

		#plt.figure(1)
		if(self.dataset_name == "CIFAR"):
			img =self.inv_normalize(torch.from_numpy(self.inputs_small[idx].copy())).numpy()
			plt.imshow(np.transpose(img, (1, 2, 0)), interpolation='nearest')
		elif(self.dataset_name == "MNIST"):
			img = self.inputs_small[idx,0].copy()
			plt.matshow(img)
		plt.grid(False)
		plt.show()

	def visualize_dimension(self, data, dims = [0,1,2,3]):
		no_images = 20
		all_images = []
		for dim in dims:
			min_value = min(data[:,dim])
			max_value = max(data[:,dim])
			thresholds = np.linspace(min_value, max_value, num = no_images)

			images = []
			print "Data shape", np.shape(data)
			for i in xrange(no_images-1):
			
				indices = np.intersect1d( np.where(data[:,dim]>=thresholds[i])[0], np.where(data[:,dim]<=thresholds[i+1])[0] )
				points = np.asarray([data[j] for j in indices])
				'''
				good_indices = np.abs(points[:,dim+1]).argsort()[:10]#[::-1]
				#print indices
				if(dim==0): 
					print thresholds[i], thresholds[i+1]
					print [points[k,dim+1] for k in good_indices]
				'''
				'''
				img = self.get_image(indices[0])
				for ind in indices[1:]:
					img+=self.get_image(ind)
				img = img/len(indices)
				images.append(img)
				'''
				idx = indices[np.argmin(np.abs(points[:,dim+1]))]

				#points = np.delete(points,dim,1)
				#idx = np.argmin(np.linalg.norm(points, axis = 1))
				images.append(self.get_image(idx))
				del indices, points
			image = images[0]
			for i in xrange(1, no_images-1):
				image = np.concatenate((image, images[i]), axis=1)
			all_images.append(image)
			del image, images

		image = all_images[0]
		for i in xrange(1,len(all_images)):
			image = np.concatenate((image, all_images[i]), axis=0)
		cv2.imwrite('./images.jpg', image)
		cv2.imshow('Images', image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

def modify_data(inputs, cat_1 = 0, cat_2 = 1):

	# choose only 0's and 1's
	n = len(inputs)
	zeros = inputs[cat_1*n/10:(cat_1+1)*n/10]
	ones = inputs[cat_2*n/10:(cat_2+1)*n/10]

	# shuffle them
	np.random.shuffle(zeros)
	np.random.shuffle(ones)

	# split in halves:
	m = len(zeros)/2
	normal_zeros = zeros[:m]
	modified_zeros = zeros[m:]
	normal_ones = ones[:m]
	modified_ones = ones[m:]
	
	# modify other halves:

	# option A: negative
	
	old_zeros = modified_zeros[...]
	
	modified_zeros = -modified_zeros
	modified_ones = -modified_ones
	'''
	# option B: rotate
	modified_zeros = rotate(modified_zeros,90, (2,3))
	modified_ones = rotate(modified_ones,90, (2,3))
	'''
	print np.shape(old_zeros), np.shape(modified_zeros)
	f, axarr = plt.subplots(1, 2)
	axarr[0].matshow(old_zeros[0,0,...])
	axarr[1].matshow(modified_zeros[0,0,...])
	plt.show()

	new_inputs = np.concatenate((normal_zeros, modified_zeros, normal_ones, modified_ones), axis = 0)
	new_ground_truth = np.concatenate((np.zeros(m),np.ones(m), 2*np.ones(m), 3*np.ones(m)), axis = 0)
	new_classes = ["Normal "+str(cat_1)+"'s", "Color-reversed "+str(cat_1)+"'s", "Normal "+str(cat_2)+"'s", "Color-reversed "+str(cat_2)+"'s"]
	print np.shape(new_inputs)
	print np.shape(new_ground_truth)
	colors = ['r','g','b','k']
	return new_inputs, new_ground_truth, new_classes, colors

'''
def test_net(model, inputs, labels, classes):
	tot_len = len(inputs)
	affinity_matrix = np.zeros((tot_len,tot_len))
	inputs = torch.from_numpy(inputs)
	if use_cuda:
    		model.cuda()
		inputs = inputs.cuda()
	################### EXTRACTING FEATURES ################
	features = model(inputs)
	features = features.data.cpu().numpy()	

	################### CLUSTERING #########################
	kmeans = KMeans(n_clusters=2).fit(features)

	################## AFFINITY MATRIX ####################
	for j in xrange(2):
		indices = np.where(kmeans.labels_==j)[0]
		for pair in itertools.product(indices, repeat=2):
				affinity_matrix[pair[0],pair[1]]+=1
	cluster_to_class(kmeans.labels_, labels, classes, n_clusters=2)
	return affinity_matrix
'''
