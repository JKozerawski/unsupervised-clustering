import torch
import torch.optim as optim
import torch.nn as nn
from sklearn import mixture
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import entropy
from torch.autograd import Variable
from time import time
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

from load_data import Data
from models import LeNetMNIST, LeNetCIFAR, LeNetVOC, LeNetCIFAR_train, LeNetMNIST_train

class IterGMM():
	def __init__(self):
		print "Initialized"
		self.is_cuda = torch.cuda.is_available()


	def set_dataloader(self, dataloader):
		self.dataloader = dataloader

	def train(self, dataset_name, n_epochs = 7*60):


		print "Training"
		if(dataset_name == "MNIST"):
			model = LeNetMNIST_train()
			optimizer = optim.SGD(model.fc2.parameters(), lr=0.01)
		elif(dataset_name == "CIFAR"):
			model = LeNetCIFAR_train()
			optimizer = optim.SGD(model.fc3.parameters(), lr=0.1)

		if self.is_cuda:
			model.cuda()	
		criterion = nn.CrossEntropyLoss()
		criterion2 = nn.MSELoss()
		model.train()
		for epoch in xrange(n_epochs):
			train_loss = 0.0
			start = time()

			for data, target in self.dataloader:
				if self.is_cuda:
					data, target = data.cuda(), target.cuda()
				'''
				# pass to get the features:
				features = []
				for data_feat, target_feat in self.dataloader:
					if self.is_cuda:
						data_feat, target_feat = data_feat.cuda(), target_feat.cuda()
					optimizer.zero_grad()

					output_feat = model(data_feat).data.cpu().numpy()
					features.extend(output_feat)

				# features obtained:
				features = np.asarray(features)
				np.random.shuffle(features)
		
				#pca = PCA(n_components=2)
				#vis_features = pca.fit_transform(features)
				# perform gmm
				g = mixture.GaussianMixture(n_components=2, random_state=40)
				# get the entropy for every point
				g. fit(features)
				# calculate loss for backprop
				'''
				optimizer.zero_grad()
				output = model(data)
				#print output
				loss1 = criterion(output, target)
				#print output
				#print output
				
				n = int(data.size(0))
			
				loss_num = float(loss1.data.cpu().numpy())
				
				for i in xrange(n):
					output[i].data.copy_(torch.tensor([loss_num for j in xrange(10)]))
				#print target
				#loss_var = Variable(torch.tensor([loss_num for i in xrange(n)])).cuda()
				target_var = Variable(torch.tensor([[0. for j in xrange(10)] for i in xrange(n)])).cuda()
				#target_var = target_var
				#loss_var.grad_fn = output.grad_fn
				#print loss_var
				loss = criterion2( output.cuda(), target_var.cuda())
				'''
				print "True loss:"
				print loss
				#output = pca.fit_transform(output)
				output = output.data.cpu().numpy()
				probs = g.predict_proba(output)
		
				entropy_data = np.mean(np.asarray([entropy(prob) for prob in probs]))
				loss = Variable(torch.tensor([entropy_data]).data, requires_grad=True)
				print "Our loss:"
				print loss
				'''
				loss.backward()
				# perform a single optimization step (parameter update)
				optimizer.step()
				# update training loss
				train_loss += loss.item()*data.size(0)

				
			
			#train_loss = train_loss/len(self.dataloader.dataset)
			train_loss = train_loss/len(self.dataloader.dataset)
			# print training/validation statistics 
			print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
			#print "time:",time()-start


		def draw_ellipse(position, covariance, ax=None, **kwargs):
			"""Draw an ellipse with a given position and covariance"""
			ax = ax or plt.gca()

			# Convert covariance to principal axes
			if covariance.shape == (2, 2):
				U, s, Vt = np.linalg.svd(covariance)
				angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
				width, height = 2 * np.sqrt(s)
			else:
				angle = 0
				width, height = 2 * np.sqrt(covariance)

			# Draw the Ellipse
			for nsig in range(1, 4):
				ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))

		def plot_gmm(gmm, X, label=True, ax=None):
			ax = ax or plt.gca()
			labels = gmm.fit(X).predict(X)
			if label:
				ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
			else:
				ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
			ax.axis('equal')

			w_factor = 0.2 / gmm.weights_.max()
			for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
				draw_ellipse(pos, covar, alpha=w * w_factor)
			plt.show()
		plot_gmm(g, features)
		'''
		x = features[:,0]
		y = features[:,1]
		X, Y = np.meshgrid(x, y)
		XX = np.array([X.ravel(), Y.ravel()]).T
		Z = -g.score_samples(XX)
		Z = Z.reshape(X.shape)

		CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
		levels=np.logspace(0, 3, 10))

		plt.scatter(x,y)
		plt.show()
		'''


dset = Data("MNIST")
dset.load_dataset()

gmm = IterGMM()
gmm.set_dataloader(dset.test_loader)
gmm.train("MNIST")
