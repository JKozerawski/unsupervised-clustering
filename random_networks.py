import numpy as np
from sklearn.cluster import MiniBatchKMeans
import itertools
import torch
import torch.optim as optim
from models import LeNet5, LeNetCIFAR_train, LeNetFilter
from itertools import permutations
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter
from math import radians

class RandomNetworks:
	def __init__(self, inputs, is_cuda, n_categories, n_of_models, radius = 1.0):
		self.is_cuda = is_cuda
		self.inputs = inputs
		self.tot_len = len(inputs)
		self.max_size = 128
		self.n_categories = n_categories
		self.n_clusters = n_categories
		self.n_of_models = n_of_models
		self.learning_rate = 0.1
		self.weight = 0.3
		self.radius = radius
		self.vectorized_points = np.asarray([np.zeros(n_categories) for i in xrange(n_categories)])
		for i in xrange(n_categories):
			self.vectorized_points[i,i] = radius
		#self.chosen_points = self.vectorized_points
		scales = np.arange(0.25,1.25,0.25)
		#self.chosen_points = np.asarray([[1.0,1.0], [1.0,0], [1.0,-1.0], [-1.0,-1.0], [-1.0,1.0], [-1.0,0], [0,-1.0], [0,1.0], [0.3,0.],[-0.3,0.]])
		self.chosen_points = np.asarray([pol2cart(self.radius, i * 360 / n_categories) for i in xrange(n_categories)])
		#print "Class initialized"



	def create_models(self, dset, model=None):
		all_models = []
		for i in xrange(self.n_of_models):
			if model=="filter":
				model = dset.create_network()

			elif dset.dataset_name=="MNIST": model = LeNet5()
			else: model = LeNetCIFAR_train()
			all_models.append(model)
		self.models = all_models
		#return all_models

	def our_accuracy(self, features, labels):
		predictions = []
		for i in xrange(len(features)):
			distances = np.asarray([np.linalg.norm(self.chosen_points[j] - features[i]) for j in xrange(len(self.chosen_points))])
			min_idx = np.argmin(distances)
			predictions.append(int(min_idx))
		return accuracy_score(labels, predictions), np.asarray(predictions)

	def get_features(self, model, input = None):
		if input == None:
			inputs = torch.from_numpy(self.inputs)
		else:
			inputs = torch.from_numpy(input)
		if self.is_cuda:
			model.cuda()
		tot_len = len(inputs)
		no_of_passes = -((-tot_len) // self.max_size)
		for k in xrange(no_of_passes):
			start_idx = k * self.max_size
			end_idx = min([(k + 1) * self.max_size, tot_len])
			if k == 0:
				features = model(inputs[start_idx:end_idx].cuda()).data.cpu().numpy()
			else:
				features = np.concatenate((features, model(inputs[start_idx:end_idx].cuda()).data.cpu().numpy()),
										  axis=0)
		return features

	def get_centroids(self, features, labels):
		centroids = []
		for j in xrange(self.n_categories):
			indices = np.where(labels == j)[0]
			current_centroid = np.mean(np.asarray([features[idx] for idx in indices]), axis=0)
			centroids.append(current_centroid)
		return np.asarray(centroids)

	def get_predicted_labels(self, features, labels, centroids):
		predicted_labels = []
		correct_categories = np.zeros(self.n_categories)
		for i, point in enumerate(features):
			min_dist = 1000000
			closest_centroid = -1
			for j in xrange(self.n_categories):
				dist = np.linalg.norm(point - centroids[j])
				if (dist < min_dist):
					closest_centroid = j
					min_dist = dist
			predicted_labels.append(closest_centroid)
			if (closest_centroid == labels[i]): correct_categories[closest_centroid] += 1
		return np.asarray(predicted_labels), correct_categories

	def centroids_to_points(self, centroids):
		return centroids
		chosen_points = self.chosen_points
		min_dist = 1000000
		for perm in permutations(np.arange(self.n_categories)):
			points = np.asarray([self.chosen_points[i] for i in perm])
			dist = np.sum(np.abs(points-centroids))
			if dist<min_dist:
				min_dist = dist
				chosen_points = points
		return chosen_points

	def get_target(self, inputs, features, centroids, mean_data, labels, predicted_labels, start_idx, chosen_points):
		target = []
		weight = self.weight
		for i in xrange(len(inputs)):
			label = labels[start_idx + i]
			#pred_label = predicted_labels[start_idx + i]
			# temp_target = label*np.ones(len(features[0]))
			# target.append(temp_target)
			#if (label == pred_label):
				#target.append(features[start_idx + i])
			#else:

			temp_target = self.chosen_points[label]#+np.random.normal(0.0, 0.5, (2))
			target.append(temp_target)
			#target.append(self.vectorized_points[label])
			#target.append(chosen_points[label])

			#target.append(centroids[label] + weight * (centroids[label] - mean_data))
		target = np.asarray(target)
		# target = np.asarray([centroids[label] for label in labels[start_idx:end_idx]])
		return torch.from_numpy(target).cuda().float()

	def create_affinity_matrix(self):
		affinity_matrix = np.zeros((self.tot_len, self.tot_len))
		for model in self.models:
			model.eval()
			features = self.get_features(model)
			affinity_matrix = (affinity_matrix + self.cluster_features(features))
		return affinity_matrix

	def cluster_features(self, features):
		affinity_matrix = np.zeros((self.tot_len, self.tot_len))
		try:
			kmeans = MiniBatchKMeans(n_clusters=self.n_clusters).fit(features)
			for j in xrange(self.n_clusters):
				indices = np.where(kmeans.labels_ == j)[0]
				for pair in itertools.product(indices, repeat=2):
					affinity_matrix[pair[0], pair[1]] += 1
			del kmeans, indices
		except:
			print "Error clustering network output"
		return affinity_matrix

	def clustering_accuracy(self, predictions, ground_truth):
		max_trace = 0
		c_matrix = confusion_matrix(ground_truth, predictions)
		for perm in permutations(np.arange(self.n_categories)):
			i = np.argsort(perm)
			temp_c_matrix = c_matrix[:, i]
			curr_trace = np.trace(temp_c_matrix)
			if curr_trace > max_trace:
				max_trace = curr_trace
		return max_trace / float(np.sum(c_matrix))

	def clustering_accuracy_fast(self, predictions, ground_truth):
		max_trace = 0
		c_matrix = confusion_matrix(predictions, ground_truth)
		permutation = -1*np.ones((self.n_categories),dtype=int)
		for i in xrange(self.n_categories):
			permutation[i] = np.argmax(c_matrix[i,...])
		if len(np.unique(permutation))!=self.n_categories:
			# permute over what repeats and what is absent
			repeated_elements = [item for item, count in Counter(permutation).items() if count > 1]
			missing_elements = list(set(np.arange(self.n_categories).tolist()) - set(permutation))
			elements_to_use = repeated_elements+missing_elements

			indices_of_repetitions = []
			for elem in repeated_elements:
				indices_of_repetitions.extend(np.where(permutation==elem)[0])
			# try only those permutations now:
			for perm in permutations(elements_to_use):
				new_permutations = permutation.copy()
				for i in xrange(len(indices_of_repetitions)):
					new_permutations[indices_of_repetitions[i]] = perm[i]
				temp_c_matrix = c_matrix[:, new_permutations]
				curr_trace = np.trace(temp_c_matrix)
				if curr_trace > max_trace:
					max_trace = curr_trace
				#print curr_trace / float(np.sum(c_matrix))
			new_predictions = [new_permutations[i] for i in predictions]
		else:
			temp_c_matrix = c_matrix[:, permutation]
			max_trace =  np.trace(temp_c_matrix)
			new_predictions = [permutation[i] for i in predictions]
		return max_trace / float(np.sum(c_matrix)), np.asarray(new_predictions, dtype=int)


	def kmeans_regular(self, data):
		data = self.flatten_data(data)
		kmeans = MiniBatchKMeans(n_clusters=self.n_clusters).fit(data)
		return kmeans.labels_, kmeans.cluster_centers_

	def flatten_data(self, data):
		return np.reshape(data, (len(data), -1))

	def update_single_network(self, model, labels):
		no_of_passes = -((-self.tot_len) // self.max_size)
		features = self.get_features(model)
		#optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
		optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
		mean_data = np.mean(features, axis=0)
		centroids = self.get_centroids(features, labels)
		predicted_labels, correct_categories = self.get_predicted_labels(features, labels, centroids)
		chosen_points = self.centroids_to_points(centroids)
		model.train()
		for i in xrange(1):
			train_loss = 0
			if self.is_cuda:
				model.cuda()
			for k in xrange(no_of_passes):
				start_idx = k * self.max_size
				end_idx = min([(k + 1) * self.max_size, self.tot_len])
				optimizer.zero_grad()
				inputs = torch.from_numpy(self.inputs[start_idx:end_idx]).cuda()
				target = self.get_target(inputs, features, centroids, mean_data, labels, predicted_labels, start_idx, chosen_points)
				output = model(inputs)  # .data.cpu().numpy()
				loss_fn = torch.nn.MSELoss()
				loss = loss_fn(output, target)
				loss.backward()
				optimizer.step()
				train_loss += loss.item()
		# print i+1, "Loss:", train_loss
		del features, centroids, inputs
		return train_loss

	def update_all_networks(self, labels):
		loss = 0.0
		for model in self.models:
			loss += self.update_single_network(model, labels)
		return loss/len(self.models)

def pol2cart(rho, phi):
	phi = radians(phi)
	x = rho * np.cos(phi)
	y = rho * np.sin(phi)
	return [x, y]