import warnings
import os
import numpy as np
import shutil
from load_data import Data
from time import time
from matplotlib import pyplot as plt
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import cv2
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.special import expit

from random_networks import RandomNetworks
from models import LeNet5, LeNetCIFAR_train
from sklearn.cluster import MiniBatchKMeans, SpectralClustering
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from itertools import combinations

from random_networks import pol2cart

warnings.filterwarnings("ignore", category=DeprecationWarning)
#datset_name = "MNIST"
datset_name = "CIFAR"

shutil.rmtree('./temp/')
os.makedirs('./temp/')

def get_data(dset):
    #train_data, train_labels = dset.get_data(dset.train_loader, 500)

    N = 100
    dim = 2
    '''
    norm = np.random.normal#(0,1,dim)
    normal_deviates = norm(size=(dim, N))

    radius = np.sqrt((normal_deviates ** 2).sum(axis=0))
    points = normal_deviates / radius

    #fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    plt.scatter(*points)
    #ax.set_aspect('equal')

    plt.show()
    return
    '''
    train_images = 2000
    val_images = 1000
    test_images = 1000
    n_categories = 10
    categories = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    categories_2 = [0, 1, 2, 3, 4, 5, 6]

    trainval_data, trainval_labels = dset.get_data(dset.train_loader, train_images+val_images)
    test_data, test_labels = dset.get_data(dset.test_loader, test_images)

    # choose only n categories:
    trainval_data, trainval_labels = select_n_categories(trainval_data, trainval_labels, n_categories, train_images+val_images, categories)
    test_data, test_labels = select_n_categories(test_data, test_labels, n_categories, test_images, categories)
    train_data, train_labels, val_data, val_labels = split_train_val(trainval_data, trainval_labels, train_images, val_images)
    train_data, train_labels = shuffle_data(train_data, train_labels)
    val_data, val_labels = shuffle_data(val_data, val_labels)
    test_data, test_labels = shuffle_data(test_data, test_labels)

    '''
    for i in xrange(10):
        print i
        new_train_labels = get_hierarchy_category(train_labels, c=i)
        new_val_labels = get_hierarchy_category(val_labels, c=i)
        new_test_labels = get_hierarchy_category(test_labels, c=i)
        radius = 0.7
        randomNet = train_single_network_ours(train_data, new_train_labels, val_data, new_val_labels,
                                              test_data, new_test_labels, dset, n_categories=2, radius=radius)

    return
    '''
    all_accu = []
    right_categories = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    left_categories = []
    for j in xrange(5):
        best_accu = 0
        best_category = -1
        for i in right_categories:
            chosen_categories = left_categories+[i]
            #print chosen_categories
            new_train_labels = get_two_groups(train_labels, chosen_categories)
            new_val_labels = get_two_groups(val_labels, chosen_categories)
            new_test_labels = get_two_groups(test_labels, chosen_categories)
            radius = 1.0
            randomNet, accu = train_single_network_ours(train_data, new_train_labels, val_data, new_val_labels,
                                                  test_data, new_test_labels, dset, n_categories=2, radius=radius)
            if accu>best_accu:
                best_accu = accu
                best_category = i
                print "Improvement:", best_accu, left_categories+[best_category]
        print "Best accuracy:", best_accu,  left_categories+[best_category]
        all_accu.append(best_accu)
        left_categories.append(best_category)
        right_categories.remove(best_category)
    return
    for perm in combinations(np.arange(n_categories), 5):
        print perm
        new_train_labels = get_two_groups(train_labels, perm)
        new_val_labels = get_two_groups(val_labels, perm)
        new_test_labels = get_two_groups(test_labels, perm)
        radius = 0.7
        randomNet = train_single_network_ours(train_data, new_train_labels, val_data, new_val_labels,
                                              test_data, new_test_labels, dset, n_categories=2, radius=radius)
    return
    train_labels_hierarchy = get_hierarchy(train_labels)
    val_labels_hierarchy = get_hierarchy(val_labels)
    test_labels_hierarchy = get_hierarchy(test_labels)

    train_labels_hierarchy_2 = get_hierarchy_2(train_labels)
    val_labels_hierarchy_2 = get_hierarchy_2(val_labels)
    test_labels_hierarchy_2 = get_hierarchy_2(test_labels)

    #baseline_hog(dset, test_data, test_labels, n_categories)
    #baseline_sift(dset, test_data, test_labels, n_categories)
    #baseline_kmeans(dset, test_data, test_labels, n_categories)
    #get_random_network_accuracy(dset, test_data, test_labels, n_categories)
    #get_random_network_accuracy(dset, test_data, test_labels, n_categories, model_type="filter")
    #get_random_networks_trained_accuracy(dset, test_data, test_labels, n_categories)

    #train_single_network_classical(train_data, train_labels_hierarchy, val_data, val_labels_hierarchy, test_data,
                                   #test_labels_hierarchy, dset, n_categories=2)
    radii = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 5.0]
    #for radius in radii:
    radius = 0.7
    randomNet = train_single_network_ours(train_data, train_labels_hierarchy, val_data, val_labels_hierarchy, test_data,
                                          test_labels_hierarchy, dset, n_categories=2, radius=radius)

    new_radius = 1.0*radius

    randomNet.chosen_points = np.zeros((10,2))
    for idx, elem in enumerate([0,1,8,9]):
        # divide in 6
        randomNet.chosen_points[elem] = pol2cart(new_radius, (idx * 360. / 4)-45)+np.asarray([radius,0])
        #randomNet.chosen_points[elem] = pol2cart(new_radius, (idx-1 * 360. / 10)-18)
    for idx, elem in enumerate([2,3,4,5,6,7]):
        # divide in 8
        randomNet.chosen_points[elem] = pol2cart(new_radius, (idx * 360. / 6) - 30) + np.asarray([-radius,0])
        #randomNet.chosen_points[elem] = pol2cart(new_radius, (idx - 1 * 360. / 14) - (360./28)+180)
    randomNet.chosen_points = np.asarray([[radius,new_radius],[radius,-new_radius],[-radius,new_radius],[-radius-new_radius,0],[-radius,-new_radius]])
    randomNet.learning_rate = 0.1
    randomNet = train_single_network_ours(train_data, train_labels_hierarchy_2, val_data, val_labels_hierarchy_2, test_data,
                                          test_labels_hierarchy_2, dset, n_categories=5, radius=radius, randomNetwork=randomNet)

    '''
    randomNetworks = RandomNetworks(inputs = train_data, is_cuda=dset_class.is_cuda, n_categories=n_categories, n_of_models=50)
    predictions_baseline, centroids = randomNetworks.kmeans_regular(train_data)
    accu, predictions_baseline = randomNetworks.clustering_accuracy_fast(predictions_baseline, train_labels)
    print "Baseline kmeans accuracy:", accu
    randomNetworks.create_models(dset)
    all_accuracy = []
    start = time()
    for i in xrange(20000):
        affinity_matrix = randomNetworks.create_affinity_matrix() # run_random_networks(test_data, models, N_CATEGORIES)
        predictions_ours, centroids = randomNetworks.kmeans_regular(affinity_matrix)
        #predicted_labels, correct_categories = get_predicted_labels(affinity_matrix, test_labels, centroids)
        #accu = np.mean(correct_categories)/float(images_per_class)
        average_loss = randomNetworks.update_all_networks(predictions_ours)
        if i%1==0:
            for j in xrange(5):
                features = randomNetworks.get_features(randomNetworks.models[j], input=train_data)
                predictions, centroids = randomNetworks.kmeans_regular(features)
                accu_train, predictions = randomNetworks.clustering_accuracy_fast(predictions, train_labels)
                show_feature_space(features, train_labels, predictions, accu_train, 10*j+i + 1, n_categories)
                print j, "Single net classification accuracy:", accu_train
            accu, predictions_ours = randomNetworks.clustering_accuracy_fast(predictions_ours, train_labels)
            all_accuracy.append(accu)
            print "Iteration:",i, "Classification accuracy:", accu, "Average loss:", average_loss,"Elapsed:", time()-start
            start = time()
        if i%450:
            randomNetworks.learning_rate = 0.001
    plt.plot(all_accuracy)
    plt.show()
    '''

def baseline_hog(dset, data, labels, n_categories):
    features = get_hog_features(dset, data)
    randomNetworks = RandomNetworks(inputs=data, is_cuda=dset.is_cuda, n_categories=n_categories, n_of_models=1)
    predictions, centroids = randomNetworks.kmeans_regular(features)
    accu, predictions = randomNetworks.clustering_accuracy_fast(predictions, labels)
    print "HOG accuracy:", accu

def baseline_kmeans(dset, data, labels, n_categories):
    randomNetworks = RandomNetworks(inputs=data, is_cuda=dset.is_cuda, n_categories=n_categories, n_of_models=1)
    predictions, centroids = randomNetworks.kmeans_regular(data)
    accu, predictions = randomNetworks.clustering_accuracy_fast(predictions, labels)
    print "Baseline K-Means accuracy:", accu

def baseline_sift(dset, data, labels, n_categories):
    sift_features = extract_sift(dset, data)
    randomNetworks = RandomNetworks(inputs=data, is_cuda=dset.is_cuda, n_categories=n_categories, n_of_models=1)
    predictions, centroids = randomNetworks.kmeans_regular(sift_features)
    accu, predictions = randomNetworks.clustering_accuracy_fast(predictions, labels)
    print "SIFT accuracy:", accu

def get_hog_features(dset, data):
    start = time()
    features = [[] for i in xrange(len(data))]
    hog = cv2.HOGDescriptor()
    for i in xrange(len(data)):
        img = dset.inv_normalize(torch.from_numpy(data[i].copy())).numpy()
        img = cv2.cvtColor(np.transpose(img, (1, 2, 0)), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (160,160))
        img = np.uint8(255 * img)
        features[i] = hog.compute(img)
        del img
    print "HOG features extracted", time()-start
    return np.asarray(features)

def get_random_network_accuracy(dset, data, labels, n_categories, model_type = None):
    randomNetworks = RandomNetworks(inputs=data, is_cuda=dset.is_cuda, n_categories=n_categories,
                                    n_of_models=150)
    randomNetworks.create_models(dset, model_type)
    affinity_matrix = randomNetworks.create_affinity_matrix()  # run_random_networks(test_data, models, N_CATEGORIES)
    predictions_ours, centroids = randomNetworks.kmeans_regular(affinity_matrix)
    accu, predictions_ours = randomNetworks.clustering_accuracy_fast(predictions_ours, labels)
    print "Random networks kmeans accuracy:", accu
    spectral = MiniBatchKMeans(n_clusters=n_categories).fit(affinity_matrix)
    accu_spectral, predictions_ours = randomNetworks.clustering_accuracy_fast(spectral.labels_, labels)
    print "Random networks spectral accuracy:", accu_spectral

def get_random_networks_trained_accuracy(dset, data, labels, n_categories, model_type = None):
    randomNetworks = RandomNetworks(inputs=data, is_cuda=dset.is_cuda, n_categories=n_categories,
                                    n_of_models=35)
    randomNetworks.create_models(dset, model_type)
    start = time()
    for i in xrange(10000):
        affinity_matrix = randomNetworks.create_affinity_matrix()
        predictions_ours, centroids = randomNetworks.kmeans_regular(affinity_matrix)
        # predicted_labels, correct_categories = get_predicted_labels(affinity_matrix, test_labels, centroids)
        # accu = np.mean(correct_categories)/float(images_per_class)
        average_loss = randomNetworks.update_all_networks(predictions_ours)
        if i % 20 == 0:
            accu, predictions_ours = randomNetworks.clustering_accuracy_fast(predictions_ours, labels)
            print "Iteration:", i, "Classification accuracy:", accu, "Average loss:", average_loss, "Elapsed:", time() - start
            start = time()
        if i % 200:
            randomNetworks.learning_rate = 0.01

def extract_sift(dset, data):

    dim = 64
    #sift = cv2.xfeatures2d.SIFT_create()
    sift = cv2.xfeatures2d.SURF_create()
    #sift = cv2.ORB_create()


    start = time()
    features = []
    train_features = []
    for i in xrange(len(data)):
        img = dset.inv_normalize(torch.from_numpy(data[i].copy())).numpy()
        img = cv2.cvtColor(np.transpose(img, (1, 2, 0)), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(160,160))
        img = np.uint8(255*img)
        kps, des = sift.detectAndCompute(img, None)
        if des is None: feature = np.zeros((1, dim))
        else: feature = np.asarray(des)
        train_features.append(feature)
        features.extend(feature)
    features = np.asarray(features)
    print "SIFT features extracted", time()-start
    start = time()
    # Bag of Visual Words:
    K = 20
    kmeans = MiniBatchKMeans(n_clusters=K).fit(features)  # KMeans(n_clusters=K).fit(features)

    # vectorize bag of words:
    vectors = []
    for i in xrange(len(data)):
        des = train_features[i]
        vector = np.zeros((1, K))
        labels = kmeans.predict(des)
        for j in xrange(len(des)):
            vector[0, labels[j]] += 1.0
        vectors.extend(normalize(vector))

    return np.asarray(vectors)

def select_n_categories(data, labels, n, images_per_class, categories = None):
    if categories == None :
        possible_categories = np.unique(labels).tolist()
        random.shuffle(possible_categories)
        chosen_categories = possible_categories[:n]
    else: chosen_categories = categories
    print "Chosen categories are:", chosen_categories
    chosen_data = data[chosen_categories[0]*images_per_class:(chosen_categories[0] + 1)*images_per_class]
    chosen_labels = np.zeros(images_per_class, dtype=np.int8)
    for i in xrange(1,len(chosen_categories)):
        category = chosen_categories[i]
        chosen_data = np.concatenate((chosen_data, data[category * images_per_class:(category + 1) * images_per_class]))
        chosen_labels = np.concatenate((chosen_labels, i*np.ones(images_per_class, dtype=np.int8)))
    return chosen_data, chosen_labels

def get_two_groups(labels, group1):
    for i in xrange(10):
        if(i in group1):
            labels = np.where(labels == i, 11, labels)
        else:
            labels = np.where(labels==i , 12, labels)
    labels = np.where(labels == 11, 0, labels)
    labels = np.where(labels == 12, 1, labels)
    return labels

def get_hierarchy_category(labels, c = 0):
    for i in xrange(10):
        if(i==c):
            labels = np.where(labels == i, 11, labels)
        else:
            labels = np.where(labels==i , 12, labels)
    labels = np.where(labels == 11, 0, labels)
    labels = np.where(labels == 12, 1, labels)
    return labels

def get_hierarchy(labels):
    for i in [0,1,8,9]:
        labels = np.where(labels==i , 0, labels)
    for i in [2,3,4,5,6,7]:
        labels = np.where(labels==i, 1, labels)
    return labels

def get_hierarchy_2(labels):
    for i in [0,8]:
        labels = np.where(labels==i , 0, labels)
    for i in [1,9]:
        labels = np.where(labels==i , 1, labels)
    for i in [2, 6]:
        labels = np.where(labels == i, 2, labels)
    for i in [3,5]:
        labels = np.where(labels==i, 3, labels)
    for i in [4,7]:
        labels = np.where(labels==i, 4, labels)
    return labels

def split_train_val(data, labels, train_images, val_images):
    trainval_images = train_images+val_images
    n = len(np.unique(labels))
    train_data = data[:train_images]
    train_labels = labels[:train_images]
    val_data = data[train_images:trainval_images]
    val_labels = labels[train_images:trainval_images]
    for i in xrange(1,n):
        train_data = np.concatenate((train_data, data[i * trainval_images:(i * trainval_images + train_images)]))
        train_labels = np.concatenate((train_labels, i * np.ones(train_images, dtype=np.int8)))
        val_data = np.concatenate((val_data, data[(i * trainval_images + train_images):((i +1)* trainval_images)]))
        val_labels = np.concatenate((val_labels, i * np.ones(val_images, dtype=np.int8)))
    return train_data, train_labels, val_data, val_labels

def shuffle_data(data, labels):
    s = np.arange(data.shape[0])
    np.random.shuffle(s)
    return np.asarray(data[s]), np.asarray(labels[s])

def train_single_network_classical(data, labels, val_data, val_labels, test_data, test_labels, dset, n_categories):
    randomNetwork = RandomNetworks(inputs=data, is_cuda=dset_class.is_cuda, n_categories=n_categories, n_of_models=1)
    print
    print "Classical approach"
    tot_len = len(data)
    max_size = 128
    no_of_passes = -((-tot_len) // max_size)
    model = LeNetCIFAR_train(final_size=n_categories)
    #model = LeNet5(final_size=n_categories)
    accuracies = [[],[],[]]
    iters = []
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    if dset.is_cuda:
        model.cuda()
    frame_count = 1
    for i in xrange(251):
        #if i==260: optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        if i == 10: optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        if i == 80: optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
        if i == 200: optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
        model.train()
        train_loss = 0
        for k in xrange(no_of_passes):
            start_idx = k * max_size
            end_idx = min([(k + 1) * max_size, tot_len])
            optimizer.zero_grad()
            inputs = torch.from_numpy(data[start_idx:end_idx]).cuda()
            target = torch.from_numpy(np.asarray(labels[start_idx:end_idx], dtype=float)).cuda().long()
            output = model(inputs)  # .data.cpu().numpy()
            loss_fn = torch.nn.NLLLoss()
            loss = loss_fn(F.log_softmax(output), target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        if (i % 10 == 0):
            # accuracy calculation:
            features = randomNetwork.get_features(model, input=data)
            features = expit(features)
            # show features, predictions etc
            #predictions, centroids = randomNetwork.kmeans_regular(features)
            #accu_train, predictions = randomNetwork.clustering_accuracy_fast(predictions, labels)
            accu_train, predictions  = classical_accuracy(features, labels)

            features = randomNetwork.get_features(model, input = val_data)
            features = expit(features)

            #predictions, centroids = randomNetwork.kmeans_regular(features)
            #accu_val, predictions = randomNetwork.clustering_accuracy_fast(predictions, val_labels)
            accu_val, _ = classical_accuracy(features, val_labels)
            features = randomNetwork.get_features(model, input=test_data)
            #predictions, centroids = randomNetwork.kmeans_regular(features)
            #accu_test, predictions = randomNetwork.clustering_accuracy_fast(predictions, test_labels)
            features = expit(features)
            accu_test, predictions = classical_accuracy(features, test_labels)
            #show_feature_space(features, test_labels, predictions, accu_test, frame_count, n_categories)
            frame_count += 1
            accuracies[0].append(accu_train)
            accuracies[1].append(accu_val)
            accuracies[2].append(accu_test)
            iters.append(i)
            print "Iteration:", i, "Train accuracy:", accu_train, "Val accuracy:", accu_val, "Test accuracy:", accu_test
    display_accuracy(accuracies, iters)
    #save_movie(movie_name="Classical")

def display_accuracy(accuracies, iters):
    plt.plot(iters, accuracies[0], label="Train", c='k')
    plt.plot(iters, accuracies[1], label="Val", c='b')
    plt.plot(iters, accuracies[2], label="Test", c='r')
    plt.legend()
    plt.title("Classification accuracy")
    plt.show()

def train_single_network_ours(data, labels, val_data, val_labels, test_data, test_labels, dset, n_categories, radius, randomNetwork = None):
    if(randomNetwork==None):
        randomNetwork = RandomNetworks(inputs = data, is_cuda=dset_class.is_cuda, n_categories=n_categories, n_of_models=1, radius=radius)
        randomNetwork.create_models(dset)
    #randomNetwork.radius = radius
    #print
    #print "Our approach with radius:", radius

    accuracies = [[], [], []]
    iters = []
    frame_count = 1
    for i in xrange(101):
        average_loss = randomNetwork.update_all_networks(labels)
        if i in [250, 550]:
            randomNetwork.learning_rate /= 10
        if(i==100):
            try:
                # accuracy calculation:
                features = randomNetwork.get_features(randomNetwork.models[0], input=data)
                #predictions, centroids = randomNetwork.kmeans_regular(features)
                #accu_train, predictions = randomNetwork.clustering_accuracy_fast(predictions, labels)
                accu_train, predictions = randomNetwork.our_accuracy(features, labels)

                #accu_train, predictions = classical_accuracy(features, labels)
                #show_feature_space(features, labels, predictions, accu_train, i+1, n_categories)
                features = randomNetwork.get_features(randomNetwork.models[0], input=val_data)
                #predictions, centroids = randomNetwork.kmeans_regular(features)
                #accu_val, predictions = randomNetwork.clustering_accuracy_fast(predictions, val_labels)
                accu_val, predictions = randomNetwork.our_accuracy(features, val_labels)
                #accu_val, predictions = classical_accuracy(features, val_labels)
                features = randomNetwork.get_features(randomNetwork.models[0], input=test_data)
                #predictions, centroids = randomNetwork.kmeans_regular(features)
                #accu_test, predictions = randomNetwork.clustering_accuracy_fast(predictions, test_labels)
                accu_test, predictions = randomNetwork.our_accuracy(features, test_labels)
                #show_feature_space(features, test_labels, predictions, accu_test, frame_count, n_categories, 2.0)
                confusion_m = confusion_matrix(test_labels, predictions)
                #accu_test, predictions = classical_accuracy(features, test_labels)
                accuracies[0].append(accu_train)
                accuracies[1].append(accu_val)
                accuracies[2].append(accu_test)
                iters.append(i)

                frame_count += 1
                #print "Iteration:", i, "Train accuracy:", accu_train, "Val accuracy:", accu_val, "Test accuracy:", accu_test
            except:
                print "Value error"
    #display_accuracy(accuracies, iters)
    #save_movie(movie_name = "Ours")
    return randomNetwork, accu_test

def classical_accuracy(features, labels):
    predictions = np.argmax(features, axis=1)
    return accuracy_score(labels, predictions), predictions

def show_feature_space(features, labels, predictions, accuracy, iteration, n_categories, max_size=1.0):
    #tsne = TSNE(n_components=2, random_state=0)
    #features = tsne.fit_transform(features)
    color_range = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'maroon', 'coral', 'olive']
    means = []
    for i in xrange(n_categories):
        indices = np.where(labels==i)[0]
        points_x = [features[idx, 0] for idx in indices]
        points_y = [features[idx, 1] for idx in indices]
        true_labels = [color_range[labels[idx]] for idx in indices]
        pred_labels = [color_range[predictions[idx]] for idx in indices]
        plt.scatter(points_x, points_y, s=7, c=true_labels, edgecolors=pred_labels, marker='s')
    for i in xrange(n_categories):
        indices = np.where(predictions == i)[0]
        current_mean = np.mean(np.asarray([features[idx, :] for idx in indices]), axis=0)
        means.append(current_mean)
    means = np.asarray(means)
    plt.scatter(means[:,0], means[:,1], s=25, c='k', marker='o')
    plt.xlim([-max_size,max_size])
    plt.ylim([-max_size, max_size])
    plt.title("Iteration "+str(iteration)+", accuracy = "+str(round(accuracy,3)))
    plt.savefig('./temp/iter_'+str(iteration).zfill(5)+'.png', dpi=300)
    #plt.show()
    plt.cla()

def save_movie(movie_name = "movie"):
    os.system("ffmpeg -r 5 -i ./temp/iter_%05d.png -vcodec mpeg4 -y ./"+movie_name+".mp4")
    shutil.rmtree('./temp/')
    os.makedirs('./temp/')


dset_class = Data(datset_name)
dset_class.load_dataset()
get_data(dset_class)