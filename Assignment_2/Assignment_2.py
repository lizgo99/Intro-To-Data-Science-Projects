#
# This is a sample Notebook to demonstrate how to read "MNIST Dataset"
#
import numpy as np
import numpy as np
import struct
from array import array
from os.path import join
import os
import random
import matplotlib.pyplot as plt

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)  
		
		
#
# Verify Reading Dataset via MnistDataloader class
#
#
# Set file paths based on added MNIST Datasets
#
cwd = os.getcwd()
input_path = cwd + '/Assignment_2/MNIST'
training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')

#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1
    plt.show()
#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()


#
# Show some random training and test images 
#
images_2_show = []
titles_2_show = []
for i in range(0, 10):
    r = random.randint(1, 60000)
    images_2_show.append(x_train[r])
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))    

for i in range(0, 5):
    r = random.randint(1, 10000)
    images_2_show.append(x_train[r])        
    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))    

show_images(images_2_show, titles_2_show)

# #-----------------------------------------------------------------------------------------------------------------------------------------#

# # a
# np_divided_x_train = np.subtract(np.divide(np.array(x_train), 255), 0.5)

# #-----------------------------------------------------------------------------------------------------------------------------------------#

# # b.1
# X = np.reshape(np_divided_x_train, (60000, 784))
# Θ = np.cov(X.T)

# #-----------------------------------------------------------------------------------------------------------------------------------------#

# # b.2

# # Compute the eigendecomposition of Θ
# eigenvalues, eigenvectors = np.linalg.eig(Θ)

# # Sort the eigenvalues and eigenvectors in descending order
# idx = eigenvalues.argsort()[::-1]
# eigenvalues = eigenvalues[idx]
# eigenvectors = eigenvectors[:, idx]

# # Compute U, Σ, and UT
# U = eigenvectors
# Σ = np.diag(eigenvalues)
# UT = eigenvectors.T

# eigendecomposition = U @ Σ @ Σ @ UT

# singular_values = np.sqrt(np.diag(Σ)) #sqrt? 

# # Plot the singular values
# plt.plot(singular_values)
# plt.xlabel("Index")
# plt.ylabel("Singular value")
# plt.show()

# #-----------------------------------------------------------------------------------------------------------------------------------------#

# # b.3
# p = 40
# Up = U[:, :p]

# X_reduced = np.dot(X, Up)

# X_reconstructed = np.dot(X_reduced, Up.T)

# #-----------------------------------------------------------------------------------------------------------------------------------------#

# # b.4
# r = random.randint(1, 60000)

# # Visualize the original image
# plt.imshow(X[r].reshape(28,28), cmap='gray')
# plt.show()

# # Visualize the reconstructed image
# plt.imshow(X_reconstructed[r].reshape(28,28), cmap='gray')
# plt.show()

# #-----------------------------------------------------------------------------------------------------------------------------------------#

# # c
# def dist(x, y):
#     return np.sqrt(np.sum((x - y)**2))

# def kmeans(X, k, max_iter=100):
#     # Initialize centroids randomly
#     n_samples, n_features = X.shape
#     centroids = np.random.uniform(-0.5, 0.5, size=(k, n_features))
#     for _ in range(max_iter):
#         # Assign samples to closest centroids
#         labels = np.zeros(n_samples, dtype=int)
#         for i, x in enumerate(X):
#             distances = np.array([dist(x, c) for c in centroids])
#             labels[i] = np.argmin(distances)
        
#         # Update centroids
#         for i in range(k):
#             centroids[i] = np.mean(X[labels == i], axis=0)
    
#     return labels, centroids

# #-----------------------------------------------------------------------------------------------------------------------------------------#

# # d
# labels, centroids = kmeans(X_reconstructed, 10, 1) 
# # np.savetxt('labels.txt', labels, delimiter=',')

# #-----------------------------------------------------------------------------------------------------------------------------------------#

# # e
# def experiment(labels_array):
#   A = [[] for i in range(10)]
#   for i,l in enumerate(labels_array):
#       A[l].append(y_train[i])
  
#   cluster_labels = []   
#   for i in range(10):
#     cluster_labels.append(find_cluster_label(A[i]))
#   return cluster_labels

# def find_cluster_label(A):
#   M = np.zeros(10, dtype=int)
#   for l in A:
#     M[l] += 1;

#   max_index = 0
#   for i in range(len(M)):
#     if M[i] > M[max_index]:
#       max_index = i
#   return max_index

# print(experiment(labels))

# #-----------------------------------------------------------------------------------------------------------------------------------------#

# # f
# def find_success_percent(labels_array):
#   counter = 0
#   cluster_labels = experiment(labels_array)
#   for i in range(60000):
#     true_label = y_train[i]
#     cluster = labels_array[i]
#     cluster_label = cluster_labels[cluster]
#     if true_label == cluster_label:
#       counter += 1

#   percent = (counter/60000)*100
#   return percent

# print(find_success_percent(labels))

# #-----------------------------------------------------------------------------------------------------------------------------------------#
# # g
# labels1, centroids1 = kmeans(X_reconstructed, 10, 2) 
# print(find_success_percent(labels1))
# labels2, centroids2 = kmeans(X_reconstructed, 10, 2) 
# print(find_success_percent(labels2))
# labels3, centroids3 = kmeans(X_reconstructed, 10, 2) 
# print(find_success_percent(labels3))

# print("We can see that it's consistant around 40%")

# #-----------------------------------------------------------------------------------------------------------------------------------------#
# # h
# p12 = 12
# Up12 = U[:, :p12]

# X_reduced_to_12 = np.dot(X, Up12)

# X_reconstructed_from_12 = np.dot(X_reduced_to_12, Up12.T)

# labels12, centroids12 = kmeans(X_reconstructed_from_12, 10, 2) 

# print(find_success_percent(labels12))

# print("We can see that the result is a bit different")

# #-----------------------------------------------------------------------------------------------------------------------------------------#

# # i
# C = [[] for i in range(10)]

# for i, l in enumerate(np.array(y_train)):
#   if len(C[l]) < 10:
#     C[l].append(X[i])

# def kmeans_mean(X, k, max_iter=100):
#   # Initialize centroids by using the mean of 10 reduced images
#   n_samples, n_features = X.shape
#   centroids = []
#   for i in range(10):
#     mean = np.mean(C[i], axis=0)
#     centroids.append(mean)
#   for _ in range(max_iter):
#       # Assign samples to closest centroids
#       labels = np.zeros(n_samples, dtype=int)
#       for i, x in enumerate(X):
#           distances = np.array([dist(x, c) for c in centroids])
#           labels[i] = np.argmin(distances)
      
#       # Update centroids
#       for i in range(k):
#           centroids[i] = np.mean(X[labels == i], axis=0)
  
#   return labels, centroids

# mean_labels, mean_centroids = kmeans_mean(X_reconstructed, 10, 2) 

# print(find_success_percent(labels))
# print(find_success_percent(mean_labels))
# print("We can see that the result is better")

