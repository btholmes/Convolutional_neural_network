import pickle
import numpy as np
from collections import defaultdict
from app.model.preprocessor import Preprocessor as img_prep
import random

class LiteOCR:
	def __init__(self, fn="app/model/alpha_weights.pkl", pool_size=2):
		# [weights, meta] = pickle.load(open(fn, 'rb'), encoding='latin1') #currently, this class MUST be initialized from a pickle file
		[weights, meta] = pickle.load(open(fn, 'rb'))

		# print "\n\n\n\n" 
		# print weights 
		# print "\n\n\n\n"

		# meta equals, just the vocab as metadata
		# {'vocab': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':)', ':(', 'triangle', '5star', 'scribble'], 'img_side': 20}

		# ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':)', ':(', 'triangle', '5star', 'scribble']
		self.vocab = meta["vocab"]

		#img_rows = 20 						
		self.img_rows = meta["img_side"] 
		# img_cols = 20
		self.img_cols = meta["img_side"]


		self.CNN = LiteCNN()
		# Weights is a large multidimensional list
		self.CNN.load_weights(weights)
		# pool size = 2
		self.CNN.pool_size=int(pool_size)

	def predict(self, image):
		# print(image.shape)
		X = np.reshape(image, (1, 1, self.img_rows, self.img_cols))
		X = X.astype("float32")

		predicted_i = self.CNN.predict(X)
		return self.vocab[predicted_i]

class LiteCNN:
	def __init__(self):
		self.layers = [] # a place to store the layers
		self.pool_size = None # size of pooling area for max pooling

	def load_weights(self, weights):
		assert not self.layers, "Weights can only be loaded once!"
		#  weights.keys() is	 
		# ['layer_9', 'layer_8', 'layer_1', 'layer_0', 'layer_3', 'layer_2', 'layer_5', 'layer_4', 'layer_7', 'layer_6', 'layer_11', 'layer_10']
		for k in range(len(weights.keys())):
			# self.layers just added all of the layer_ keys, so only param_1 and param_0 are left as keys
			# weightDict = weights['layer_{}'.format(k)]
			A= weights['layer_{}'.format(k)]
			# Below code converts all the weights to random numbers
			# for key in A: 
			# 	array = A[key]
			# 	for x in range(len(array)):
			# 		array[x] = random.uniform(-1,1)
			# 	A[key] = array

			# self.layers.append(weights['layer_{}'.format(k)])
			# self.layers.append(weightDict)
			self.layers.append(A)


		# print len(self.layers)




	def predict(self, X):
		assert not not self.layers, "Weights must be loaded before making a prediction!"
		h = self.cnn_layer(X, layer_i=0, border_mode="full") ; X = h
		h = self.relu_layer(X) ; X = h
		h = self.cnn_layer(X, layer_i=2, border_mode="valid") ; X = h
		h = self.relu_layer(X) ; X = h
		h = self.maxpooling_layer(X) ; X = h
		h = self.dropout_layer(X, .25) ; X = h
		h = self.flatten_layer(X) ; X = h
		h = self.dense_layer(X, layer_i=7) ; X = h
		h = self.relu_layer(X) ; X = h
		h = self.dropout_layer(X, .5) ; X = h
		h = self.dense_layer(X, layer_i=10) ; X = h
		h = self.softmax_layer2D(X) ; X = h
		max_i = self.classify(X)
		return max_i[0]

	def maxpooling_layer(self, convolved_features):
		nb_features = convolved_features.shape[0]
		nb_images = convolved_features.shape[1]
		conv_dim = convolved_features.shape[2]
		res_dim = int(conv_dim / self.pool_size)       #assumed square shape

		pooled_features = np.zeros((nb_features, nb_images, res_dim, res_dim))
		for image_i in range(nb_images):
			for feature_i in range(nb_features):
				for pool_row in range(res_dim):
					row_start = pool_row * self.pool_size
					row_end   = row_start + self.pool_size

					for pool_col in range(res_dim):
						col_start = pool_col * self.pool_size
						col_end   = col_start + self.pool_size

						patch = convolved_features[feature_i, image_i, row_start : row_end,col_start : col_end]
						pooled_features[feature_i, image_i, pool_row, pool_col] = np.max(patch)
		return pooled_features


	def poolMe(self, convolved_features):

		return 0


	def cnn_layer(self, X, layer_i=0, border_mode = "full"):
		# Features are the weights matrix in this case, although the true definition goes like this: 
		# Feature map is  filter/kernel applied to the previous layer.Feature map is not the filter/kernel/weight itself.
		features = self.layers[layer_i]["param_0"]

		bias = self.layers[layer_i]["param_1"]


		# patch_dim is 3 ->3, columns in features
		patch_dim = features[0].shape[-2]
		# print patch_dim 
		# nb_features is 32 ->32, number of feature maps
		nb_features = features.shape[0]
		# print nb_features



		# >>> c = np.array([    [      [ [1,2,3], [1,2,3], [3,2,2], [3,3,3], [3,3,3] ]     ]    ]  )
		# >>> c.shape
		# (1, 1, 5, 3)

		# >>> c = np.array([  [   [0,1,2],[1,2,3]  ]  ])
		# >>> c.shape
		# (1, 2, 3)

		# print X 
		# print "\n\n\n\n\n\n\n\n\n"

		# image_dim is 20 -> then 22, is a square
		image_dim = X.shape[2] #assume image square
		# print image_dim
		# image_channels is 1 -> 32 
		image_channels = X.shape[1]
		# print image_channels
		# nb_images is 1 -> 1. it is the number of images
		nb_images = X.shape[0]
		# print nb_images


		# is 22 -> 20
		if border_mode == "full":
			conv_dim = image_dim + patch_dim - 1
		elif border_mode == "valid":
			conv_dim = image_dim - patch_dim + 1

	
										# 1, 			32, 	22		, 	22
		# convolved_features = np.zeros((nb_images, nb_features, conv_dim, conv_dim));
		convolved_features = np.zeros((nb_images, nb_features, image_dim, image_dim));

		# print convolved_features

		# For each image, only 1
		for image_i in range(nb_images):
			# For each filter row in nb_features (32 of them) : Filter stays same for all convolved layers
			for feature_i in range(nb_features):
				# Start convolved image out with all 0's, of dimension 22,22 -> 20,20
				# convolved_image = np.zeros((conv_dim, conv_dim))
				convolved_image = np.zeros([image_dim, image_dim])

				# Channel represents the colors, there will be a [image_dim, image_dim] matrix for each color channel
				# image_channels is 1 for this
				for channel in range(image_channels):
					# feature is a 3x3 matrix
					# Eatch image channel has its own feature matrix
					feature = features[feature_i, channel, :, :]

					# Pull the image out of the 4D array that is X [[[[a,a,a],[b,b,b],[c,c,c]]]]
					image   = X[image_i, channel, :, :]
					# convolved_image += self.convolve2d(image, feature, border_mode);
					convolved_image += self.convolveMe(image, feature, 1)

				# Add the bias value to the image
				if feature_i < convolved_image.shape[0]: 
					convolved_image = convolved_image + bias[feature_i]

				convolved_features[image_i, feature_i, :, :] = convolved_image
		# print convolved_features
		# print "\n\n\n\n"
		return convolved_features

	def dense_layer(self, X, layer_i=0):
		W = self.layers[layer_i]["param_0"]
		b = self.layers[layer_i]["param_1"]
		# np.dot is actually matrix multiplication
		output = np.dot(X, W) + b
		return output


	# This is the convolve me function I wrote to mimic the convolve2d function
	def convolveMe(self, image, feature, stride): 
		# pad border with 0's all around the image
		padded_image = np.zeros([image.shape[0]+2, image.shape[1]+2])
		padded_image[1:padded_image.shape[0]-1, 1:padded_image.shape[1]-1] = image 

		conv_dim = padded_image.shape[0] - feature.shape[0] + 1


		convoluted_image = np.zeros([conv_dim, conv_dim])
		mid_of_feature = feature.shape[0]/2
		for row in range(mid_of_feature, padded_image.shape[0] - mid_of_feature, stride):
			for col in range( mid_of_feature, padded_image.shape[1] - mid_of_feature, stride):
				padded_image_section = padded_image[row-1:row-1+feature.shape[0], col-1:col-1+feature.shape[1]]
				convoluted_image[row-1][col-1] = np.sum(padded_image_section * feature)
		return convoluted_image

		# Below two lines speed up the process, and also produce more accurate results...
		# result = np.fft.fft2(image, [conv_dim, conv_dim]) * np.fft.fft2(feature, [conv_dim, conv_dim])
		# result = np.fft.ifft2(result).real 
		# return result




	# @staticmethod
	# def convolve2d(image, feature, border_mode="full"):
	# 	# [22,22]
	# 	image_dim = np.array(image.shape)
	# 	# [3,3]
	# 	feature_dim = np.array(feature.shape)
	# 	# [24,24]
	# 	target_dim = image_dim + feature_dim - 1

	# 	# # Dot products between image and feature
	# 	fft_result = np.fft.fft2(image, target_dim) * np.fft.fft2(feature, target_dim)
	# 	# print fft_result
	# 	target = np.fft.ifft2(fft_result).real
	# 	# print target

	# 	# for i in range(feature_dim.shape[0]):
	# 	# 	for j in range(feature_dim.shape[1]): 

	# 	if border_mode == "valid":
	# 		# To compute a valid shape, either np.all(x_shape >= y_shape) or
	# 		# np.all(y_shape >= x_shape).
	# 		valid_dim = image_dim - feature_dim + 1
	# 		# returns true if any dimensions are less than 1
	# 		if np.any(valid_dim < 1):
	# 			valid_dim = feature_dim - image_dim + 1
	# 		# brings start_i down to nearest whole number
	# 		# [2,2]
	# 		start_i = (target_dim - valid_dim) // 2
	# 		# [22,22]
	# 		end_i = start_i + valid_dim
	# 		# Target has 20 rows and 20 columns	
	# 		target = target[start_i[0]:end_i[0], start_i[1]:end_i[1]]
	# 	return target



	@staticmethod
	def shuffle(X, y):
		assert X.shape[0] == y.shape[0], "X and y first dimensions must match"
		p = np.random.permutation(X.shape[0])
		return X[p], y[p]

	@staticmethod
	def vectorize(y, vocab):
		nb_classes = len(vocab)
		Y = np.zeros((len(y), nb_classes))
		for i in range(len(y)):
			index = np.where(np.char.find(vocab, y[i]) > -1)[0][0]
			Y[i, index] = 1.
		return Y

	@staticmethod
	def trtest_split(X,y,fraction):
		boundary=int(X.shape[0]*fraction)
		return (X[:boundary], y[:boundary]), (X[boundary:], y[boundary:])

	@staticmethod
	def sigmoid(x):
		return 1.0/(1.0+np.exp(-x))

	@staticmethod
	def hard_sigmoid(x):
		slope = 0.2
		shift = 0.5
		x = (x * slope) + shift
		x = np.clip(x, 0, 1)
		return x

	@staticmethod
	def relu_layer(x):
		# Creates an array of 0's same dimensions as x
		z = np.zeros_like(x)
		# >>> A
		# array([[ 0,  1,  2,  3],
		#        [ 4,  5,  6,  7],
		#        [ 8,  9, 10, 11],
		#        [12, 13, 14, 15],
		#        [16, 17, 18, 19]])
		# >>> np.where(A<11)
		# (array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]), array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2]))
		# Where its greater than 0, keep it, else replace it with 0
		return np.where(x>=z,x,z)

	@staticmethod
	def softmax_layer2D(w):
		maxes = np.amax(w, axis=1)
		maxes = maxes.reshape(maxes.shape[0], 1)
		e = np.exp(w - maxes)
		dist = e / np.sum(e, axis=1, keepdims=True)
		return dist

	@staticmethod
	def repeat_vector(X, n):
		y = np.ones((X.shape[0], n, X.shape[2])) * X
		return y

	@staticmethod
	def dropout_layer(X, p):
		retain_prob = 1. - p
		X *= retain_prob
		return X

	@staticmethod
	def classify(X):
		return X.argmax(axis=-1)

	@staticmethod
	def flatten_layer(X):
		flatX = np.zeros((X.shape[0],np.prod(X.shape[1:])))
		for i in range(X.shape[0]):
			flatX[i,:] = X[i].flatten(order='C')
		return flatX
