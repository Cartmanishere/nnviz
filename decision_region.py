import numpy as np
# from keras.preprocessing import image
# from keras import layers
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.utils import to_categorical
import matplotlib.pyplot as plt
import math
import random
import pandas as pd
import time
# from keras.models import Model
import itertools
import time
from PIL import Image
import random
import matplotlib.markers as marker

class DecisionBoundary():

	def __init__(self, _range, resolution=80, workers=5):
		'''
		
		'''
		self.workers = 5
		self.scatter_alpha = 0.8
		self.colormap = [self.__get_random_color() for i in range(10)]
		self.markers = list(marker.MarkerStyle.markers.keys())[:-4]
		self.max_range = _range
		self.resolution = resolution
		self._increment = 2 * self.max_range / self.resolution
		# X-coordinates
		self.Xs = np.arange(-self.max_range, self.max_range, self._increment)
		# Y-coordinates
		self.Ys = np.arange(self.max_range, -self.max_range, -self._increment)
		
		self._preds = np.zeros((resolution, resolution, 3))
 		


	#######################
	## Utility Functions ##
	#######################

	def __get_random_color(self):
		r = lambda: random.randint(0,255)
		return np.array([r(), r(), r()])

	def __get_classifier(self, model, layer, neuron):
		assert model.input.get_shape()[1] == 2

		if layer is None:
			layer = len(model.layers)-1
			classifier = model
		else:
			assert layer < len(model.layers)-1
			output = model.layers[layer].output
			classifier = Model(inputs=model.input, outputs=output)


		assert neuron < model.layers[layer].output.get_shape()[1] or neuron is None
		neuron_index = neuron

		return classifier, layer, neuron_index


	def __point_gen(self):
		for i in itertools.product(self.Xs, self.Ys):
			yield np.array([i])


	def _map(self, x, y):
		X = (x / (self._increment) + self.resolution / 2)
		Y = (- y / (self._increment) + self.resolution / 2)
		return X, Y


	def __validate_weights(self, weights, biases):
		cur_dim = 2
		for weight_matrix, layer_bias in zip(weights, biases):
			assert len(weight_matrix) == len(biases)
			assert weight_matrix.T.shape[-1] == cur_dim
			cur_dim = weight_matrix.shape[-1]


	##########################
	## Secondary Functions ##
	##########################

	def draw_keras_binary(self, model, layer=None, neuron=0):
		'''
		This function is to be used for visualizing decision boundary by keras models.

		Args:
			model: the keras model for which you want to draw the boundary.
			layer: (optional) the layer you want to visualize (usually used in conjunction with the neuron argument)
			neuron: (optional) the neuron from the specified layer 

		Returns:
			An image matrix of shape self.resolution

		Raises:
			AssertionError: if any invalid input is provided
		'''


		classifier, layer, neuron_index = self.__get_classifier(model, layer, neuron)

		img = np.zeros((self.resolution, self.resolution, 3))
		start_time = time.time()
		
		_vals = classifier.predict_generator(self.__point_gen(), steps=self.resolution**2, workers=self.workers, verbose=0).reshape((self.resolution, self.resolution))
		for i in range(self.resolution):
			for j in range(self.resolution):
				_ = _vals[i, j]
				if _ > 0.5:
					img[i, j, 0] = _
				elif _ < 0.5:
					img[i, j, 2] = _
				else:
					img[i, j, :] = 0

		img = np.transpose(img, (1, 0, 2))
		img = Image.fromarray((img * 255).astype('uint8'), 'RGB')
		self.last_img = img
		plt.imshow(img)

		return img


	def draw_keras_categorical(self, model, colormap=None):
		'''

		'''
		num_classes = int(model.layers[-1].output.get_shape()[1])
		if colormap is None:
			if len(self.colormap) < num_classes:
				colormap = self.colormap + [self.__get_random_color() for i in range(num_classes - len(self.colormap))]
			else:
				colormap = self.colormap
		else:
			assert len(colormap) == num_classes
			self.colormap = colormap

		classifier, _, _ = self.__get_classifier(model, None, None)

		img = np.zeros((self.resolution, self.resolution, 4))

		_vals = classifier.predict_generator(self.__point_gen(), steps=self.resolution**2, workers=self.workers, verbose=0).reshape((self.resolution, self.resolution, num_classes))
		for i in range(self.resolution):
			for j in range(self.resolution):
				_ = _vals[i, j]
				index = np.argmax(_)
				img[i, j] = np.append(colormap[index], np.max(_))

		img = np.transpose(img, (1, 0, 2))
		img = Image.fromarray((img * 255).astype('uint8'), mode='RGBA')
		self.last_img = img
		plt.imshow(img)

		return img


	def draw(self, predict, multi_threaded=False, **kwargs):
		'''
		A general purpose function that will draw the boundary for any classifier

		Args:
			predict: a classification function which takes in two inputs [x1, x2] and gives one output strictly in the range
			[0-1]. All the logic behind this function is to be handled by the user.
			multithreaded: Whether to use multithreading. Using this with any of tensorflow models will give error. 
			so for tensorflow models put this as False.
			**kwaargs: any parameters you want to pass to your predict function

		Returns:
			An image matrix of shape self.resolution

		Raises:
			AssertionError: if any invalid input is provided
		'''

		_predict_fn = predict
		img = np.zeros((self.resolution, self.resolution, 3))

		if multi_threaded:
			# Multithreaded computation
			_vals = np.zeros((self.resolution**2,))
			# TODO: Implmenet multi-threaded inference
		else:
			for i in range(self.resolution):
				for j in range(self.resolution):
					_ = _predict_fn(self.Xs[i], self.Ys[j], **kwargs)
					if _ > 0.5:
						img[i, j, 0] = _
					elif _ < 0.5:
						img[i, j, 2] = _
					else:
						img[i, j, :] = 0


		img = np.transpose(img, (1, 0, 2))
		img = Image.fromarray((img * 255).astype('uint8'), 'RGB')
		self.last_img = img
		plt.imshow(img)

		return img


	def draw_categorical(self, predict, num_classes, multi_threaded=False, colormap=None, **kwargs):
		'''
		A general purpose function that will draw the boundary for any multi-class classifier

		Args:
			predict: a classification function which takes in two inputs [x1, x2] and gives output vector that is
			one-hot encoding (ndarray) of the class value.
			multithreaded: Whether to use multithreading. Using this with any of tensorflow models will give error. 
			so for tensorflow models put this as False.
			**kwaargs: any parameters you want to pass to your predict function

		Returns:
			An image matrix of shape self.resolution

		Raises:
			AssertionError: if any invalid input is provided
		'''

		if colormap is None:
			if len(self.colormap) < num_classes:
				colormap = self.colormap + [self.__get_random_color() for i in range(num_classes - len(self.colormap))]
			else:
				colormap = self.colormap

		else:
			assert len(colormap) == num_classes
			self.colormap = colormap


		_predict_fn = predict
		img = np.zeros((self.resolution, self.resolution, 4))

		if multi_threaded:
			# Multithreaded computation
			_vals = np.zeros((self.resolution**2,))
			# TODO: Implmenet multi-threaded inference
		else:
			for i in range(self.resolution):
				for j in range(self.resolution):
					_ = _predict_fn(self.Xs[i], self.Ys[j], **kwargs)
					index = np.argmax(_)
					img[i, j] = np.append(colormap[index], np.max(_))


		img = np.transpose(img, (1, 0, 2))
		img = Image.fromarray((img * 255).astype('uint8'), mode='RGBA')
		self.last_img = img
		plt.imshow(img)

		return img


	def draw_from_weights(self, weights, biases, activations):
		'''
		A function that will draw the boundary for a neural network based classifer constructed using the provided weights.

		Args:
			weights: an array of weight matrices for a neural network

			NOTE: ith row in the weight matrix of any layer corresponds to the weights of
			ith neuron in that layer.

			NOTE: output is calculated as-
			dot(Wn.T, ...(dot(W3.T, dot(W2.T, dot(W1.T, X)))))
			Dimensions of all the weight matrices is strictly validated before computation

			biases: an array of biases. Can be a list of biases for multi-layer model

			acitvations: activation function or list of activation functions for multi-layer models

			NOTE: pass references to actual activation functions which can are used as follows:
			activation_fn(ndarray_vector) =. return nd_array of activated outputs


		Returns:
			An image matrix of shape self.resolution

		Raises:
			AssertionError: if any invalid input is provided
		'''

		if type(weights) == type([]):
			assert len(weights) == len(biases)
			if type(activations) != type([]):
				activation_fns = [ activations for i in range(len(weights))]
			else:
				assert len(weights) == len(activations)


		self.__validate_weights(weights, biases)

		# Detect binary class or multi-class
		output_dim = weights[-1].T.shape[-1]

		# pseudo model
		def predict(x1, x2, weights=None, biases=None, activations=None):
			cur_output = np.array([x1, x2])
			for weight_matrix, bias, activation_fn in zip(weights, biases, activations):
				# calculating one layer at a time
				cur_output = activation_fn(np.dot(cur_output, weight_matrix.T) + bias)

			return cur_output

		if output_dim > 1:
			pass





	def plot_with_data(self, patterns, targets, alpha=0.7):
		'''
		A general purpose function that will draw the boundary for any multi-class classifier

		Args:
			patterns: an ndarray of shape (?, 2) which contains the data points
			targets: an ndarray containing target values for data points. 
			This can be a single value or one-hot encoded vector only.

		Returns:
			An image will be plotted but nothing is returned

		Raises:
			AssertionError: if any invalid input is provided
		'''

		assert len(patterns) == len(targets)
	
		if not (targets[0].shape == ()): ## TODO:: Change this condition to something more robust
			# Multi-class classifier

			colormap = np.array(self.colormap)
			plt.imshow(self.last_img, alpha=alpha)
			for i in range(len(patterns)):
				x, y = patterns[i]
				x, y = self._map(x, y)
				index = np.argmax(targets[i])
				plt.scatter(x, y, marker=self.markers[index], c=tuple(self.colormap[index] / 255), alpha=self.scatter_alpha)

		else:
			# Binary
			plt.imshow(self.last_img, alpha=alpha)
			for i in range(len(patterns)):
				x, y = patterns[i]
				x, y = self._map(x, y)
				_ = targets[i]
				if _ > 0.5:
					plt.scatter(x, y, marker='x', c='r', alpha=self.scatter_alpha)
				elif _ <= 0.5:
					plt.scatter(x, y, marker='o', c='b', alpha=self.scatter_alpha)


	####################
	## Main Functions ##
	####################


	def draw_keras(self, model, layer=None, neuron=0, colormap=None):
		'''
		This function is to be used for visualizing decision boundary by keras models.

		Args:
			model: the keras model for which you want to draw the boundary.
			layer: (optional) the layer you want to visualize (usually used in conjunction with the neuron argument)
			neuron: (optional) the neuron from the specified layer 
			colormap: (optional) in case of multiclass classifier

		Returns:
			An image matrix of shape self.resolution

		Raises:
			AssertionError: if any invalid input is provided
		'''

		# Detect output of model and call appropriate method
		output_dim = int(model.layers[-1].output.get_shape()[1])
		if output_dim > 1:
			return self.draw_keras_categorical(model, colormap)
		else:
			return self.draw_keras_binary(model, layer, neuron)



	###############
	## TEMP FUNC ##
	###############

	def draw_rdpta(self, predict, num_classes, multi_threaded=False, colormap=None, **kwargs):
		'''
		A general purpose function that will draw the boundary for any multi-class classifier

		Args:
			predict: a classification function which takes in two inputs [x1, x2] and gives output vector that is
			one-hot encoding (ndarray) of the class value.
			multithreaded: Whether to use multithreading. Using this with any of tensorflow models will give error. 
			so for tensorflow models put this as False.
			**kwaargs: any parameters you want to pass to your predict function

		Returns:
			An image matrix of shape self.resolution

		Raises:
			AssertionError: if any invalid input is provided
		'''

		if colormap is None:
			if len(self.colormap) < num_classes:
				colormap = self.colormap + [self.__get_random_color() for i in range(num_classes - len(self.colormap))]
			else:
				colormap = self.colormap

		else:
			assert len(colormap) == num_classes
			self.colormap = colormap


		_predict_fn = predict
		img = np.zeros((self.resolution, self.resolution, 4))

		if multi_threaded:
			# Multithreaded computation
			_vals = np.zeros((self.resolution**2,))
			# TODO: Implmenet multi-threaded inference
		else:
			for i in range(self.resolution):
				for j in range(self.resolution):
					_ = _predict_fn(self.Xs[i], self.Ys[j], **kwargs)
					if np.where(_ > 0)[0].shape[-1] == 1:
						index = np.argmax(_)
						img[i, j] = np.append(colormap[index], np.max(_))
					else:
						img[i, j] = [0., 0., 0., 1.]

		img = np.transpose(img, (1, 0, 2))
		img = Image.fromarray((img * 255).astype('uint8'), mode='RGBA')
		self.last_img = img
		plt.imshow(img)

		return img
