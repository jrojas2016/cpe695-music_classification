'''
Neural Network Layer Class:
	Description

Date: 4/16/2016

Author(s):
	Jorge Rojas
'''
import random

class NeuralLayer:
	'''
	Doc String
	'''

	def __init__(self, numInputs, numNeurons, weights = None):
		self.num_inputs = numInputs
		self.num_neurons = numNeurons
		self.neuron_outputs = [ 0 for neuron in xrange(numNeurons) ]
		self.neuron_delta = [ 0 for neuron in xrange(numNeurons) ]

		if weights not None:
			assert len(weights) == self.numNeurons, 'The dimensions of the weights do not match with the number of neurons!'
			assert len(weights[0]) == self.numInputs, 'The dimensions of the weights do not match with the number of inputs!'
			self.weights = weights
		else:
			#Initialize weights at random if none provided
			# self.weights = [ [ random.uniform(-1, 1) ] * self.num_inputs ] * self.num_neurons
			#Which method is better?
			for i in xrange(self.num_neurons):
				neuron_weights = [ random.uniform(-1, 1) for j in xrange(self.num_inputs) ]
				self.weights.append(neuron_weights)

	def updateWeights(self):
		'''
		Author(s):
			Vinay
			Zhiyuan

		Update weights for layer inputs with neuron weights, learning rate, and sigma values
		'''
		pass

def debug():
	print "Script to test Neural Layer class"

if __name__ == '__main__':
	debug()

