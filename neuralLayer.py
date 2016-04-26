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

		''' Layer Architecture '''
		self.num_inputs = numInputs
		self.num_neurons = numNeurons
		self.neuron_outputs = [ 0 for neuron in xrange(numNeurons) ]

		''' Back Propagation Parameters '''
		self.neuron_delta = [ 0 for neuron in xrange(numNeurons) ]
		self.neuron_momentum = [ [0] * (self.num_inputs+1) ] * self.num_neurons
        
		if weights is not None:
			assert len(weights) == self.num_neurons, 'The dimensions of the weights do not match with the number of neurons!'
			assert len(weights[0]) == (self.num_inputs+1) , 'The dimensions of the weights do not match with the number of inputs!'
			self.weights = weights
		else:
			#Initialize weights at random if none provided
			#Plus one included to account for constant weight term
			self.weights = []
			for i in xrange(0,self.num_neurons):
				vector=[ random.uniform(-1, 1) ] * (self.num_inputs + 1)
				self.weights.append(vector) 

def debug():
	print "Script to test Neural Layer class"

if __name__ == '__main__':
	debug()

