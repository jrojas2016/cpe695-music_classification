'''
Neural Network Class:
	Description

Date: 4/16/2016

Author(s): 
	Jorge Rojas
'''
import math

class NeuralNet:
	'''
	Doc string
	'''
	def __init__(self, numInputs, layers, numOutputs):
		self.numInputs = numInputs
		self.numOutputs = numOutputs

	def train(self, samples, labels, num_epoch):
		for epoch in num_epoch:
			for sample in samples:
				netff()
				netbp()

	def test(self, data):
		pass

	def netff(self):
		'''
		Network Feed Forward:
		'''
		pass

	def netbp(self):
		'''
		Network Back Propagate
		'''
		pass

def sigmoid(x):
	return 1/(1 + math.exp(x))

def perceptron(x, threshold):
	return 1 if x > threshold else -1

def debug():
	print "Script to test Neural Net class"

if __name__ == '__main__':
	debug()

