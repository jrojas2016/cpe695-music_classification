'''
Neural Network Class:
	Description

Date: 4/16/2016

Author(s): 
	Jorge Rojas
'''
import math
import neuralLayer as nl

class NeuralNet:
	'''
	Doc string
	'''

	def __init__(self, numInputs, layers = [10], numOutputs):
		self.layers = []
		self.num_inputs = numInputs
		self.num_outputs = numOutputs
		self.num_layers = len(layers)

		assert isinstance(layers, list), "layers argument is a list of the number of neurons per layer"
		for layer_id, layer in enumerate(layers):
			if layer_id == 0:
				self.layers.append( nl.NeuralLayer(self.num_inputs, layer) )
			else:
				self.layers.append( nl.NeuralLayer(layers[layer_id - 1], layer) )

	def train(self, samples, labels, num_epoch = 10):
		for epoch in num_epoch:
			for sample_id, sample in enumerate(samples):
				#The arguments for netff and netbp where arbitrarily placed
				#feel free to modify!
				self.netff(sample)
				self.netbp(labels[sample_id])

	def test(self, data, labels = None):
		pass

	def netff(self, trainSample):
		'''
		Author(s):
			Vinay
			Zhiyuan

		Network Feed Forward:
		'''
    #Zhiyuan add
        InputvalueOfNeurons = [0] * self.num_inputs
        HiddenValueOfNeurons = [0] *  hiddenLayer.num_neurons
        OutputValueOfNeurons = [0] * self.num_outputs
        #assume only one hidden layer first
        for i in range(0,self.num_inputs):
            InputValueOfNeurons[i]= trainSample[i]
        for j in range(0,self.num_neurons):
            temp=0
            for k in range(0,self.num_inputs):
                temp+= InputValueOfNeurons[k] * hiddenLayer.weights[j][k]
            #weights[j]'s length should be num_inputs+1
            temp += hiddenLayer.weights[j][self.num_inputs]
            HiddenValueOfNeurons[j] = sigmoid(temp)

        for jj in range(0,self.num_outputs):
            temp=0
            for kk in range(0,hiddenLayer.num_neurons):
                temp+= HiddenValueOfNeurons[jj]* outputLayer.weights[jj][kk]
            temp += outputLayer.weights[jj][hiddenLayer.num_neurons]
            OutputValueOfNeurons[jj]= sigmoid(temp)
        
        # Array  OutputValueOfNeurons is the result of feed forword
    #Zhiyuan end
		pass

	def netbp(self, trainLabel):
		'''
		Author(s):
			Vinay
			Zhiyuan

		Network Back Propagate
		'''
		pass

	def updateWeights(self):
		'''
		Update Weights for all layers of network
		'''
		for layer in self.layers:
			layer.updateWeights()

def sigmoid(x):
	return 1/(1 + math.exp(-x))

def perceptron(x, threshold):
	return 1 if x > threshold else -1

def debug():
	print "Script to test Neural Net class"

if __name__ == '__main__':
	debug()

