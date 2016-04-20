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

	def __init__(self, numInputs = 9, layers = [10], numOutputs = 4, weights = None):
		'''
		Doc string
		'''

		''' Neurons '''
		self.num_inputs = numInputs
		self.num_outputs = numOutputs
		self.num_hidden_layers = len(layers)

		''' Layers '''
		self.layers = []
		self.input_values = [0] * self.num_inputs
		#Last hidden layer serves as input to output layer
		self.output_layer = nl.NeuralLayer(layers[-1], self.num_outputs)

		assert isinstance(layers, list), "layers argument is a list of the number of neurons per layer"
		for hidden_layer_id, hidden_layer in enumerate(layers):
			if hidden_layer_id == 0:
				self.layers.append( nl.NeuralLayer(self.num_inputs, hidden_layer) )
			else:
				self.layers.append( nl.NeuralLayer(layers[hidden_layer_id - 1].num_neurons, hidden_layer) )

	def printArchitecture():
		print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
		print "Number of Input Neurons: ", self.num_inputs
		print "Number of Hidden Layers: ", self.num_hidden_layers

		for layer_id, layer in enumerate(self.layers):
			print "     Number of neurons in hidden layer #%s: "%layer_id, layer.num_neurons

		print "Number of Output Neurons: ", self.num_outputs 
		print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

	def train(self, trainSamples, labels, learningRate, num_epoch = 10):
		for epoch in num_epoch:
			for sample_id, sample in enumerate(trainSamples):
				#The arguments for netff and netbp where arbitrarily placed
				#feel free to modify!
				self.netff(sample)
				self.netbp(labels[sample_id])
				self.updateWeights(learningRate)

	def test(self, testSamples, labels = None):
		num_correct_classifications = 0
		total_num_samples = len(testSamples)

		if labels is None:
			for sample in testSamples:
				classification = netff(sample)
				print classification
		else:
			#Calculate accuracy if labels present
			for sample_id, sample in enumerate(testSamples):
				classification = netff(sample)	#netff() needs to return network output
				if validateClassification(classification, labels[sample_id]):
					num_correct_classifications += 1
				print "Sample Label: ", labels[sample_id], " => Classification: ", classification

			print "Neural Network Accuracy % = ", ( num_correct_classifications/float(total_num_samples) )*100

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
            temp += hiddenLayer.weights[j][self.num_inputs]	#out of range?
            HiddenValueOfNeurons[j] = sigmoid(temp)

        for jj in range(0,self.num_outputs):
            temp=0
            for kk in range(0,hiddenLayer.num_neurons):
                temp+= HiddenValueOfNeurons[jj]* outputLayer.weights[jj][kk]
            temp += outputLayer.weights[jj][hiddenLayer.num_neurons]
            OutputValueOfNeurons[jj]= sigmoid(temp)
        
        # Array  OutputValueOfNeurons is the result of feed forword
    #Zhiyuan end
		pass	#No need to keep this here

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
		self.output_layer.updateWeights()

def sigmoid(x):
	return 1/(1 + math.exp(-x))

def perceptron(x, threshold):
	return 1 if x > threshold else -1

def validateClassification(classification, label):
	#Index with the greatest value would be the classification of the sample
	classification_index = classification.index( max(classification) )
	if label[classification_index] == 1:
		return True
	else:
		return False

def debug():
	print "Script to test Neural Net class"

if __name__ == '__main__':
	debug()

