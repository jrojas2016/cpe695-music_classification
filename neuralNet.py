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

	def __init__(self, numInputs = 9, layers = [10], numOutputs = 4, weights = None, learningRate, momentum):
		'''
		Doc string
		'''

		''' Learning Parameters '''
		self.learningRate = learningRate
		self.momentum = momentum

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
				self.layers.append( nl.NeuralLayer(self.num_inputs, hidden_layer, learningRate = self.learningRate, momentum = self.momentum) )
			else:
				self.layers.append( nl.NeuralLayer(layers[hidden_layer_id - 1].num_neurons, hidden_layer, learningRate = self.learningRate, momentum = self.momentum) )

	def printArchitecture():
		print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
		print "Number of Input Neurons: ", self.num_inputs
		print "Number of Hidden Layers: ", self.num_hidden_layers

		for layer_id, layer in enumerate(self.layers):
			print "     Number of neurons in hidden layer #%s: "%layer_id, layer.num_neurons

		print "Number of Output Neurons: ", self.num_outputs 
		print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

	def train(self, trainSamples, labels, num_epoch = 10):
		for epoch in num_epoch:
			for sample_id, sample in enumerate(trainSamples):
				#The arguments for netff and netbp where arbitrarily placed
				#feel free to modify!
				self.netff(sample)
				self.netbp(labels[sample_id])
				self.updateWeights()

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
		#assume only one hidden layer first
		for i in range(0,inputLayer.num_neurons):
			inputLayer.neuron_sigmas[i]= trainSample[i]
		for j in range(0,hiddenLayer.num_neurons):
			temp=0
			for k in range(0,inputLayer.num_neurons):
				temp+= inputLayer.neuron_sigmas[k] * hiddenLayer.weights[j][k]
			#weights[j]'s length should be num_inputs+1

			temp += hiddenLayer.weights[j][self.num_inputs] # I add one more when initialize the weight
			hiddenLayer.neuron_sigmas[j] = sigmoid(temp)


		for jj in range(0,outputLayer.num_neurons):
			temp=0
			for kk in range(0,hiddenLayer.num_neurons):
				temp+= hiddenLayer.neuron_sigmas[jj]* outputLayer.weights[jj][kk]
			temp += outputLayer.weights[jj][hiddenLayer.num_neurons]
			outputLayer.neuron_sigmas[jj]= sigmoid(temp)
		
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
	#Zhiyuan add
		for i in (0,self.num_outputs):
			outputLayer.neuron_delta[i]= outputLayer.neuron_sigmas[i] *(1- outputLayer.neuron_sigmas[i])* (trainLabel[i]-outputLayer.neuron_sigmas[i])
		for j in range(0,hiddenLayer.num_neurons):
			temp=0
			for k in range(0,outputLayer.num_neurons):
				temp+= outputLayer.weights[k][j] * outputLayer.neuron_delta[k]
			hiddenLayer.neuron_delta[j]=  hiddenLayer.neuron_sigmas[j] * (1 -  hiddenLayer.neuron_sigmas[j]) * temp
		updateWeights()
	#Zhiyuan end
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
	#Zhiyuan add
	dataDir='/xxx/xxxx/xxx/'
	train_data=dataDir + 'train.csv'
	test_data=dataDir+ 'test.csv'
	#output_file= dataDir + 'result.csv'
	fTest = open (test_data,'r')
	fTrain = open(train_data,'r')
	#fOut = open(output_file, 'w')
	#no idea of the structure of data
	for line in fTrain:
		Traindata=....
		Trainlabels=....
	for line in fTest:
		#no idea of the structure of data
		Testdata=....
		#TestLabels=...
	nn = NeuralNet(9,10,4)
	nn.train(Traindata,Trainlabels,10)
	nn.test(Testdata)

	fTest.close()
	fTrain.close()
	#fOut.close()
	#Zhiyuan end

if __name__ == '__main__':
	debug()

