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

	def printArchitecture(self):
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
		for i in range(0,self.num_inputs):
			self.input_values[i]= Traindata[i]
        for hidden_layer_id, hidden_layer in enumerate(layers):
            if hidden_layer_id == 0:
                for j in range(0,hidden_layer.num_neurons):
                    temp=0
                    for k in range(0,self.num_inputs):
                        temp+= self.input_values[k] * hidden_layer.weights[j][k]
			#weights[j]'s length should be num_inputs+1
                    temp += hidden_layer.weights[j][self.num_inputs] # I add one more when initialize the weight
                    hidden_layer.neuron_outputs[j] = sigmoid(temp)
            else:
                for j in range(0,hidden_layer.num_neurons):
                    temp=0
                    for k in range(0,layers[hidden_layer_id -1].num_neurons):
                        temp+= layers[hidden_layer_id -1].neuron_outputs[k]* hidden_layer.weights[j][k]
                    temp += hidden_layer.weights[j][hidden_layer.num_neurons]
                    hidden_layer.neuron_outputs[j]= sigmoid(temp)
        for j in range(0,self.num_outputs):
            temp=0
            for k in range(0,layers[len(layers)-1].num_neurons):
                temp+= layers[len(layers)-1].neuron_outputs[k]* output_layer.weights[j][k]
            temp += output_layer.weights[j][layers[len(layers)-1].num_neurons]
            output_layer.neuron_outputs[j]= sigmoid(temp)
		
		# Array  OutputValueOfNeurons is the result of feed forword
	#Zhiyuan end


	def netbp(self, trainLabel):
		'''
		Author(s):
			Vinay
			Zhiyuan

		Network Back Propagate
		'''
	#Zhiyuan add
		for i in (0,self.num_outputs):
			output_layer.neuron_delta[i]= output_layer.neuron_outputs[i] *(1- output_layer.neuron_outputs[i])* (trainLabel[i]-output_layer.neuron_outputs[i])
        for j in range(0,layers[len(layers)-1].num_neurons):
            temp=0
            for k in range(0,output_layer.num_neurons):
                temp+= output_layer.weights[k][j] * output_layer.neuron_delta[k]
            layers[len(layers)-1].neuron_delta[j]=  layers[len(layers)-1].neuron_outputs[j] * (1 -  layers[len(layers)-1].neuron_outputs[j]) * temp

        for i in xrange(len(layers)-1, 0):
            for j in range(0,layers[i].num_neurons):
                temp=0
                for k in range(0,layers[i+1].num_neurons):
                    temp+= layers[i+1].weights[k][j] * layers[i+1].neuron_delta[k]
                layers[i].neuron_delta[j]=  layers[i].neuron_outputs[j] * (1 -  layers[i].neuron_outputs[j]) * temp
	#Zhiyuan end

	def updateWeights(self):
		'''
		Update Weights for all layers of network
		'''
        for hidden_layer_id, hidden_layer in enumerate(layers):
            if hidden_layer_id == 0:
                for i in range(0,hidden_layer.num_neurons):
                    for j in range(0,self.num_inputs):
                        change=hidden_layer.learningRate * hidden_layer.neuron_delta[i] * self.input_values[j]
                        hidden_layer.weights[i][j]+= change+ hidden_layer.momentum * hidden_layer.neuron_momentum[i][j]
                        hidden_layer.neuron_momentum[i][j]= change
            else:
                for i in range(0,self.num_neurons):
                    for j in range(0,self.num_inputs):
                        change=hidden_layer.learningRate * hidden_layer.neuron_delta[i] * layers[hidden_layer_id -1].neuron_outputs[j]
                        hidden_layer.weights[i][j]+= change+ hidden_layer.momentum * hidden_layer.neuron_momentum[i][j]
                        hidden_layer.neuron_momentum[i][j]= change
        for i in range(0,self.num_outputs):
            for j in range(0,layers[len(layers)-1].num_neurons):
                change=self.learningRate * self.output_layer.neuron_delta[i] * layers[len(layers)-1].neuron_outputs[j]
                self.output_layer.weights[i][j]+= change+ self.momentum * self.output_layer.neuron_momentum[i][j]
                self.output_layer.neuron_momentum[i][j]= change

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

