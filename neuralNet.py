'''
Neural Network Class:
	Description

Date: 4/16/2016

Author(s): 
	Jorge Rojas
	Zhiyuan Chen
'''
import math
import xlrd
import random
import neuralLayer as nl

class NeuralNet:
	'''
	Doc string
	'''

	def __init__(self, numInputs = 9, layers = [10], numOutputs = 4, weights = None, learningRate = 0.1, momentum = 0.01):
		'''
		Doc string
		'''

		''' Learning Parameters '''
		self.momentum = momentum
		self.learningRate = learningRate

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

	def printArchitecture(self):
		print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
		print "Number of Input Neurons: ", self.num_inputs
		print "Number of Hidden Layers: ", self.num_hidden_layers

		for layer_id, layer in enumerate(self.layers):
			print "     Number of neurons in hidden layer #%s: "%layer_id, layer.num_neurons

		print "Number of Output Neurons: ", self.num_outputs 
		print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

	def train(self, trainSamples, labels, num_epoch = 10):
		for epoch in xrange(0, num_epoch):
			error=0
			for sample_id, sample in enumerate(trainSamples):
				outV=self.netff(sample)
				# print outV
				for i in xrange(0,len(outV)):
					error+= 0.5*(outV[i]-labels[sample_id][i])*(outV[i]-labels[sample_id][i])
				self.netbp(labels[sample_id])
				self.updateWeights()
			print "error is %f when epoch is %d" %(error,epoch+1) 

	def test(self, testSamples, labels = None):
		num_correct_classifications = 0
		total_num_samples = len(testSamples)

		if labels is None:
			for sample in testSamples:
				classification = self.netff(sample)
				print classification
		else:
			#Calculate accuracy if labels present
			for sample_id, sample in enumerate(testSamples):
				classification = self.netff(sample)	#netff() needs to return network output
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
		self.input_values = trainSample
		for hidden_layer_id, hidden_layer in enumerate(self.layers):
			if hidden_layer_id == 0:
				for j in range(0,hidden_layer.num_neurons):
					temp=0
					for k in range(0,self.num_inputs):
						try:
							temp += self.input_values[k] * hidden_layer.weights[j][k]
						except TypeError:
							#Spotify does not have the appropriate feature. Assume it's midway between 0 and 1
							temp += 0.5 * hidden_layer.weights[j][k]
					#weights[j]'s length should be num_inputs+1
					temp += hidden_layer.weights[j][self.num_inputs] # I add one more when initialize the weight
					# print temp
					hidden_layer.neuron_outputs[j] = sigmoid(temp)
			else:
				for j in range(0,hidden_layer.num_neurons):
					temp=0
					for k in range(0,self.layers[hidden_layer_id -1].num_neurons):
						temp+= self.layers[hidden_layer_id -1].neuron_outputs[k]* hidden_layer.weights[j][k]
					temp += hidden_layer.weights[j][hidden_layer.num_neurons]
					hidden_layer.neuron_outputs[j]= sigmoid(temp)
		for j in range(0,self.num_outputs):
			temp=0
			for k in range(0,self.layers[-1].num_neurons):
				temp+= self.layers[-1].neuron_outputs[k]* self.output_layer.weights[j][k]
			temp += self.output_layer.weights[j][self.layers[-1].num_neurons]
			self.output_layer.neuron_outputs[j]= sigmoid(temp)
		return self.output_layer.neuron_outputs
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
		for i in range(0,self.num_outputs):
			self.output_layer.neuron_delta[i]= self.output_layer.neuron_outputs[i] *(1- self.output_layer.neuron_outputs[i])* (trainLabel[i]- self.output_layer.neuron_outputs[i])
		# print "output layer neuron delta"
		# print self.output_layer.neuron_delta
		for j in range(0,self.layers[-1].num_neurons):
			temp=0
			for k in range(0,self.output_layer.num_neurons):
				temp+= self.output_layer.weights[k][j] * self.output_layer.neuron_delta[k]
			self.layers[-1].neuron_delta[j]=  self.layers[-1].neuron_outputs[j] * (1 -  self.layers[-1].neuron_outputs[j]) * temp
		# print "hidden layer neuron delta"
		# print self.layers[-1].neuron_delta
		for i in xrange(0,self.num_hidden_layers -1):
			for j in range(0,self.layers[self.num_hidden_layers - 2 - i].num_neurons):
				temp=0
				for k in range(0,self.layers[self.num_hidden_layers - 1 - i].num_neurons):
					temp+= self.layers[self.num_hidden_layers - 1 - i].weights[k][j] * self.layers[self.num_hidden_layers - 1 - i].neuron_delta[k]
				self.layers[self.num_hidden_layers - 2 - i].neuron_delta[j]=  self.layers[self.num_hidden_layers - 2 - i].neuron_outputs[j] * (1 -  self.layers[self.num_hidden_layers - 2 - i].neuron_outputs[j]) * temp
	#Zhiyuan end

	def updateWeights(self):
		'''
		Update Weights for all layers of network
		'''
		for hidden_layer_id, hidden_layer in enumerate(self.layers):
			if hidden_layer_id == 0:
				for i in range(0,hidden_layer.num_neurons):
					for j in range(0,self.num_inputs):
						try:
							change=self.learningRate * hidden_layer.neuron_delta[i] * self.input_values[j]
						except TypeError:
							#Spotify does not have the appropriate feature. Assume it's midway between 0 and 1
							change=self.learningRate * hidden_layer.neuron_delta[i] * 0.5
						hidden_layer.weights[i][j]+= change+ self.momentum * hidden_layer.neuron_momentum[i][j]
						hidden_layer.neuron_momentum[i][j]= change
					hidden_layer.weights[i][hidden_layer.num_inputs] += self.learningRate * hidden_layer.neuron_delta[i] + self.momentum * hidden_layer.neuron_momentum[i][hidden_layer.num_inputs]
			else:
				for i in range(0,hidden_layer.num_neurons):
					for j in range(0,hidden_layer.num_inputs):
						change=self.learningRate * hidden_layer.neuron_delta[i] * self.layers[hidden_layer_id -1].neuron_outputs[j]
						hidden_layer.weights[i][j]+= change+ self.momentum * hidden_layer.neuron_momentum[i][j]
						hidden_layer.neuron_momentum[i][j]= change
					hidden_layer.weights[i][hidden_layer.num_inputs] += self.learningRate * hidden_layer.neuron_delta[i] + self.momentum * hidden_layer.neuron_momentum[i][hidden_layer.num_inputs]
		for i in range(0,self.num_outputs):
			for j in range(0,self.layers[-1].num_neurons):
				change=self.learningRate * self.output_layer.neuron_delta[i] * self.layers[-1].neuron_outputs[j]
				self.output_layer.weights[i][j]+= change+ self.momentum * self.output_layer.neuron_momentum[i][j]
				self.output_layer.neuron_momentum[i][j]= change
			self.output_layer.weights[i][self.output_layer.num_inputs] += self.learningRate * self.output_layer.neuron_delta[i] + self.momentum * self.output_layer.neuron_momentum[i][self.output_layer.num_inputs]

def sigmoid(x):
	return 1/(1 + math.exp(-x))

def perceptron(x, threshold):
	return 1 if x > threshold else -1

def validateClassification(classification, label):
	#Index with the greatest value would be the classification of the sample
	# print "classification: ", classification
	classification_index = classification.index( max(classification) )
	if label[classification_index] == 1:
		return True
	else:
		return False

def debug():
	#Jorge Test
#	 trainData = []
#	 labels = []
#	 for sample_num in xrange(0, 10):
#	 	feature_vector = []
#	 	for feature_num in xrange(0, 9):
#	 		feature_vector.append( random.uniform(0, 1) )
#	 	trainData.append( feature_vector )
#	 	label = [0, 0, 0, 0]
#	 	label_index = random.randint(0,3)
#	 	label[label_index] = 1
#	 	labels.append( label )
#	 # print trainData 	#DEBUGGING
#	 # print labels	#DEBUGGING
#
#	 nn = NeuralNet(learningRate = 0.9, momentum = 0.1)
#	 nn.printArchitecture()
#	 nn.train(trainData, labels, 100)
#
#	 nn.test(trainData, labels)

	#Zhiyuan Test
	print "Script to test Neural Net class"
	#Zhiyuan add
	dataDir='/Users/zhiyuanchen/Documents/git/'
	train_data=dataDir + 'TrainData.xls'
	train_label=dataDir +'TrainLabel.xls'
	trainD = xlrd.open_workbook(train_data)
	try:
		trainData = trainD.sheet_by_name("Sheet1")
	except:
		print "no sheet in %s named Sheet1" % fname
	nrows = trainData.nrows
	ncols = trainData.ncols
	traindata_list = []
	print "nrows %d, ncols %d" % (nrows,ncols)
	for i in range(0,nrows):
		row_data = trainData.row_values(i)
		traindata_list.append(row_data)
	trainT = xlrd.open_workbook(train_label)
	try:
		trainLabels = trainT.sheet_by_name("Sheet1")
	except:
		print "no sheet in %s named Sheet1" % fname
	nrows = trainLabels.nrows
	ncols = trainLabels.ncols
	trainLabel_list = []
	# print "nrows %d, ncols %d" % (nrows,ncols)
	for i in range(0,nrows):
		tlabel=[0]*9
		row_data = trainLabels.row_values(i)
		label_index = int(row_data[0])
		tlabel[label_index-1]=1
		trainLabel_list.append(tlabel)
	# print "labellist is %d" %len(trainLabel_list )
	# print "trainlist is %d" %len(traindata_list )
	nn = NeuralNet(numInputs = 150,numOutputs = 9,learningRate = 1.2, momentum = 0.5)
	nn.printArchitecture()

	small_train=[]
	small_label=[]
	small_train.append(traindata_list[0])
	small_train.append(traindata_list[1])
	small_label.append(trainLabel_list[0])
	small_label.append(trainLabel_list[1])
	nn.train(traindata_list, trainLabel_list, 50)
	nn.test(traindata_list, trainLabel_list)
	# #Zhiyuan end

if __name__ == '__main__':
	debug()

