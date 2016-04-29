'''
Music Classification by Genre:
	Description

Date: 4/26/2016

Author(s): 
	Jorge Rojas
'''
import sys
import csv
import json
import thread
import neuralNet as NN
import spotifyAuth as sa
from optparse import OptionParser

def createDataFileFromSpotify(trainDataFileName):

	print "Creating CSV File..."
	csvfile = open(trainDataFileName,'wb')
	csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)

	tempo_normalization, loudness_normalization = getMaxMin()
	for track_id, track_sample in sa.getTrainData().iteritems():
		#Includes the track ID and the sample label
		isValid = validInput(track_sample)

		if isValid:
			#z_i = (x_i - min)/(max - min)
			track_sample[2] = (track_sample[2] - tempo_normalization[0])/(tempo_normalization[1] - tempo_normalization[0])
			track_sample[7] = (track_sample[7] - loudness_normalization[0])/(loudness_normalization[1] - loudness_normalization[0])
			csv_writer.writerow([track_id] + track_sample)

	print "CSV file successfully created"

def getTrainDataFromFile(trainDataFileName):
	train_data = []
	train_label = []

	print "Openning csv file..."
	csvfile = open(trainDataFileName, 'rb')
	csv_reader = csv.reader(csvfile)

	for row in csv_reader:
		#0th element is track id, which is not needed
		#last element is genre label
		isValid = validInput(row)
		if isValid:
			label = json.loads(row.pop())
			train_label.append( label )
			train_data.append( [float(feature) for feature in row[1:]] )

	return train_data, train_label

def getTrainDataFromSpotify():
	train_data = []
	train_label = []
	
	print "Reading Spotify's data..."
	tempo_normalization, loudness_normalization = getMaxMin()
	for track_id, track_sample in sa.getTrainData().iteritems():
		#DEBUGGING
		# if track_id == '63yKj3bpZUCWcJ4Xh6Ygl1': 
		# 	print track_id, '=> ', track_sample
		isValid = validInput(track_sample)

		if isValid:
			train_label.append( track_sample.pop() )
			#z_i = (x_i - min)/(max - min)
			track_sample[2] = (track_sample[2] - tempo_normalization[0])/(tempo_normalization[1] - tempo_normalization[0])
			track_sample[7] = (track_sample[7] - loudness_normalization[0])/(loudness_normalization[1] - loudness_normalization[0])
			train_data.append( track_sample )
	print train_data[0][2], train_data[0][7]
	return train_data, train_label

def getMaxMin():

	data_transpose = [[] for i in xrange(2)]
	for track_id, track_sample in sa.getTrainData().iteritems():
		isValid = validInput(track_sample)

		if isValid:
			data_transpose[0].append( track_sample[2] )
			data_transpose[1].append( track_sample[7] )
	return [min(data_transpose[0]), max(data_transpose[0])], [min(data_transpose[1]), max(data_transpose[1])]

def validInput(trainSample):
	try:
		none_index = trainSample.index(None)
		return 0
	except ValueError:
		return 1

def waitForAccessToken():
	access_token = sa.getAccessToken()
	while not access_token:
		access_token = sa.getAccessToken()
	# print 'Access token = %s'%access_token
	return access_token[0]

def createNetwork(createFile, trainFromFile):
	''' TRAIN DATA PROCESSING '''
	train_data_file_name = 'data/spotifyTrainData.csv'
	access_token = waitForAccessToken()
	sa.crawlSpotifyData(access_token)
	sa.labelSpotifyData(access_token)

	if createFile:
		createDataFileFromSpotify(train_data_file_name)
	else:
		if trainFromFile:
			train_data, train_label = getTrainDataFromFile(train_data_file_name)
		else:
			train_data, train_label = getTrainDataFromSpotify()

		''' NEURAL NETWORK '''
		nn = NN.NeuralNet(numInputs = 9,numOutputs = 4,learningRate = 1.2, momentum = 0.5)
		nn.printArchitecture()

		nn.train(train_data, train_label, num_epoch = 50)
		nn.test(train_data[:10], train_label[:10])

def main():
	parser = OptionParser()

	parser.add_option( "--ecsv",
					dest = "exportCSV",
					default = 0,
					help = "Flag to export spotify data to csv file. 1 for true, 0 for false. Default value is 0")

	parser.add_option( "--ofi",
					dest = "openFromFile",
					default = 0,
					help = "Flag to train neural network from csv file. 1 for true, 0 for false. Default value is 0")

	(options, args) = parser.parse_args()

	if isinstance(options.exportCSV, int):
		create_file = int(options.exportCSV)
	else:
		sys.exit('--csv argument was not an int value. Please type 0 or 1.')

	if isinstance(options.openFromFile, int):
		train_from_file = int(options.openFromFile)
	else:
		sys.exit('--csv argument was not an int value. Please type 0 or 1.')

	try:
		thread.start_new_thread(createNetwork, (create_file, train_from_file, ) )	#No arguments needed. Pass empty tuple
	except:
		print "Unable to start thread!"

	sa.runAuth()

if __name__ == '__main__':
	main()

