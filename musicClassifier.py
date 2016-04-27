'''
Music Classification by Genre:
	Description

Date: 4/26/2016

Author(s): 
	Jorge Rojas
'''
import csv
import thread
import neuralNet as NN
import spotifyAuth as sa

def createNetwork(createFile):
	''' TRAIN DATA PROCESSING '''

	''' NEURAL NETWORK '''

def main():

	#Flag to create csv from spotify data
	create_file = 0
	try:
		thread.start_new_thread(createNetwork, (create_file, ) )
	except:
		print "Unable to start thread!"
	sa.runAuth()

if __name__ == '__main__':
	main()