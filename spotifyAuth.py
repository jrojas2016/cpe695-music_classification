'''
Authorize Spotify API:
	Aquire access token given client ID and 
	client secret for project cpe695-music-classification.
	API used to aquire the music dataset.

	output -> [indieRock%, jazz%, trance%, pop%] *all four outputs are between 0 and 1

Date: 04/04/2016

Author(s): 
	Jorge Rojas

Genres to be considered: 
		EDM
		Indie Rock
		Jazz
		Reggae

Track Audio Features Example:
	{
		u'track_href': u'https://api.spotify.com/v1/tracks/2f1sMZx3jWEkswEHq6HFgV', 
		u'analysis_url': u'http://echonest-analysis.s3.amazonaws.com/TR/wL2hQwAeevmp7w-NPnDY72TdDuOnd3wxp_fVL3eK5T-Fs5o11vMbhiRZw0zQdsL4Xf5cq0yXg9uoWd0Xw=/3/full.json?AWSAccessKeyId=AKIAJRDFEY23UEVW42BQ&Expires=1459987283&Signature=szV90VZUBfLvhHxtTJKs0vhNvBs%3D', 
		u'energy': 0.885, u'liveness': 0.0869, u'tempo': 155.02, u'speechiness': 0.0375, 
		u'uri': u'spotify:track:2f1sMZx3jWEkswEHq6HFgV', u'acousticness': 4.22e-05, 
		u'instrumentalness': 0.00333, u'time_signature': 4, u'danceability': 0.467, 
		u'key': 1, u'duration_ms': 214861, u'loudness': -4.852, u'valence': 0.439, 
		u'type': u'audio_features', u'id': u'2f1sMZx3jWEkswEHq6HFgV', u'mode': 1
	}
'''
import json
import requests
import requests.auth
import urllib, urllib2
from flask import Flask, abort, request

''' VARIABLES '''
test_data = {}
train_data = {}
access_token = []

''' CONSTANTS '''
USER_ID = '1248308979'
SPOTIFY_API_URL = 'https://api.spotify.com/'
CLIENT_ID = 'XXXXXXXXXX'	#Visit https://developer.spotify.com to create your ID and Secret!
CLIENT_SECRET = 'XXXXXXXXXX'

#This URI can be set in the spotify developer console. Make it your localHost/callback add it and save it
REDIRECT_URI = 'http://127.0.0.1:5000/spotify_callback'
#Needed to successfully label the training data
SPOTIFY_TRAINING_PLAYLISTS = '2VB8ds8bjD78gVHRsCcMTl'
GENRE_PLAYLISTS = {
				'0XUlpafP8eIlIWt3VHSd7q':[1, 0, 0, 0],
				'5O2ERf8kAYARVVdfCKZ9G7':[0, 1, 0, 0],
				'6XChIaijnUBzPDrQOX02AJ':[0, 1, 0, 0],
				'6yPRSeVGw7IK7vDiXuXgr4':[0, 1, 0, 0],
				'570tVSLPdLnRU0z0bqd8Wk':[0, 1, 0, 0],
				'04MJzJlzOoy5bTytJwDsVL':[0, 0, 1, 0],
				'5ILSWr90l2Bgk89xuhsysy':[0, 0, 1, 0],
				'6vaaau50gPDmKcjDrs4pA2':[0, 0, 0, 1],
				'0ifGUu1vx6PVcCASyG3t8m':[0, 0, 0, 1],
				'779MortitnXMMJnVSAcOOT':[0, 0, 0, 1],
				'1JCZJ9vKg2r8eBaBLz14MT':[0, 0, 0, 1],
				'43SNeW90KHzTpEgaK0OATU':[0, 0, 0, 1]
				}
				
SPOTIFY_API_ENDPOINTS = {	
					'audio_features': 'v1/audio-features?ids=', 
					'track': 'v1/tracks/%s', 
					'playlists': 'v1/users/' + USER_ID + '/playlists/',
					'playlist_tracks': 'v1/users/spotify/playlists/%s/tracks'
						}

''' GET FUNCTION '''
def getAccessToken():
	return access_token

def getTrainData():
	return train_data

def getTestData():
	return test_data

''' UTILITY FUNCTIONS '''
def isValidState(state):
	return True

def saveCreatedState(state):
	pass

def getToken(code):
	client_auth = requests.auth.HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET)
	post_data = {"grant_type": "authorization_code", "code": code, "redirect_uri": REDIRECT_URI}
	response = requests.post("https://accounts.spotify.com/api/token", auth=client_auth, data=post_data)
	token_json = response.json()
	return token_json["access_token"]

def curl(url, data = None, authToken = None):

	if data is not None:
		req = urllib2.Request( url, data )
	else:
		req = urllib2.Request( url )

	if authToken is not None:
		req.add_header( 'Authorization', 'Bearer %s'%authToken )

	response = urllib2.urlopen( req )
	res = response.read()
	return json.loads(res)

def makeAuthorizationUrl():
	# Generate a random string for the state parameter
	# Save it for use later to prevent xsrf attacks
	from uuid import uuid4
	state = str(uuid4())
	saveCreatedState(state)
	params = {"client_id": CLIENT_ID,
			  "response_type": "code",
			  "state": state,
			  "redirect_uri": REDIRECT_URI,
			  "duration": "temporary"}
	url = "https://accounts.spotify.com/authorize?" + urllib.urlencode(params)
	return url

def crawlSpotifyData(accessToken):
	''' Get Training Playlist '''
	playlist_url = SPOTIFY_API_URL + SPOTIFY_API_ENDPOINTS['playlists'] + SPOTIFY_TRAINING_PLAYLISTS
	# print playlist_url	#DEBUGGING
	res_json = curl(playlist_url, authToken = accessToken)
	# print res 	#DEBUGGING

	''' Get Training Tracks '''
	tracks_url = res_json['tracks']['href']
	# print "TRACK URL => %s"%tracks_url	#DEBUGGING
	while tracks_url is not None:
		res_json = curl(tracks_url, authToken = accessToken)
		tracks_json = res_json['items']
		tracks_url = res_json['next']	#url for next "page" of tracks. Continue until null
		# print len(tracks_json)	#DEBUGGING

		''' Get Track Features'''
		track_ids = ''
		for i, track in enumerate(tracks_json):
			# if i > 10: break
			track_ids += track['track']['id'] + ','
		track_ids = track_ids[:-1]	#remove last comma
		audio_features_url = SPOTIFY_API_URL + SPOTIFY_API_ENDPOINTS['audio_features'] + track_ids
		# print audio_features_url 	#DEBUGGING
		res_json = curl(audio_features_url, authToken = accessToken)
		# print res_json	#DEBUGGING
		track_features = res_json['audio_features']
		# print track_features	#DEBUGGING
		for features in track_features:
			if features is not None:
				temp_feature_vector = [features['energy'], 
										features['liveness'], features['tempo'], features['speechiness'], 
										features['acousticness'], features['instrumentalness'], 
										features['danceability'], features['loudness'], features['valence']]
				train_data[features['id']] = temp_feature_vector
			else:
				pass

	print "Number of training samples: %s"%len(train_data)

def labelSpotifyData(accessToken):
	checked_tracks = []
	missing_tracks = []
	genre_song_count = [0, 0, 0, 0]

	for playlist_id, genre in GENRE_PLAYLISTS.iteritems():
		playlist_tracks_url = SPOTIFY_API_URL + SPOTIFY_API_ENDPOINTS['playlist_tracks']%playlist_id

		while playlist_tracks_url is not None:
			res_json = curl(playlist_tracks_url, authToken = accessToken)
			tracks_json = res_json['items']		#list of track objects
			playlist_tracks_url = res_json['next']	#url for next "page" of tracks. Continue until null
			# print len(tracks_json)	#DEBUGGING

			''' Get Track Features'''
			for track in tracks_json:
				track_id = track['track']['id']

				try:
					if track_id not in checked_tracks:
						train_data[track_id].append(genre)
						genre_song_count[genre.index(1)] += 1
				except KeyError:
					missing_tracks.append(track_id)

				checked_tracks.append(track_id)

	print "Number of missing tracks = %s"%len(missing_tracks)
	print "Genre Count:\nIndie Rock = %s"%genre_song_count[0]
	print "Jazz = %s"%genre_song_count[1]
	print "EDM = %s"%genre_song_count[2]
	print "Reggae = %s"%genre_song_count[3]
	print "Labeled sample: ", train_data['43BLqP9em5cm0F8CWeDTfz']
	print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
	# print missing_tracks	#DEBUGGING

''' OAUTH2 PROCESS '''
app = Flask(__name__)
@app.route('/')
def homepage():
	text = '<a href="%s">Authenticate with Spotify</a>'
	return text % makeAuthorizationUrl()

@app.route('/spotify_callback')
def spotify_callback():
	error = request.args.get('error', '')
	if error:
		return "Error: " + error
	state = request.args.get('state', '')
	if not isValidState(state):
		# Uh-oh, this request wasn't started by us!
		abort(403)
	code = request.args.get('code')
	access_token.append(getToken(code))
	return "got an access token! %s" % access_token[0]

def runAuth():
	app.run(debug=True, port=5000)

if __name__ == '__main__':
	runAuth()

