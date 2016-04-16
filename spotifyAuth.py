'''
Authorize Spotify API:
	Aquire access token given client ID and 
	client secret for project cpe695-music-classification.
	API used to aquire the music dataset.

Date: 04/04/2016

Author(s): 
	Jorge Rojas

Genres to be considered: 
		Trance
		Indie Rock
		Jazz
		Pop

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
from optparse import OptionParser
from flask import Flask, abort, request

''' VARIABLES '''
train_data = []
access_token = ''

''' CONSTANTS '''
USER_ID = '1248308979'
SPOTIFY_API_URL = 'https://api.spotify.com/'
CLIENT_ID = 'f9c78ea0cd2347f1bff7843192120461'	#Visit https://developer.spotify.com to create your ID and Secret!
CLIENT_SECRET = 'f1320710dfe842bb9c228d4dbcefa881'

#This URI can be set in the spotify developer console. Make it your localHost/callback add it and save it
REDIRECT_URI = 'http://127.0.0.1:5000/spotify_callback'
SPOTIFY_PLAYLISTS = {'training':'2VB8ds8bjD78gVHRsCcMTl', 'testing':'6etBG7ccLcMhQb4nUud9UE'}
SPOTIFY_API_ENDPOINTS = {'audio_features': 'v1/audio-features/', 
						'track': 'v1/tracks/%s', 
						'playlists': 'v1/users/' + USER_ID + '/playlists/'}

''' UTILITY FUNCTIONS'''
def crawl_spotify_data(access_token):
	''' Get Training Playlist '''
	playlist_url = SPOTIFY_API_URL + SPOTIFY_API_ENDPOINTS['playlists'] + SPOTIFY_PLAYLISTS['training']
	print playlist_url	#DEBUGGING
	res = curl(playlist_url, authToken = access_token)
	# print res 	#DEBUGGING
	res_json = json.loads(res)
	print json.dumps(res_json, indent=4, sort_keys=True)	#DEBUGGING

	''' Get Training Tracks '''
	tracks_url = res_json['tracks']['href']
	print "TRACK URL => %s"%tracks_url[:tracks_url.index('&')]	#DEBUGGING
	res = curl(tracks_url[:tracks_url.index('&')], authToken = access_token)
	res_json = json.loads(res)
	tracks_json = res_json['items']

	''' Get Track Features'''
	for i, track in enumerate(tracks_json):
		if i > 10: break

		audio_features_url = SPOTIFY_API_URL + SPOTIFY_API_ENDPOINTS['audio_features'] + track['track']['id']
		# print audio_features_url 	#DEBUGGING
		res = curl(audio_features_url, authToken = access_token)
		res_json = json.loads(res)
		temp_feature_vector = [res_json['energy'], res_json['liveness'], res_json['tempo'], 
							res_json['speechiness'], res_json['acousticness'], res_json['instrumentalness'], 
							res_json['danceability'], res_json['loudness'], res_json['valence']]
		train_data.append(temp_feature_vector)

	print "Number of training samples: %s"%len(train_data)
	print train_data		

def curl( url, data = None, authToken = None ):

	if data is not None:
		req = urllib2.Request( url, data )
	else:
		req = urllib2.Request( url )

	if authToken is not None:
		req.add_header( 'Authorization', 'Bearer %s'%authToken )

	response = urllib2.urlopen( req )
	res = response.read()
	return res

def get_token(code):
	client_auth = requests.auth.HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET)
	post_data = {"grant_type": "authorization_code", "code": code, "redirect_uri": REDIRECT_URI}
	response = requests.post("https://accounts.spotify.com/api/token", auth=client_auth, data=post_data)
	token_json = response.json()
	return token_json["access_token"]

def is_valid_state(state):
	return True

def save_created_state(state):
	pass

''' OAUTH2 PROCESS '''
app = Flask(__name__)
@app.route('/')
def homepage():
	text = '<a href="%s">Authenticate with Spotify</a>'
	return text % make_authorization_url()

def make_authorization_url():
	# Generate a random string for the state parameter
	# Save it for use later to prevent xsrf attacks
	from uuid import uuid4
	state = str(uuid4())
	save_created_state(state)
	params = {"client_id": CLIENT_ID,
			  "response_type": "code",
			  "state": state,
			  "redirect_uri": REDIRECT_URI,
			  "duration": "temporary"}
	url = "https://accounts.spotify.com/authorize?" + urllib.urlencode(params)
	return url

@app.route('/spotify_callback')
def spotify_callback():
	error = request.args.get('error', '')
	if error:
		return "Error: " + error
	state = request.args.get('state', '')
	if not is_valid_state(state):
		# Uh-oh, this request wasn't started by us!
		abort(403)
	code = request.args.get('code')
	# We'll change this next line in just a moment
	access_token = get_token(code)
	crawl_spotify_data(access_token)
	return "got an access token! %s" % access_token

def runAuth():
	app.run(debug=True, port=5000)

if __name__ == '__main__':
	runAuth()

