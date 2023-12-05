import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import euclidean_distances
"""
# NewMusic Song Recommender
"""

# Read in all data
df = pd.read_csv('spotify_songs_edit_2.csv')
df = df.drop_duplicates(subset=['track_name', 'track_artist'])
#print(len(df)) # Shows rows with unique track_name and track_artist remain
#df.info() # Validate that all values are non-null and not missing

# Read in non-numeric data for use in User-Interface
df_tracks = pd.read_csv('non_numeric_data.csv')
df_tracks = df_tracks[['track_name', 'track_artist']].drop_duplicates()
df_tracks['track_and_artist'] = df_tracks['track_name'] + " --- by " + df_tracks['track_artist']

# Select Box
input_artist_name = st.selectbox('Type a song name and make selection', df_tracks[['track_and_artist']], help="Not all artists are available. Please make a selection from the listed choices.")

# Attributes that will be used to create clusters
att = df[['energy', 'key', 'valence', 'acousticness', 'speechiness']]
scaler = StandardScaler()
attributes_scaled = scaler.fit_transform(att)
num_clusters = 6

# Use KMeans to determine clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
df['cluster'] = kmeans.fit_predict(attributes_scaled)
df['cluster'] = df['cluster'].astype('category')

#Generate Similar songs
st.button("Generate Similar Songs", type="primary")
song_and_artist_list = input_artist_name.split(' --- by ')

search = df.query('track_name == @song_and_artist_list[0] and track_artist == @song_and_artist_list[1]')

liked_song_att = search[['energy', 'key', 'valence', 'acousticness', 'speechiness']]
liked_song_att_scaled = scaler.transform(liked_song_att)
predict_cluster = kmeans.predict(liked_song_att_scaled)
songs_in_same_cluster = df[df['cluster'] == predict_cluster[0]]

songs_in_same_cluster_att = songs_in_same_cluster[['energy', 'key', 'valence', 'acousticness', 'speechiness']]
songs_in_same_cluster_att_scaled = scaler.transform(songs_in_same_cluster_att)
distances = euclidean_distances(liked_song_att_scaled, songs_in_same_cluster_att_scaled)

songs_in_same_cluster['distance_to_liked'] = distances[0]
ranked_songs = songs_in_same_cluster.sort_values(by='distance_to_liked')
top_ten_ranked = ranked_songs.iloc[1:11, [1,2,3,10]]
top_ten_ranked['Ranked Similarity'] = range(1, 11)
top_ten_ranked.set_index('Ranked Similarity', inplace=True)
st.dataframe(top_ten_ranked, use_container_width=True)
# df.loc[df['column_name'] == some_value]

# Fix column names in table
# Display histograms
# Make button actually work
# Start with no selections
