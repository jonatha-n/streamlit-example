import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import euclidean_distances
"""
# Title Placeholder
"""

# Read in all data
df = pd.read_csv('spotify_songs_edit_2.csv')
df = df.drop_duplicates(subset=['track_name'])

# Read in non-numeric data for use in User-Interface
df_tracks = pd.read_csv('non_numeric_data.csv')
df_tracks = df_tracks[['track_artist', 'track_name']].drop_duplicates()
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

# df.loc[df['column_name'] == some_value]
