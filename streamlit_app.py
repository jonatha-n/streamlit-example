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

df = pd.read_csv('spotify_songs_edit.csv')
df = df.drop_duplicates(subset=['track_name'])
df_tracks = pd.read_csv('non_numeric_data.csv')
df_tracks = df_tracks[['track_artist', 'track_name']].drop_duplicates()
df_tracks['track_and_artist'] = df_tracks['track_name'] + " --- by " + df_tracks['track_artist']

box_disabled = True

input_artist_name = st.selectbox('Type a song name and make selection', df_tracks[['track_and_artist']], help="Not all artists are available. Please make a selection from the listed choices.")

# df.loc[df['column_name'] == some_value]
# num_points = st.slider("Number of points in spiral", 1, 10000, 1100)
# num_turns = st.slider("Number of turns in spiral", 1, 300, 31)
