import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

"""
# Title Placeholder

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""
df = pd.read_csv('spotify_songs_edit.csv')
df = df.drop_duplicates(subset=['track_name'])
df_names = pd.read_csv('non_numeric_data.csv')
list_of_artists = df_names[['track_artist']].drop_duplicates()

box_disabled = True

input_artist_name = st.selectbox('Enter an artist name or band', list_of_artists)
input_track_name = st.selectbox('Enter a track name', ['a', 'b', 'c'], disabled=box_disabled)

# df.loc[df['column_name'] == some_value]
# num_points = st.slider("Number of points in spiral", 1, 10000, 1100)
# num_turns = st.slider("Number of turns in spiral", 1, 300, 31)
