#import os
import streamlit as st
import pandas as pd

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import polarplot
import songrecommendations

#from dotenv import load_dotenv
#load_dotenv()

SPOTIPY_CLIENT_ID = 'fcbb0f29936b4f8ba90c4a606e516a67'
SPOTIPY_CLIENT_SECRET = '9b66b9a7c4ec4f2ba5fe21b6575da778'

auth_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

from streamlit_extras.app_logo import add_logo
from streamlit_player import st_player

page_bg_img = """
<style>
div.stButton > button:first-child {
    width: 150px;
    height: 60px;
    padding: 0.3em 1em;
    border: none;
    outline: none;
    color: rgb(255, 255, 255);
    background: #1a1a1a;
    cursor: pointer;
    position: relative;
    z-index: 0;
    border-radius: 10px;
    user-select: none;
    -webkit-user-select: none;
    touch-action: manipulation;
}

div.stButton > button:before {
    content: "";
  background: linear-gradient(
    45deg,
    #ff0000,
    #ff7300,
    #fffb00,
    #48ff00,
    #00ffd5,
    #002bff,
    #7a00ff,
    #ff00c8,
    #ff0000
  );
  position: absolute;
  top: -2px;
  left: -2px;
  background-size: 400%;
  z-index: -1;
  filter: blur(5px);
  -webkit-filter: blur(5px);
  width: calc(100% + 4px);
  height: calc(100% + 4px);
  animation: glowing-button 20s linear infinite;
  transition: opacity 0.3s ease-in-out;
  border-radius: 10px;
}

@keyframes glowing-button {
    0% {
        background-position: 0 0;
        }
    50% {
        background-position: 400% 0;
        }
    100% {
        background-position: 0 0;
        }
}

div.stButton > button:after {
    z-index: -1;
    content: "";
    position: absolute;
    width: 100%;
    height: 100%;
    background: #1a1a1a;
    left: 0;
    top: 0;
    border-radius: 10px;
}

[data-testid="stAppViewContainer"] {
background-image: url("https://images.unsplash.com/photo-1597773150796-e5c14ebecbf5?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1470&q=80");
background-size: cover;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}

[data-testid="stSidebar"] > div:first-child {
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
background : black;
}

[data-testid="stHeader"] {
background: rgba(0,0,0,0);
}

[data-testid="stToolbar"] {
right: 2rem;
}
</style>
"""
add_logo("https://github.com/NebulaTris/vibescape/blob/main/logo.png?raw=true")

st.markdown(page_bg_img, unsafe_allow_html=True)

st.header('Analyzer & Recommender 📊')


search_choices = ['Song/Track', 'Artist']
search_selected = st.sidebar.selectbox("Your search choice please: ", search_choices)

search_keyword = st.text_input(search_selected + " (Keyword Search)")
button_clicked = st.button("Search")


search_results = []
tracks = []
artists = []
albums = []
if search_keyword is not None and len(str(search_keyword)) > 0:
    if search_selected == 'Song/Track':
        st.write("Start song/track search")
        tracks = sp.search(q='track:'+ search_keyword,type='track', limit=20)
        tracks_list = tracks['tracks']['items']
        if len(tracks_list) > 0:
            for track in tracks_list:
                #st.write(track['name'] + " - By - " + track['artists'][0]['name'])
                search_results.append(track['name'] + " - By - " + track['artists'][0]['name'])
        
    elif search_selected == 'Artist':
        st.write("Start artist search")
        artists = sp.search(q='artist:'+ search_keyword,type='artist', limit=20)
        artists_list = artists['artists']['items']
        if len(artists_list) > 0:
            for artist in artists_list:
                # st.write(artist['name'])
                search_results.append(artist['name'])
 
selected_artist = None
selected_track = None
if search_selected == 'Song/Track':
    selected_track = st.selectbox("Select your song/track: ", search_results)
elif search_selected == 'Artist':
    selected_artist = st.selectbox("Select your artist: ", search_results)


if selected_track is not None and len(tracks) > 0:
    tracks_list = tracks['tracks']['items']
    track_id = None
    if len(tracks_list) > 0:
        for track in tracks_list:
            str_temp = track['name'] + " - By - " + track['artists'][0]['name']
            if str_temp == selected_track:
                track_id = track['id']
                track_album = track['album']['name']
                img_album = track['album']['images'][1]['url']
                #songrecommendations.save_album_image(img_album, track_id)
    selected_track_choice = None            
    if track_id is not None:
        #image = songrecommendations.get_album_mage(track_id)
       # st.image(image)
        track_choices = ['Song Features', 'Similar Songs Recommendation']
        selected_track_choice = st.sidebar.selectbox('Please select track choice: ', track_choices)        
        if selected_track_choice == 'Song Features':
            track_features  = sp.audio_features(track_id) 
            df = pd.DataFrame(track_features, index=[0])
            df_features = df.loc[: ,['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence']]
            st.dataframe(df_features)
            polarplot.feature_plot(df_features)
        elif selected_track_choice == 'Similar Songs Recommendation':
            token = songrecommendations.get_token(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET)
            similar_songs_json = songrecommendations.get_track_recommendations(track_id, token)
            recommendation_list = similar_songs_json['tracks']
            recommendation_list_df = pd.DataFrame(recommendation_list)
            # st.dataframe(recommendation_list_df)
            recommendation_df = recommendation_list_df[['name', 'explicit', 'duration_ms', 'popularity']]
            st.dataframe(recommendation_df)
            # st.write("Recommendations....")
            songrecommendations.song_recommendation_vis(recommendation_df)
            
    else:
        st.write("Please select a track from the list")       

st.sidebar.text("Developed by Shambhavi")
# elif selected_album is not None and len(albums) > 0:
#     albums_list = albums['albums']['items']
#     album_id = None
#     album_uri = None    
#     album_name = None
#     if len(albums_list) > 0:
#         for album in albums_list:
#             str_temp = album['name'] + " - By - " + album['artists'][0]['name']
#             if selected_album == str_temp:
#                 album_id = album['id']
#                 album_uri = album['uri']
#                 album_name = album['name']
#     if album_id is not None and album_uri is not None:
#         st.write("Collecting all the tracks for the album :" + album_name)
#         album_tracks = sp.album_tracks(album_id)
#         df_album_tracks = pd.DataFrame(album_tracks['items'])
#         # st.dataframe(df_album_tracks)
#         df_tracks_min = df_album_tracks.loc[:,
#                         ['id', 'name', 'duration_ms', 'explicit', 'preview_url']]
#         # st.dataframe(df_tracks_min)
#         for idx in df_tracks_min.index:
#             with st.container():
#                 col1, col2, col3, col4 = st.columns((4,4,1,1))
#                 col11, col12 = st.columns((8,2))
#                 col1.write(df_tracks_min['id'][idx])
#                 col2.write(df_tracks_min['name'][idx])
#                 col3.write(df_tracks_min['duration_ms'][idx])
#                 col4.write(df_tracks_min['explicit'][idx])   
#                 if df_tracks_min['preview_url'][idx] is not None:
#                     col11.write(df_tracks_min['preview_url'][idx])  
#                     with col12:   
#                         st.audio(df_tracks_min['preview_url'][idx], format="audio/mp3")                            
                        
                        
if selected_artist is not None and len(artists) > 0:
    artists_list = artists['artists']['items']
    artist_id = None
    artist_uri = None
    selected_artist_choice = None
    if len(artists_list) > 0:
        for artist in artists_list:
            if selected_artist == artist['name']:
                artist_id = artist['id']
                artist_uri = artist['uri']
    
    if artist_id is not None:
        artist_choice = ['Albums', 'Top Songs']
        selected_artist_choice = st.sidebar.selectbox('Select artist choice', artist_choice)
                
    if selected_artist_choice is not None:
        if selected_artist_choice == 'Albums':
            artist_uri = 'spotify:artist:' + artist_id
            album_result = sp.artist_albums(artist_uri, album_type='album') 
            all_albums = album_result['items']
            col1, col2, col3 = st.columns((6,4,2))
            for album in all_albums:
                col1.write(album['name'])
                col2.write(album['release_date'])
                col3.write(album['total_tracks'])
        elif selected_artist_choice == 'Top Songs':
            artist_uri = 'spotify:artist:' + artist_id
            top_songs_result = sp.artist_top_tracks(artist_uri)
            for track in top_songs_result['tracks']:
                with st.container():
                    col1, col2, col3, col4 = st.columns((4,4,2,2))
                    col11, col12 = st.columns((10,2))
                    col21, col22 = st.columns((11,1))
                    col31, col32 = st.columns((11,1))
                    col1.write(track['id'])
                    col2.write(track['name'])
                    if track['preview_url'] is not None:
                        col11.write(track['preview_url'])  
                        with col12:   
                            st.audio(track['preview_url'], format="audio/mp3")  
                    with col3:
                        def feature_requested():
                            track_features  = sp.audio_features(track['id']) 
                            df = pd.DataFrame(track_features, index=[0])
                            df_features = df.loc[: ,['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence']]
                            with col21:
                                st.dataframe(df_features)
                            with col31:
                                polarplot.feature_plot(df_features)
                            
                        feature_button_state = st.button('Track Audio Features', key=track['id'], on_click=feature_requested)
                    with col4:
                        def similar_songs_requested():
                            token = songrecommendations.get_token(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET)
                            similar_songs_json = songrecommendations.get_track_recommendations(track['id'], token)
                            recommendation_list = similar_songs_json['tracks']
                            recommendation_list_df = pd.DataFrame(recommendation_list)
                            recommendation_df = recommendation_list_df[['name', 'explicit', 'duration_ms', 'popularity']]
                            with col21:
                                st.dataframe(recommendation_df)
                            with col31:
                                songrecommendations.song_recommendation_vis(recommendation_df)

                        # similar_songs_state = st.button('Similar Songs', key=track['id'], on_click=similar_songs_requested)
                    st.write('----')
