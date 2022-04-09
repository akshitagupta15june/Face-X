import streamlit as st
import cv2
import numpy as np
from getEmbeddings import calc_Embeddings
from recognizer import recognize
from detectVideo import detect
import tempfile


# all_files = []
# names = []

if 'names' not in st.session_state:
    st.session_state['names'] = []

if 'all_files' not in st.session_state:
	st.session_state['all_files'] = []
try:
	video = st.file_uploader("Upload video",type=["mp4"],key="video",accept_multiple_files=False,)
	if video != None:
		tfile = tempfile.NamedTemporaryFile(delete=False) 
		tfile.write(video.read())

		num = st.slider("No of people to detect",min_value=1,max_value=5,)
		total = 1
		for i in range(num):
			name = st.text_input("Enter name",key=total)
			uploaded_files = st.file_uploader(f"Upload Images(Minimum 6) for {name}", type=None, key=total,accept_multiple_files=True, disabled=False)
			if total < num:
				if st.button("Confirm",key=total):
					if len(uploaded_files) < 6 :
						st.header("NOT ENOUGH IMAGES")
					else:
						st.session_state['all_files'].append(uploaded_files)
						st.session_state['names'].append(name)
						# st.session_state['names'] = names
						# st.session_state['all_files'] = all_files
		 

			else:
				if st.button("Confirm",key=name):
					if len(uploaded_files) < 6 :
						st.header("NOT ENOUGH IMAGES")
					else:
						st.session_state['all_files'].append(uploaded_files)
						st.session_state['names'].append(name)
						# st.session_state['names'] = names
						# st.session_state['all_files'] = all_files

						print(st.session_state['names'])
						print(len(st.session_state['all_files']))

						embeddings,names = calc_Embeddings(st.session_state['all_files'],st.session_state['names'])

						le,model = recognize(embeddings,names)

						detect(tfile.name,model,le)


			total += 1

except:
	st.write("SEEMS LIKE AN ERROR IS ENCOUNTERED. RELOAD THE PAGE")








# print(opencv_image.shape)
# for file in files_uploaded:
# 	print(file.shape())