import streamlit as st
import webbrowser
from streamlit_extras.app_logo import add_logo
from streamlit_player import st_player

page_bg_img = """
<style>
div.stButton > button:first-child {
    all: unset;
    width: 300px;
    height: 60px;
    font-size: 32px;
    background: transparent;
    border: none;
    position: relative;
    color: #f0f0f0;
    cursor: pointer;
    z-index: 1;
    padding: 10px 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    white-space: nowrap;
    user-select: none;
    -webkit-user-select: none;
    touch-action: manipulation;

}
div.stButton > button:before, div.stButton > button:after {
    content: '';
    position: absolute;
    bottom: 0;
    right: 0;
    z-index: -99999;
    transition: all .4s;
}

div.stButton > button:before {
    transform: translate(0%, 0%);
    width: 100%;
    height: 100%;
    background: #1a0000;
    border-radius: 10px;
}
div.stButton > button:after {
  transform: translate(10px, 10px);
  width: 35px;
  height: 35px;
  background: #ffffff15;
  backdrop-filter: blur(5px);
  -webkit-backdrop-filter: blur(5px);
  border-radius: 50px;
}

div.stButton > button:hover::before {
    transform: translate(5%, 20%);
    width: 110%;
    height: 110%;
}


div.stButton > button:hover::after {
    border-radius: 10px;
    transform: translate(0, 0);
    width: 100%;
    height: 100%;
}

div.stButton > button:active::after {
    transition: 0s;
    transform: translate(0, 5%);
}

[data-testid="stAppViewContainer"] {
background-image: url("https://images.unsplash.com/photo-1584303450431-5a0312051758?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=687&q=80");
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
st.title("❤️ Vibescape-Youtube")
st.markdown(''':red[**Note : It is recommended that you scan your face , for Vibescape to groove with you!**]''')

st.sidebar.success("Youtube has been selected as your music player.")
st.sidebar.text("Developed by Shambhavi")


if "run" not in st.session_state:
    st.write("**Looks like you have skipped the face scan on the homepage and came here, just for music, just choose your vibe manually for Vibescape to groove with you!**")
    option = st.selectbox(
    'What''s your vibe today?',
    ('Happy', 'Sad', 'Angry','Fear','Surprise','Neutral'))
    if option == "Happy":
        st.session_state["run"] = "Happy"
    elif option == "Sad":
        st.session_state["run"] = "Sad"
    elif option == "Angry":
        st.session_state["run"] = "Angry"
    elif option == "Fear":
        st.session_state["run"] = "Fear"
    elif option == "Surprise":
        st.session_state["run"] = "Surprise"
    else:
        st.session_state["run"] = "Neutral"
else:
    st.write("You current emotion is: " , st.session_state["run"])
    
col1, col2 = st.columns(2)

with col1:
    hindi = st.button("Hindi")
    if hindi:
        if st.session_state["run"] == "Happy":
            st_player("https://www.youtube.com/watch?v=OcmcptbsvzQ")
        elif st.session_state["run"] == "Sad":
           st_player("https://www.youtube.com/watch?v=e6LMUGrN57A")
        elif st.session_state["run"] == "Angry":
            st_player("https://www.youtube.com/playlist?list=PLxNm0dqHxmlupV3dr7uq4Rl8L5nwlGKQA")
        elif st.session_state["run"] == "Fear":
            st_player("https://www.youtube.com/watch?v=8joIt7wsayU")
        elif st.session_state["run"] == "Surprise":
            st_player("https://www.youtube.com/watch?v=rtTI1rh9U5M")
        elif st.session_state["run"] == "Neutral":
            st_player("https://www.youtube.com/watch?v=EVF_AuhJgLg")
        else:
            st_player("hhttps://www.youtube.com/watch?v=S_TW9h7vUB8&list=RDEMyQ80_Idfp-UOKusrpqoh-g&start_radio=1")

    bengali = st.button("Bengali")
    if bengali:
        if st.session_state["run"] == "Happy":
            st_player("https://www.youtube.com/playlist?list=PLzxTmXYDR-xXsilCSwT-Bds5w96k2GOXF")
        elif st.session_state["run"] == "Sad":
           st_player("https://www.youtube.com/watch?v=RRbqeGr8mJE")
        elif st.session_state["run"] == "Angry":
            st_player("https://www.youtube.com/watch?v=_RBlE6Ar8mw")
        elif st.session_state["run"] == "Fear":
            st_player("https://www.youtube.com/watch?v=5PtVfECix1M")
        elif st.session_state["run"] == "Surprise":
            st_player("https://www.youtube.com/watch?v=39uMLYTh40Q")
        elif st.session_state["run"] == "Neutral":
            st_player("https://www.youtube.com/watch?v=39uMLYTh40Q")
        else:
            st_player("https://www.youtube.com/watch?v=5TJqoxsoXc4")
     
    marathi = st.button("Marathi")
    if marathi:
        if st.session_state["run"] == "Happy":
            st_player("https://www.youtube.com/playlist?list=PLN_pFG_Bv6D4-gpRS06J2auY87DQaEDuo")
        elif st.session_state["run"] == "Sad":
           st_player("https://www.youtube.com/playlist?list=PLpjbqr-x3QIpcfrn_McAvi80s5bGuWr9f")
        elif st.session_state["run"] == "Angry":
            st_player("https://www.youtube.com/playlist?list=PLpjbqr-x3QIpw79fcOQJDi3FdMcYwWZki")
        elif st.session_state["run"] == "Fear":
            st_player("https://soundcloud.com/miss_happy/sets/hindi-songs")
        elif st.session_state["run"] == "Surprise":
            st_player("https://www.youtube.com/watch?v=imUUogeGUio")
        elif st.session_state["run"] == "Neutral":
            st_player("https://www.youtube.com/watch?v=xUSTF4Gyj8M")
        else:
            st_player("https://www.youtube.com/watch?v=xUSTF4Gyj8M")
   
with col2:
   english = st.button("English")
   if english:
        if st.session_state["run"] == "Happy":
            st_player("https://www.youtube.com/watch?v=ONM4uXLe8CU")
        elif st.session_state["run"] == "Sad":
           st_player("https://www.youtube.com/watch?v=BwleOnML1Fo")
        elif st.session_state["run"] == "Angry":
            st_player("https://www.youtube.com/watch?v=YnezhNgzdss")
        elif st.session_state["run"] == "Fear":
            st_player("https://www.youtube.com/watch?v=G-YNNJIe2Vk")
        elif st.session_state["run"] == "Surprise":
            st_player("https://www.youtube.com/watch?v=zStYh2eHOWk")
        elif st.session_state["run"] == "Neutral":
            st_player("https://www.youtube.com/watch?v=KNaCJYX-mgY")
        else:
            st_player("https://www.youtube.com/watch?v=KNaCJYX-mgY")
            
   punjabi = st.button("Punjabi")
   if punjabi:
        if st.session_state["run"] == "Happy":
            st_player("https://www.youtube.com/watch?v=heXb7XQYVKo")
        elif st.session_state["run"] == "Sad":
           st_player("https://www.youtube.com/watch?v=8Jq0E9ryVEc")
        elif st.session_state["run"] == "Angry":
            st_player("https://www.youtube.com/watch?v=Pha76iGaauM")
        elif st.session_state["run"] == "Fear":
            st.write("No such playlist found , hence default playlist is being played.")
            st_player("https://www.youtube.com/watch?v=Id2fc96XPYE")
        elif st.session_state["run"] == "Surprise":
            st_player("https://www.youtube.com/watch?v=Id2fc96XPYE")
        elif st.session_state["run"] == "Neutral":
            st_player("https://www.youtube.com/watch?v=QbXLf5QvNUM")
        else:
            st_player("https://www.youtube.com/watch?v=QbXLf5QvNUM")
            
   telugu = st.button("Telugu")
   if telugu:
        if st.session_state["run"] == "Happy":
            st_player("https://www.youtube.com/playlist?list=PLMpEfaKcGjpWEgNtdnsvLX6LzQL0UC0EM")
        elif st.session_state["run"] == "Sad":
           st_player("https://www.youtube.com/watch?v=yvnsgYQyGXo")
        elif st.session_state["run"] == "Angry":
            st_player("https://www.youtube.com/watch?v=CYj6li9939w")
        elif st.session_state["run"] == "Fear":
            st.write("No such playlist found , hence default playlist is being played.")
            st_player("https://www.youtube.com/playlist?list=PL4sNEU2Mgm6bNnbM-qKPmTwwDromFqpMQ")
        elif st.session_state["run"] == "Surprise":
            st_player("https://www.youtube.com/watch?v=ADqgenM4uqQ")
        elif st.session_state["run"] == "Neutral":
            st_player("https://www.youtube.com/playlist?list=PL4sNEU2Mgm6bNnbM-qKPmTwwDromFqpMQ")
        else:
            st.write("No such playlist found , hence default playlist is being played.")
            st_player("https://www.youtube.com/playlist?list=PL4sNEU2Mgm6bNnbM-qKPmTwwDromFqpMQ")