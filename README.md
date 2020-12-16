# FACE-X

## Join official discord channel for discussion https://discord.gg/d5GfFfy8

### Facial recognition’s algorithms

There are several approaches for recognizing a face. The algorithm can use statistics, try to find a pattern which represents a specific person or use a convolutional neural network. 

The algorithms used for the tests are Eigenfaces, Fisherfacesand local binary patterns histograms which all come from the library OpenCV. Eigenfaces and Fisher faces are used with a Euclidean distance to predict the person. The algorithm which is using a deep convolutional neural network is the project called OpenFace.

This can be used for automatic face detection attendance system in recent technology.

Despite a variety of open-source face recognition frameworks available, there was no ready-made solution to implement. The available algorithms processed only high-resolution static shots and performed insufficiently.

### Requirements
- Python3.6+
- virtualenv (`pip install virtualenv`)

### Installation
- `virtualenvv env`
- `source venv/bin/activate` (Linux)
- `venv\Scripts\activate` (Windows)
- `pip install -r requirements.txt`
- Create an .env file, copy the content from .env.sample and add your data path. Example: `DATA_PATH = "./foto_reco/"`
