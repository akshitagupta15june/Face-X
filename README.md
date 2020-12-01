# FACE-X

## Join official discord channel for discussion https://discord.gg/d5GfFfy8

Recognition of faces by model training using OPENCV

We will first train the model by our faces and then use recognition system.

I have recorded 300 faces of mine in part 1 and trained the model using part 2 then final recognition is done in part 3.

I have added a LBPHFaceRecognizer for recognizing faces.

This can be used for automatic face detection attendance system in recent technology.

Despite a variety of open-source face recognition frameworks available, there was no ready-made solution to implement. The available algorithms processed only high-resolution static shots and performed insufficiently.

facial_recognition part 3 is the actual one for recognition.

### Requirements
- Python3.6+
- virtualenv (`pip install virtualenv`)

### Installation
- `virtualenvv env`
- `source venv/bin/activate` (Linux)
- `venv\Scripts\activate` (Windows)
- `pip install -r requirements.txt`
- Create an .env file, copy the content from .env.sample and add your data path. Example: `DATA_PATH = "./foto_reco/"`

### Execution
- `python facial_recognition_part1.py` (face images collection)
- `python facial_recognition_part2.py` (training)
- `python facial_recognition_part3.py` (final recognition)



