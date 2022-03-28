# Face Recognition

Recognize and manipulate faces from Python or from the command line with
the world's simplest face recognition library.
To recognize faces face_recognitation library has been used in this project.

## Identify faces in pictures

Recognize who appears in each photo.

## Installation

### Requirements

  * Python 3.3+ or Python 2.7
  * macOS or Linux (Windows not officially supported, but might work)

### Installation Options:

#### Installing on Mac or Linux

First, make sure you have dlib already installed with Python bindings:

  * [How to install dlib from source on macOS or Ubuntu](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)
  
Then, make sure you have cmake installed:  
 
```brew install cmake```
Finally, install this module from pypi using `pip3` (or `pip2` for Python 2):
```bash
pip3 install face_recognition
```

Alternatively, you can try this library with [Docker](https://www.docker.com/), see [this section](#deployment).

If you are having trouble with installation, you can also try out a
[pre-configured VM](https://medium.com/@ageitgey/try-deep-learning-in-python-now-with-a-fully-pre-configured-vm-1d97d4c3e9b).

#### Installing on Windows

While Windows isn't officially supported, helpful users have posted instructions on how to install this library:

  * [@masoudr's Windows 10 installation guide (dlib + face_recognition)](https://github.com/ageitgey/face_recognition/issues/175#issue-257710508)


## Usage

### Command-Line Interface

When you install `face_recognition`, you get two simple command-line 
programs:

* `face_recognition` - Recognize faces in a photograph or folder full for 
   photographs.
* `face_detection` - Find faces in a photograph or folder full for photographs.

#### `face_recognition` command line tool

The `face_recognition` command lets you recognize faces in a photograph or 
folder full  for photographs.

First, you need to provide a folder with one picture of each person you
already know. There should be one image file for each person with the
files named according to who is in the picture:

![known](https://cloud.githubusercontent.com/assets/896692/23582466/8324810e-00df-11e7-82cf-41515eba704d.png)

Next, you need a second folder with the files you want to identify:

![unknown](https://cloud.githubusercontent.com/assets/896692/23582465/81f422f8-00df-11e7-8b0d-75364f641f58.png)

Then in you simply run the command `face_recognition`, passing in
the folder of known people and the folder (or single image) with unknown
people and it tells you who is in each image:

```bash
$ face_recognition ./pictures_of_people_i_know/ ./unknown_pictures/
/unknown_pictures/unknown.jpg,Barack Obama
/face_recognition_test/unknown_pictures/unknown.jpg,unknown_person
```

There's one line in the output for each face. The data is comma-separated
with the filename and the name of the person found.

An `unknown_person` is a face in the image that didn't match anyone in
your folder of known people.

#### Python Module

You can import the `face_recognition` module and then easily manipulate
faces with just a couple of lines of code. It's super easy!

API Docs: [https://face-recognition.readthedocs.io](https://face-recognition.readthedocs.io/en/latest/face_recognition.html).

##### Automatically find all the faces in an image

```python
import face_recognition
image = face_recognition.load_image_file("my_picture.jpg")
face_locations = face_recognition.face_locations(image)
# face_locations is now an array listing the co-ordinates of each face!
```

See [this example](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture.py)
 to try it out.

You can also opt-in to a somewhat more accurate deep-learning-based face detection model.

Note: GPU acceleration (via NVidia's CUDA library) is required for good
performance with this model. You'll also want to enable CUDA support
when compliling `dlib`.

```python
import face_recognition
image = face_recognition.load_image_file("my_picture.jpg")
face_locations = face_recognition.face_locations(image, model="cnn")
# face_locations is now an array listing the co-ordinates of each face!
```

See [this example](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture_cnn.py)
 to try it out.

If you have a lot of images and a GPU, you can also
[find faces in batches](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_batches.py).


##### Recognize faces in images and identify who they are

```python
import face_recognition
picture_of_me = face_recognition.load_image_file("me.jpg")
my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]
# my_face_encoding now contains a universal 'encoding' of my facial features that can be compared to any other picture of a face!
unknown_picture = face_recognition.load_image_file("unknown.jpg")
unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]
# Now we can see the two face encodings are of the same person with `compare_faces`!
results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)
if results[0] == True:
    print("It's a picture of me!")
else:
    print("It's not a picture of me!")
```

See [this example](https://github.com/ageitgey/face_recognition/blob/master/examples/recognize_faces_in_pictures.py)
 to try it out.

