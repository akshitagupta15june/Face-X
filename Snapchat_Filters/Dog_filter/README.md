## Build a Dog Filter with Computer Vision

- A real-time filter that adds dog masks to faces on a live feed.
- A dog filter that responds to your emotions. (Couldn't find a pug mask, so I used a cat.) A generic dog for smiling "happy", a dalmation for frowning "sad", and a cat for dropped jaws "surprise".
- Utilities used for portions of the understanding, such as plotting and advesarial example generation.
- Ordinary least squares and ridge regression models using randomized features.

![step_8_emotion_dog_mask](https://user-images.githubusercontent.com/2068077/34196964-36383d58-e519-11e7-92dc-2d7c33ab29bd.gif)

# Getting Started

> (Optional) [Setup a Python virtual environment](https://www.digitalocean.com/community/tutorials/common-python-tools-using-virtualenv-installing-with-pip-and-managing-packages#a-thorough-virtualenv-how-to) with Python 3.6.

1. Install all Python dependencies.

```
pip install -r requirements.txt
```

2. Navigate into `src`.

```
cd src
```

3. Launch the script for an emotion-based dog filter:

```
python step_8_dog_emotion_mask.py
```
