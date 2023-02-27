# Author Name: Sarthak Kapaliya
# Date: 1/11/2023
# Description: This file contains the flask app and the routes for the web app. 

from facex import face_mask
import cv2
import numpy as np
import os
import flask
from flask import Flask, render_template, request
import numpy as np
from PIL import Image as im
app = Flask(__name__)

# routes
@app.route("/")
def main():
    return render_template("home.html")


@app.route("/templates/simulator", methods=['GET', 'POST'])
def index():
    return render_template("simulator.html")


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    """
    Desc: A POST request is generated using the form in simulator.html.
    The image is saved in the static folder and the mask is detected on the face. 
    Finally we render all the prediction and mask detection on the face on the webpage.

    input: We take image path

    output: We store it and detect the mask on the face. 
    The mask detection is shown on the webpage.
    A Result is shown on the webpage.
    """
    if request.method == 'POST':
        img = request.files['file']
        img_path = "static\\" + img.filename
        img.save(img_path)
        image = face_mask(img_path)
        cv2.imwrite(img_path, image)
        p = "No Mask Detected" if np.sum(image) == 0 else "Mask Detected"  


    return render_template("simulator.html", prediction=p, img_path=img_path)


if __name__ == '__main__':
    app.run(port=90,debug=True)
