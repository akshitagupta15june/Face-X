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
    if request.method == 'POST':
        img = request.files['file']

        img_path = "static/" + img.filename
        img.save(img_path)
        image = face_mask(img_path)

    return render_template("simulator.html", prediction=p, img_path=img_path)


if __name__ == '__main__':
    app.run(debug=True)
