from flask import Flask, render_template, request, flash, redirect
from Model import model
import os
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def prediction():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']
        name = request.form.get('name')
        
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        
        if file:
            cartoon_file_path = os.path.join(UPLOAD_FOLDER, 'cartoon-pic.jpg')
            if os.path.exists(cartoon_file_path):
                os.remove(cartoon_file_path)
            
            original_file_path = os.path.join(UPLOAD_FOLDER, 'original-pic.jpg')
            if os.path.exists(original_file_path):
                os.remove(original_file_path)
            
            original_image_path = os.path.join(UPLOAD_FOLDER, 'original-pic.jpg')
            file.save(original_image_path)
            
            cartoon_image = model.cartoonify_image(original_image_path)
            cartoon_image_path = os.path.join(UPLOAD_FOLDER, 'cartoon-pic.jpg')
            cv2.imwrite(cartoon_image_path, cartoon_image)
            
            return render_template('prediction.html', name=name)

if __name__ == '__main__':
    app.run(debug=True)
