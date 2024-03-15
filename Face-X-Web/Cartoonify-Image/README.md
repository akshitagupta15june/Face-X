# :zap: Cartoonification Website 

I have created a website for Cartoonify the input images with flask framework as the backend.

### :zap: Tech Stack Used 

#### :zap: Front-end 
![HTML5](https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/css3-%231572B6.svg?style=for-the-badge&logo=css3&logoColor=white)
![Bootstrap](https://img.shields.io/badge/bootstrap-%238511FA.svg?style=for-the-badge&logo=bootstrap&logoColor=white)

#### :zap: Back-end 
![Python](https://img.shields.io/badge/python-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)

![Line](https://user-images.githubusercontent.com/85225156/171937799-8fc9e255-9889-4642-9c92-6df85fb86e82.gif)

### :zap: Key features of this website are:-

1. Frontend features:
  - Taking the input image and name.
  - Having landing page and separate prediction page.

2. Backend features:
  - A POST request is generated using the form in `index.html`. The image is saved in the `upload` folder.
  - Finally, rendering all the prediction and the input image on the webpage.

### :zap: Method of Process 
- Input:
  - Upload the image as provide in the form render on `index.html` file.
- Process:
  - Store the uploaded image in `static/upload` directory with the name of `original-pic.jpg`.
  - Model has scan the image and create the cartoonified image.
  - Also save that Cartoonified image with the name of `cartoon-pic.jpg`.
- Output:
  - Finally, `prediction.html` file will render the both the images.

![Line](https://user-images.githubusercontent.com/85225156/171937799-8fc9e255-9889-4642-9c92-6df85fb86e82.gif)

### :zap: Quick Start 
- Clone this repository
```css
git clone https://github.com/<Your-GitHub-UserName>/Face-X.git
```
- Install python version 3.9 or higher versions.
- Install Virtual Environment using the below command line.
```css
python -m venv <virtual-environment-name>
```
- Activate the Environment in command line.
```css
.\<virtual-environment-name>\Scripts\activate
```
- If error occurs, use the following commands.
```css
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
- In activated environment, install all python dependencies.
```css
pip install -r requirements.txt
```
- Run code file using below command.
```css
python app.py
```

![Line](https://user-images.githubusercontent.com/85225156/171937799-8fc9e255-9889-4642-9c92-6df85fb86e82.gif)

### :zap: Output Screenshot 

![screenshot](./static/images/Screenshot.png)

### :zap: Preview of Website 

![recording](https://github.com/akshitagupta15june/Face-X/assets/114330097/f488def2-05b1-48f5-9872-4933f86da98a)


![Line](https://user-images.githubusercontent.com/85225156/171937799-8fc9e255-9889-4642-9c92-6df85fb86e82.gif)

## ❤️ Project Contributor 

<h4 align='center'>Developed By <b><i>Avdhesh Varshney</i></b> 👦</h4>
<p align='center'>
  <a href='https://www.linkedin.com/in/avdhesh-varshney'>
    <img src='https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white' />
  </a>
  <a href='https://www.github.com/Avdhesh-Varshney'>
    <img src='https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white' />
  </a>
</p>
