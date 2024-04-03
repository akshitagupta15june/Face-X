# :zap: De-Blurring Model Deployed Website 

I have deployed a image processing model by packing their weights in a file and convert it into a interactive website for De-Blurring the input images with streamlit framework as the frontend.

### :zap: Tech Stack Used 

#### :zap: Front-end 
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=for-the-badge&logo=Streamlit&logoColor=white)

#### :zap: Back-end 
![Python](https://img.shields.io/badge/python-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white)

![Line](https://user-images.githubusercontent.com/85225156/171937799-8fc9e255-9889-4642-9c92-6df85fb86e82.gif)

### :zap: Key features of this website are:-

1. Frontend features:
   - Input: A image and the name and age of the user.
   - Well maintained landing page of streamlit framework.

2. Backend features:
   - A POST request is generated using the form in `app.py` python file using streamlit framework.
   - The image is passed to the model which generate the sharp image.
   - Finally, rendering all the generated image and the input image on the webpage.

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
streamlit run app.py
```

![Line](https://user-images.githubusercontent.com/85225156/171937799-8fc9e255-9889-4642-9c92-6df85fb86e82.gif)

### :zap: Results 

#### :zap: De Blur CNN Model - 1

<table>
  <tr>
    <td><img src="./images/De-Blur-Model-Accuracy.png" alt="Accuracy"></td>
    <td><img src="./images/De-Blur-Model-Loss.png" alt="Loss"></td>
  </tr>
  <tr>
    <td colspan="2" style="text-align: center;"><img src="./images/deblur_result.png" alt="Deblur Result"></td>
  </tr>
</table>

#### :zap: Auto encoder CNN Model - 2

<table>
  <tr>
    <td><img src="./images/Autoencoder-Model-Accuracy.png" alt="Accuracy"></td>
    <td><img src="./images/Autoencoder-Model-Loss.png" alt="Loss"></td>
  </tr>
  <tr>
    <td colspan="2" style="text-align: center;"><img src="./images/autoencoder_result.png" alt="Autoencoder Result"></td>
  </tr>
</table>

### :zap: Predictions Done by Website 

<table>
  <tr>
    <td><img src="./images/sample_0.jpg" alt="0" style="width:300px;"></td>
    <td><img src="./images/output_image_0.png" alt="0" style="width:300px;"></td>
    <td><img src="./images/clear_0.jpg" alt="0" style="width:300px;"></td>
  </tr>
  <tr>
    <td><img src="./images/sample_169.jpg" alt="169" style="width:300px;"></td>
    <td><img src="./images/output_image_169.png" alt="169" style="width:300px;"></td>
    <td><img src="./images/clear_169.jpg" alt="169" style="width:300px;"></td>
  </tr>
  <tr>
    <td><img src="./images/sample_1004.jpg" alt="1004" style="width:300px;"></td>
    <td><img src="./images/output_image_1004.png" alt="1004" style="width:300px;"></td>
    <td><img src="./images/clear_1004.jpg" alt="1004" style="width:300px;"></td>
  </tr>
</table>


### :zap: Preview of Website 

![screenshot](./images/Screenshot.jpeg)


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
