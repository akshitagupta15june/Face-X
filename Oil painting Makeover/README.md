# Awesome Face Operation: Oil Painting Makeover
This filter involves detecting the person, 
and applying an Oil Painting Effect on the face

This is and image processing script that utilizes HaarCascades
and Mediapipe's selfie segmentation model.

How it works:

1. The selfie segmentation model is loaded from Mediapipe.

2. HaarCascades from opencv are loaded

3. Image is loaded from the directory

4. The frame size is obtained, and the background image is loaded.

5. The the image is processed:

   a. The segmentation mask is extracted using selfie segmentation.

   b. A condition mask is created based on the threshold value.

   c. The Oil painting effect is applied across the original image

   d. The condition mask is used to merge the background of the original 
      image and the face from the oil effect.

6. The final result (`output_image`) is converted back to RGB and saved as "oil_effect.png".

## Sample

![output](oil_effect.png)

## Getting Started

* Clone this repository.
```bash
  git clone https://github.com/akshitagupta15june/Face-X.git
```
* Navigate to the required directory.
```bash
  cd Awesome-face-operations/Oil\ painting\ Makeover/

```
* Install the Python dependencies.

```bash
  pip install -r requirements.txt
```
* Run the script.
```bash
  python oil.py
```

## Author

[Abir-Thakur](https://github.com/Inferno2211)

