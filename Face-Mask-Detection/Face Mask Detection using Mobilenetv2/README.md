
Detecting face mask with OpenCV and TensorFlow. Using simple CNN or model provided by TensorFlow as MobileNetV2, VGG16, Xception.


## Data

Raw data collected from kaggle and script `crawl_image.py`, split to 'Mask' and 'Non Mask' class.

Using `build_data.py` to extract faces from raw dataset and resize to 64x64.

## Installation

Clone the repo

```
git clone git@github.com:ksvbka/face-mask-detector.git
```
cd to project folder and create virtual env

```
virtualenv .env
source .env/bin/activate
pip install -r requirements.txt
```

Download raw dataset and execute script build_dataset.py to preprare dataset for training
```
cd data
bash download_data.sh
cd -
python3 build_dataset.py --data-dir data/dataset_raw/ --output-dir data/64x64_dataset
```
## Training

Execute `train.py` script and pass  network architecture type dataset dir and epochs to it.
Default network type is MobileNetV2.
```
python3 train.py --net-type MobileNetV2 --data-dir data/64x64_dataset --epochs 20
```


## Output :
![evaluation](https://user-images.githubusercontent.com/65017645/120110276-722afe80-c18a-11eb-8189-837d122a3c00.png)

![8](https://user-images.githubusercontent.com/65017645/120110323-a2729d00-c18a-11eb-8ea2-5bff0036ad40.jpg)


