#!/usr/bin/env bash

W_FILENAME=yolov3-wider_16000.weights.zip
H5_FILENAME=YOLO_Face.h5.zip

if [ ! -d "./model-weights" ]; then
    mkdir -p ./model-weights;
fi

cd model-weights

# Download yoloface models
echo "*** Downloading the trained models..."

wget --load-cookies /tmp/cookies.txt -r "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=13gFDLFhhBqwMw6gf8jVUvNDH2UrgCCrX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=13gFDLFhhBqwMw6gf8jVUvNDH2UrgCCrX" -O $W_FILENAME && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt -r "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1a_pbXPYNj7_Gi6OxUqNo_T23Dt_9CzOV' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1a_pbXPYNj7_Gi6OxUqNo_T23Dt_9CzOV" -O $H5_FILENAME && rm -rf /tmp/cookies.txt

# Unzip
unzip -q $W_FILENAME
unzip -q $H5_FILENAME

# Delete .zip files
rm -rf $W_FILENAME $H5_FILENAME

echo "*** All done!!!"
