#!/usr/bin/env bash

python yoloface.py \
    --model-cfg './cfg/yolov3-face.cfg' \
    --model-weights './model-weights/yolov3-wider_16000.weights' \
    --image './samples/yolo_face_test.jpg' \
    --output-dir './outputs'
