# Face-X GUI Applications

## Video Face Recognition

<p align="center">
  <img src="https://github.com/shireenchand/Face-X/blob/face_rec/Face-X-GUI-applications/Video-Face-Recognition/Media/meida4.gif" width="400">
</p>

What if you could know whether a person is present in a video or not, without seeing the video. That is exactly what's done here.
Just give an input video alongwith a few photos of the person/s you want to recognize and voila! The video is displayed along with the name and bounding boxes around the face to be recognized. Adding to that, if a face doesn't match with the input face/s, it gives it the tag "Unknown".

This application has very minimal dependencies.

### This task is acheived by the following steps:

- Detect Faces
- Compute 128-d face embeddings to quantify a face
- Train a Support Vector Machine (SVM) on top of the embeddings
- Recognize faces in the video stream
