#85 

**The  process**:
- Using the OpenCV detecting the face in the face
- For every Face I had to detect the landmarks on the face using the 68 facial landmark coordinates
- Using the coordinates I had to detect the coordinates 29(top_nose), 30(center_nose), 31(left_nose) and 35(right_nose)
- After detecting the coordinates, reduce the size of the pig_nose into that area
- and then mask the pig_nose into the actual frame

The Facial Landmark Coordinates
![facemarks points](https://user-images.githubusercontent.com/55532999/103528913-cb785580-4eaa-11eb-8258-ce73e09824fb.png)

The screenshot with the filter
![234](https://user-images.githubusercontent.com/55532999/103529108-14300e80-4eab-11eb-903e-3e9a6c95f032.PNG)
