Implementation of Deepface Module to enhance emotion detection. it is increasing the accuracy along with adding more emotions to detect.

Accuracy = 97%
Trained over 4 million images by Facebook
Additional parameter of confidence percentage
Gives dominant emotion along with percentage of all emotions
Able to recognise emotions of multiple faces from a single image
![Screenshot 2024-05-28 003506](https://github.com/Nancyjikadra/Face-X/assets/108074513/a7f17e51-6f0f-4373-94c4-88d7be99f87d)

Screenshots
Results = [{'emotion': {'angry': 6.07668232124459e-08, 'disgust': 1.0169186632464069e-16, 'fear': 1.1963964502846973e-08, 'happy': 99.96163845519561, 'sad': 2.964388381100604e-07, 'surprise': 5.324124506074761e-06, 'neutral': 0.038352074419087244}, 'dominant_emotion': 'happy', 'region': {'x': 398, 'y': 212, 'w': 59, 'h': 59, 'left_eye': None, 'right_eye': None}, 'face_confidence': 0.93},

{'emotion': {'angry': 0.044974877259894935, 'disgust': 3.866733671934057e-06, 'fear': 0.05902597209874128, 'happy': 54.59856336133917, 'sad': 0.18385377616315768, 'surprise': 0.2438808034949537, 'neutral': 44.86969472531143}, 'dominant_emotion': 'happy', 'region': {'x': 671, 'y': 286, 'w': 126, 'h': 126, 'left_eye': (758, 340), 'right_eye': (714, 332)}, 'face_confidence': 0.9},

{'emotion': {'angry': 0.029080110834911466, 'disgust': 0.03639304486569017, 'fear': 5.474400892853737, 'happy': 9.535442292690277, 'sad': 0.9942812845110893, 'surprise': 0.10400796309113503, 'neutral': 83.82639288902283}, 'dominant_emotion': 'neutral', 'region': {'x': 992, 'y': 322, 'w': 51, 'h': 51, 'left_eye': None, 'right_eye': None}, 'face_confidence': 0.94}]


