import tensorflow as tf
import numpy as np
import pre_trained_facenet
import cv2
from mtcnn.mtcnn import MTCNN
import glob

for i in range(4):
  camera = cv2.VideoCapture(0)
  return_value, camshot = camera.read()
  cv2.imwrite('captured/cam.jpg', camshot)
  detector = MTCNN()
  del(camera)
  # some constants kept as default from facenet
  minsize = 20
  threshold = [0.6, 0.7, 0.7]
  factor = 0.709
  margin = 44
  input_image_size = 160

  sess = tf.Session()



  # read 20170512-110547 model file downloaded from https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk
  pre_trained_facenet.load_model("20170512-110547/20170512-110547.pb")

  # Get input and output tensors
  images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
  embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
  phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
  embedding_size = embeddings.get_shape()[1]

  def getFace(img):
    faces=[]
    result = detector.detect_faces(img)
    # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
    bounding_box = result[0]['box']
    keypoints = result[0]['keypoints']
    cv2.rectangle(img,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0,155,255),
                  2)
    cv2.circle(img,(keypoints['left_eye']), 2, (0,155,255), 2)
    cv2.circle(img,(keypoints['right_eye']), 2, (0,155,255), 2)
    cv2.circle(img,(keypoints['nose']), 2, (0,155,255), 2)
    cv2.circle(img,(keypoints['mouth_left']), 2, (0,155,255), 2)
    cv2.circle(img,(keypoints['mouth_right']), 2, (0,155,255), 2)
    cv2.imwrite("check/dd_drawn.jpg", img)
    cropped=img[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]]
    cv2.imwrite("check/dd_cropped.jpg", cropped)
    rearranged= cv2.resize(cropped, (input_image_size,input_image_size),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("check/dd_resized.jpg",rearranged)
    prewhitened = pre_trained_facenet.prewhiten(rearranged)
    faces.append({'face':rearranged,'embedding':getEmbedding(prewhitened)})
    return faces
  def getEmbedding(resized):
    reshaped = resized.reshape(-1,input_image_size,input_image_size,3)
    feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
    embedding = sess.run(embeddings, feed_dict=feed_dict)
    return embedding

  def compare2face(img1,img2):
    face1 = getFace(img1)
    face2 = getFace(img2)
    if face1 and face2:
        # calculate Euclidean distance
        dist = np.sqrt(np.sum(np.square(np.subtract(face1[0]['embedding'], face2[0]['embedding']))))
        return dist
    return -1


  files = (glob.glob('database'+'/*.jpg'))
  for f in files:
    img1 = cv2.imread(f)
  img2 = cv2.imread("captured/cam.jpg")
  distance=compare2face(img1, img2)
  threshold = 0.9    # set yourself to meet your requirement
  print("distance = "+str(distance))
  print("Result = " + ("same person" if distance <= threshold else "not same person"))
  if(distance <= threshold):
    break
  print("Photocapture Remaining"+str(3-i)+"times")
