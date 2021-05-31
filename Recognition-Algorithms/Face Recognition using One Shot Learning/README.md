Face recognition application using inception model
Its a basic face recognizer application which can identify the face(s) of the person(s) showing on a web cam. The implementation is inspired by two path breaking papers on facial recognition using deep convoluted neural network, namely FaceNet and DeepFace.
I have used pre trained model Keras-OpenFace which is an open source Keras implementation of the OpenFace (Originally Torch implemented)
The pretrained model that I have used is by Victor Sy Wang’s implementation and was loaded using his code: https://github.com/iwantooxxoox/Keras-OpenFace.

One Shot Learning

In one shot learning, only one image per person is stored in the database which is passed through the neural network to generate an embedding vector. This embedding vector is compared with the vector generated for the person who has to be recognized. If there exist similarities between the two vectors then the system recognizes that person, else that person is not there in the database. This can be understood by below picture.

Triplet Loss Function

Here we are using OpenFace pre-trained model for facial recognition. Without going into much details on how this neural network identify two same faces, let’s say that the model is trained on a large set of face data with a loss function which groups identical images together and separate non-identical faces away from each other. Its also known as triplet loss function.