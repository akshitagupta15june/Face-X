# Face-Generation


# Procedure:
## Discriminator
Your first task will be to define the discriminator. This is a convolutional classifier like you've built before, only without any maxpooling layers. To deal with this complex data, it's suggested you use a deep network with normalization. You are also allowed to create any helper functions that may be useful.


## Generator
The generator should upsample an input and generate a new image of the same size as our training data 32x32x3. This should be mostly transpose convolutional layers with normalization applied to the outputs.


## Initialize the weights of your networks
To help your models converge, you should initialize the weights of the convolutional and linear layers in your model. From reading the original DCGAN paper, they say:

All weights were initialized from a zero-centered Normal distribution with standard deviation 0.02.

So, your next task will be to define a weight initialization function that does just this!

You can refer back to the lesson on weight initialization or even consult existing model code, such as that from the networks.py file in CycleGAN Github repository to help you complete this function.


## Build complete network
Define your models' hyperparameters and instantiate the discriminator and generator from the classes defined above. Make sure you've passed in the correct input arguments.

 
## Training
Training will involve alternating between training the discriminator and the generator. You'll use your functions real_loss and fake_loss to help you calculate the discriminator losses.

## Output

You should train the discriminator by alternating on real and fake images
Then the generator, which tries to trick the discriminator and should have an opposing loss function


It generates new human faces that look like this:  
![Image of Generated Faces](https://github.com/tfesenko/Face-Generation/blob/master/assets/Generated_faces2.png)

## Steps Involved
* Create a deeper model and use it to generate larger (say 128x128) images of faces.
* Read existing literature to see if you can use padding and normalization techniques to generate higher-resolution images.
* Implement a learning rate that evolves over time as they did in this [CycleGAN Github repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

makeup, as they did in [this paper](https://gfx.cs.princeton.edu/pubs/Chang_2018_PAS/Chang-CVPR-2018.pdf).
