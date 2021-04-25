import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils import generic_utils
import tensorflow as tf
from tensorflow.keras.optimizers import Adadelta
from tensorflow.python.keras.engine.network import Network
from glob import glob
from utils import Data
from model import generator_network, discriminator_network

# get the data of celeba
filenames = glob('img_align_celeba/img_align_celeba/*')
print(len(filenames))

# define hyper parameters
input_shape = (256, 256, 3)
local_shape = (128, 128, 3)
batch_size = 4
n_epoch = 200
tc = int(n_epoch * 0.18)
td = int(n_epoch * 0.02)
alpha = 0.0004

# define models and optimizer
generator = generator_network(input_shape)
discriminator = discriminator_network(input_shape, local_shape)
optimizer = Adadelta()

# build model
# build Completion Network model
org_img = layers.Input(shape=input_shape)
mask = layers.Input(shape=(input_shape[0], input_shape[1], 1))

in_img = layers.Lambda(lambda x: x[0] * (1 - x[1]),
                       output_shape=input_shape)([org_img, mask])
imitation = generator(in_img)
completion = layers.Lambda(lambda x: x[0] * x[2] + x[1] * (1 - x[2]),
                           output_shape=input_shape)([imitation, org_img, mask])
completion_container = Network([org_img, mask], completion)
completion_out = completion_container([org_img, mask])
completion_model = Model([org_img, mask], completion_out)
completion_model.compile(loss='mse',
                         optimizer=optimizer)
completion_model.summary()

# build Discriminator model
in_pts = layers.Input(shape=(4,), dtype='int32')
d_container = Network([org_img, in_pts], discriminator([org_img, in_pts]))
d_model = Model([org_img, in_pts], d_container([org_img, in_pts]))
d_model.compile(loss='binary_crossentropy',
                optimizer=optimizer)
d_model.summary()

d_container.trainable = False

# build Discriminator & Completion Network models
all_model = Model([org_img, mask, in_pts],
                  [completion_out, d_container([completion_out, in_pts])])
all_model.compile(loss=['mse', 'binary_crossentropy'],
                  loss_weights=[1.0, alpha], optimizer=optimizer)
all_model.summary()

# checkpoints
checkpoint_dir = 'drive/MyDrive/ColabNotebooks/GANS/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=optimizer,
                                 discriminator_optimizer=optimizer,
                                 generator=generator_network,
                                 discriminator=discriminator_network)

data_train = filenames[:3000]


def train_model(result_dir="result", data=data_train):

    train_datagen = Data(data_train, input_shape[:2], local_shape[:2])

    for n in range(n_epoch):
        progbar = generic_utils.Progbar(len(train_datagen))
        for inputs, points, masks in train_datagen.flow(batch_size):
            completion_image = completion_model.predict([inputs, masks])
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            g_loss = 0.0
            d_loss = 0.0
            if n < tc:
                g_loss = completion_model.train_on_batch([inputs, masks], inputs)
            else:
                d_loss_real = d_model.train_on_batch([inputs, points], valid)
                d_loss_fake = d_model.train_on_batch([completion_image, points], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                if n >= tc + td:
                    g_loss = all_model.train_on_batch([inputs, masks, points],
                                                      [inputs, valid])
                    g_loss = g_loss[0] + alpha * g_loss[1]
            progbar.add(inputs.shape[0], values=[("D loss", d_loss), ("G mse", g_loss)])

        num_img = min(5, batch_size)
        fig, axs = plt.subplots(num_img, 3)
        for i in range(num_img):
            axs[i, 0].imshow(inputs[i] * (1 - masks[i]))
            axs[i, 0].axis('off')
            axs[i, 0].set_title('Input')
            axs[i, 1].imshow(completion_image[i])
            axs[i, 1].axis('off')
            axs[i, 1].set_title('Output')
            axs[i, 2].imshow(inputs[i])
            axs[i, 2].axis('off')
            axs[i, 2].set_title('Ground Truth')
        fig.savefig(os.path.join(result_dir, "result_%d.png" % n))
        plt.close()

        # Save the checkpoints every 10 epochs
        if (n + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

    # save model
    generator.save(os.path.join("model", "generator.h5"))
    completion_model.save(os.path.join("model", "completion.h5"))
    discriminator.save(os.path.join("model", "discriminator.h5"))


def main():
    train_model()


if __name__ == "__main__":
    main()
