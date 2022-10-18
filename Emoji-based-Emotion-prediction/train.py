# %%
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

# %%
train_data_path = "data_and_model/data/"
img_size = 224
batch_size = 8
EPOCHS = 30

labels = os.listdir(train_data_path)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)


train_data_gen = datagen.flow_from_directory(
    train_data_path,
    target_size=(img_size, img_size),
    color_mode='rgb',
    classes=None,
    class_mode='categorical',
    subset='training',
    batch_size=batch_size,
    shuffle=True,
    seed=None,
    save_to_dir=None,
    save_prefix='',
    save_format='png',
    follow_links=False,
    interpolation='nearest'
)


val_data_gen = datagen.flow_from_directory(
    train_data_path,
    target_size=(img_size, img_size),
    color_mode='rgb',
    classes=None,
    class_mode='categorical',
    subset='validation',
    batch_size=batch_size,
    shuffle=True,
    seed=None,
    save_to_dir=None,
    save_prefix='',
    save_format='png',
    follow_links=False,
    interpolation='nearest'
)

# %%
from tensorflow.keras.applications import MobileNetV2

base_model = MobileNetV2(
            input_shape=(img_size, img_size,3),
            include_top=False,
            weights='imagenet')

#2. Locks all the base model's weights
for layer in base_model.layers:
    layer.trainable = False

#3. Adds a few layers to the end of the model
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(len(os.listdir(train_data_path)), activation='softmax')(x)


new_model = tf.keras.Model(base_model.input, x)


try:
    os.mkdir("data_and_model/model")
except:
    pass


model_dir = "data_and_model/model/Model.hdf5"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_dir,
    save_best_only=True,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max')

new_model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

new_model.summary()

# %%
#4. Trains the new model
history = new_model.fit(
    train_data_gen,
    epochs=EPOCHS,
    callbacks=[model_checkpoint_callback],
    validation_data = val_data_gen
)

# %%
img = load_img('data_and_model/test_data/images (32).jpg', target_size = (img_size, img_size))
img = img_to_array(img)
img = np.expand_dims(img, axis = 0)

plt.imshow(load_img('data_and_model/test_data/images (32).jpg', target_size = (img_size, img_size)))
plt.title(labels[np.argmax(new_model.predict(img))])

# %%
test_data_path = 'data_and_model/test_data/'

fig, axs = plt.subplots(2,4, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)

axs = axs.ravel()

for i in range(len(os.listdir(test_data_path))):
    tp = os.path.join(test_data_path, os.listdir(test_data_path)[i])
    
    img = load_img(tp, target_size = (img_size, img_size))
    img = img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    
    
    

    axs[i].imshow(load_img(tp, target_size = (img_size, img_size)))
    axs[i].set_title(labels[np.argmax(new_model.predict(img))])

# %%


# %%


# %%
import numpy as np
from tensorflow.keras.models import load_model



saved_model_path = "data_and_model/model/Model.hdf5"

img_path = 'data_and_model/test_data/images (32).jpg'

loaded_model = load_model(saved_model_path)

img = load_img(img_path, target_size = (224, 224))
img = img_to_array(img)
img = np.expand_dims(img, axis = 0)

plt.imshow(load_img(img_path, target_size = (224, 224)))
plt.title(labels[np.argmax(loaded_model.predict(img))])
plt.show()

# %%



